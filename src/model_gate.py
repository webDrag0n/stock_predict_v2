"""模型晋升门控：候选（staging）vs 冠军（champion）的公平回测对比与判定。

触发时机：每晚 predict_tomorrow / webapp train_predict 训练完候选后，比较
候选配方指纹与 champion 指纹（model_store.fingerprint_of）：

  · 无 champion        → 先跑单配方回放记录回测预期，再首次引导晋升
                           （reason="bootstrap"；回放失败只告警，不阻断）；
  · 指纹相同           → 同一配方的滚动重训，原位刷新 champion 工件，不评估；
  · 指纹不同           → 完整门控：
      1. 候选落盘 models/staging/<model_id>/；
      2. evaluate_pair：两个配方各自跑一遍最近 PROMOTE_EVAL_DAYS 个交易日的
         完整每日重放（run_daily_replay 正常逐日重训路径，非冻结模型——冻结
         artifact 回放会让训练窗更晚的候选在窗口前段处于样本内，不诚实）；
      3. should_promote：候选 Sharpe > champion Sharpe + PROMOTE_SHARPE_MARGIN，
         候选最大回撤不劣于 champion 回撤 − PROMOTE_DD_TOLERANCE，且双方日收益
         配对差值的 Newey–West t ≥ PROMOTE_MIN_T（旧调用不传收益时跳过）；
      4. 通过 → promote（旧 champion 归档 archive/）；不通过 → 候选留档
         staging + evaluation.json，champion 不变。

两遍重放只有 ensemble 配方参数（腿/超参）不同，决策/成本/风控参数完全一致，
且各自只用自己的成熟标签逐日重训——与生产同分布、无泄露。

另提供 stage_and_evaluate：候选落盘 staging + 双回放评估的独立编排，
auto_promote=False 时绝不晋升（pipeline train 模式用，晋升交人工）；
run_gate 的"配方不同"分支复用它（auto_promote=True）。
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
import model_store
from backtest_daily import run_daily_replay
from perf_metrics import compute_performance, scalar_metrics

logger = logging.getLogger(__name__)

# 配方参数键：champion manifest config_snapshot → run_daily_replay params
_RECIPE_KEYS = (
    "ensemble_models",
    "n_seeds",
    "incr_trees",
    "rebuild_days",
    "train_window",
    "ic_window",
)


def _recipe_params(snapshot: dict) -> dict:
    """从 config_snapshot 提取 run_daily_replay 认得的配方参数子集。

    ensemble_models 统一转成 list：调用处有 tuple(p["ensemble_models"]) 的
    写法，字符串会被逐字符拆开。
    """
    out = {k: snapshot[k] for k in _RECIPE_KEYS if k in snapshot}
    v = out.get("ensemble_models")
    if isinstance(v, str):
        out["ensemble_models"] = [m.strip() for m in v.split(",") if m.strip()]
    return out


def _eval_window(
    panel: pd.DataFrame, eval_days: int, warmup: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """最近 eval_days 个决策日的 [start, end]（与 run_daily_replay 同口径推决策日）。"""
    fw = int(getattr(C, "FORWARD_WINDOW", 20))
    embargo = int(getattr(C, "EMBARGO_DAYS", fw))
    all_dates = np.sort(panel["date"].unique())
    decision_dates = all_dates[warmup + embargo : -1]
    if len(decision_dates) == 0:
        raise ValueError("历史不足以预热，无法门控评估")
    start = decision_dates[-eval_days] if len(decision_dates) > eval_days else decision_dates[0]
    return pd.Timestamp(start), pd.Timestamp(decision_dates[-1])


def _replay_metrics(
    panel: pd.DataFrame,
    feat_cols: list[str],
    params: dict,
    start_date,
    end_date,
    initial_cash: float,
    lags: int,
    progress: bool,
) -> dict:
    """单配方完整重放，返回标量指标与内存态 strat_ret 日收益列表。"""
    res = run_daily_replay(
        panel,
        feat_cols,
        initial_cash=initial_cash,
        params=params,
        start_date=start_date,
        end_date=end_date,
        progress=progress,
    )
    perf = compute_performance(res["daily"], panel, res["meta"]["initial_cash"], lags=lags)
    metrics = scalar_metrics(perf)
    metrics["strat_ret"] = np.asarray(perf["strat_ret"], dtype=float).tolist()
    return metrics


def paired_diff_t(ret_a, ret_b, lags) -> float:
    a = np.asarray(ret_a, dtype=float)
    b = np.asarray(ret_b, dtype=float)
    d = a - b
    n = len(d)
    lags = int(lags)
    if n < lags + 5:
        return 0.0
    mean = float(d.mean())
    centered = d - mean
    long_var = float(centered @ centered) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_var += 2.0 * weight * float(centered[lag:] @ centered[:-lag]) / n
    return float(mean / np.sqrt(max(long_var, 1e-18) / n))


def _champion_feat_cols(panel: pd.DataFrame, champ_params: dict, fallback: list[str]) -> tuple:
    """冠军回放用的特征列：优先取其 config_snapshot 里训练时的列，而非候选的列。

    返回 (cols, missing)：cols 是快照列中当前面板仍存在的部分（保持快照顺序），
    missing 是快照里有、面板里已没有的列。快照没有 feat_cols 时回退 fallback
    （旧 manifest）。

    注意用 panel.columns 而非 list_factor_columns 判断存在性：list_factor_columns
    会把 pct_chg 等交易 meta 列排除在特征之外，但 keep_trade_cols=True 的面板里
    它们仍在，冻结 champion 线上推理也正是按自己的快照列取数。missing 非空时
    冠军只能在剩余列上回放（cross_sectional_norm 对缺列直接 KeyError，不能原样
    传入），调用方必须把 missing 写进评估结果，不得静默吞掉。
    """
    snap = champ_params.get("feat_cols") or []
    if not snap:
        return list(fallback), []
    present = set(panel.columns)
    cols = [c for c in snap if c in present]
    missing = [c for c in snap if c not in present]
    return cols, missing


def evaluate_pair(
    panel: pd.DataFrame,
    feat_cols: list[str],
    cand_params: dict,
    champ_params: dict,
    eval_days: int | None = None,
    decision_params: dict | None = None,
    initial_cash: float | None = None,
    progress: bool = True,
    progress_cb=None,
    champ_feat_cols: list[str] | None = None,
) -> dict:
    """候选配方 vs 冠军配方：同一评估窗口各跑一遍完整每日重放，返回双方指标。

    两遍只有配方参数（ensemble 腿/超参）与各自的特征列不同；决策/成本/风控
    参数、初始资金、评估窗口完全一致。窗口起点按两边较大的 train_window 预热，
    保证两个配方的决策日集合相同。candidate_ret/champion_ret 仅用于内存中的
    配对检验，stage_and_evaluate 写 evaluation.json 前会剔除这两个大字段。

    champ_feat_cols：冠军训练时的特征列（None → 与候选相同）。必须区分：特征集
    是配方的一部分（指纹含 feat_cols），两边列不同却喂同一份列，会让两次回放
    逐位相同，门控永远判「未超过」。
    """
    eval_days = int(eval_days or getattr(C, "PROMOTE_EVAL_DAYS", 120))
    initial_cash = float(initial_cash or C.INITIAL_CAPITAL)
    dec = dict(decision_params or {})
    warmup = max(
        int(cand_params.get("train_window") or C.TRAIN_WINDOW),
        int(champ_params.get("train_window") or C.TRAIN_WINDOW),
    )
    start_date, end_date = _eval_window(panel, eval_days, warmup)
    # NW 滞后阶数两边取同一常数（≈调仓持有期），保证 alpha/t 可比
    lags = int(getattr(C, "REBALANCE_DAYS", 20))

    out = {"eval_days": eval_days, "window": [str(start_date.date()), str(end_date.date())]}
    sides = (
        ("candidate", cand_params, feat_cols),
        ("champion", champ_params, champ_feat_cols if champ_feat_cols is not None else feat_cols),
    )
    for tag, recipe, cols in sides:
        # warmup_days 显式统一为两边较大值：两个配方的决策日集合完全一致
        p = {"warmup_days": warmup, **dec, **_recipe_params(recipe)}
        if progress_cb is not None:
            side = "候选" if tag == "candidate" else "champion"

            def _side_progress(phase, current, total, message="", **extras):
                progress_cb(
                    phase=f"{side} · {phase}",
                    current=current,
                    total=total,
                    message=message,
                    **extras,
                )

            p["progress_cb"] = _side_progress
        out[tag] = _replay_metrics(
            panel, cols, p, start_date, end_date, initial_cash, lags, progress
        )
        out[tag]["params"] = _recipe_params(recipe)
        out[tag]["n_features"] = len(cols)
    for tag in ("candidate", "champion"):
        out[f"{tag}_ret"] = out[tag].pop("strat_ret")
    return out


def should_promote(
    cand_metrics: dict,
    champ_metrics: dict,
    sharpe_margin: float | None = None,
    dd_tolerance: float | None = None,
    cand_ret=None,
    champ_ret=None,
    lags: int | None = None,
) -> tuple[bool, str]:
    """Sharpe、回撤与可选配对 t 检验的联合门控；旧调用不传收益时跳过 t 检验。"""
    margin = float(
        getattr(C, "PROMOTE_SHARPE_MARGIN", 0.0) if sharpe_margin is None else sharpe_margin
    )
    tol = (
        float(getattr(C, "PROMOTE_DD_TOLERANCE", 0.02) if dd_tolerance is None else dd_tolerance)
        * 100.0
    )
    ds = cand_metrics["sharpe"] - champ_metrics["sharpe"]
    d_dd = cand_metrics["max_drawdown"] - champ_metrics["max_drawdown"]
    ok_sharpe = ds > margin
    ok_dd = d_dd >= -tol
    ok_t = True
    t_value = None
    min_t = float(getattr(C, "PROMOTE_MIN_T", 1.0))
    if min_t > 0 and cand_ret is not None and champ_ret is not None:
        lag_count = int(getattr(C, "REBALANCE_DAYS", 20) if lags is None else lags)
        t_value = paired_diff_t(cand_ret, champ_ret, lag_count)
        ok_t = t_value >= min_t
    reasons = []
    if not ok_sharpe:
        reasons.append(
            f"Sharpe {cand_metrics['sharpe']:+.2f} 未超过 champion "
            f"{champ_metrics['sharpe']:+.2f}" + (f" + 门槛 {margin:.2f}" if margin > 0 else "")
        )
    if not ok_dd:
        reasons.append(
            f"最大回撤 {cand_metrics['max_drawdown']:+.1f}% 劣于 champion "
            f"{champ_metrics['max_drawdown']:+.1f}% − 容忍 {tol:.1f}pp"
        )
    if not ok_t:
        reasons.append(f"配对差值 t={t_value:.2f} 低于门槛 {min_t:.2f}")
    if ok_sharpe and ok_dd and ok_t:
        message = (
            f"Sharpe {cand_metrics['sharpe']:+.2f} > "
            f"{champ_metrics['sharpe']:+.2f}"
            + (f" + 门槛 {margin:.2f}" if margin > 0 else "")
            + f"，回撤 {cand_metrics['max_drawdown']:+.1f}% vs "
            f"{champ_metrics['max_drawdown']:+.1f}% 在容忍内"
        )
        if t_value is not None:
            message += f"，配对差值 t={t_value:.2f} ≥ 门槛 {min_t:.2f}"
        reasons.append(message)
    return ok_sharpe and ok_dd and ok_t, "；".join(reasons)


def _bootstrap_metrics(
    panel,
    feat_cols,
    cand_params,
    eval_days,
    decision_params,
    progress,
    progress_cb,
) -> dict:
    warmup = int(cand_params.get("train_window") or C.TRAIN_WINDOW)
    start_date, end_date = _eval_window(panel, eval_days, warmup)
    params = {"warmup_days": warmup, **dict(decision_params or {}), **_recipe_params(cand_params)}
    if progress_cb is not None:
        params["progress_cb"] = progress_cb
    metrics = _replay_metrics(
        panel,
        feat_cols,
        params,
        start_date,
        end_date,
        float(C.INITIAL_CAPITAL),
        int(getattr(C, "REBALANCE_DAYS", 20)),
        progress,
    )
    metrics.pop("strat_ret", None)
    return metrics


def stage_and_evaluate(
    candidate_ens,
    panel: pd.DataFrame,
    feat_cols: list[str],
    cand_params: dict,
    eval_days: int | None = None,
    auto_promote: bool = False,
    decision_params: dict | None = None,
    progress: bool = True,
    progress_cb=None,
) -> dict:
    """候选落盘 staging + 与 champion 双回放评估；auto_promote=False 时**绝不晋升**。

    两处调用方：
      · run_gate 的"配方不同"分支（auto_promote=True，线上门控：通过即晋升）；
      · pipeline train 模式的 gate_eval 步骤（auto_promote=False：只产 staging
        候选 + evaluation.json，晋升交给 webapp 模型实验室的人工操作）。

    无 champion 时不做双回放（无可比对象），候选照常留档 staging，
    should_promote=False；auto_promote=True 则按首次引导直接晋升。

    返回 dict：
      staging_id / staging_dir : 候选留档位置（models/staging/<model_id>/）
      champion_id              : 评估时点的 champion model_id（无 champion 为 None）
      candidate / champion     : 双方重放标量指标（无 champion 时为 None）
      window / eval_days       : 评估窗口 [start, end] 与长度（无 champion 为 None）
      should_promote           : 门控指标判定（仅代表指标通过，不代表已晋升）
      reason                   : 人类可读判定原因
      promoted                 : 是否实际执行了晋升（auto_promote=True 且通过/引导）
      evaluation               : 写入 evaluation.json 的完整 dict
    """
    eval_days = int(eval_days or getattr(C, "PROMOTE_EVAL_DAYS", 120))
    staging_dir, cand_manifest = model_store.stage_candidate(candidate_ens, feat_cols, cand_params)
    staging_id = cand_manifest["model_id"]
    champ_manifest = model_store.load_champion_manifest()

    # 无 champion：无可比对象，候选留档待人工评审（或 auto_promote 时引导晋升）
    if champ_manifest is None:
        decided_at = datetime.now().isoformat(timespec="seconds")
        if auto_promote:
            reason = "bootstrap（首个 champion，未经对比直接晋升）"
            record = {"reason": reason, "decided_at": decided_at}
            try:
                record["candidate_metrics"] = _bootstrap_metrics(
                    panel,
                    feat_cols,
                    cand_params,
                    eval_days,
                    decision_params,
                    progress,
                    progress_cb,
                )
            except Exception as exc:
                logger.warning("bootstrap 单配方回测失败，仍按原流程晋升：%s", exc)
            model_store.promote(staging_id, promotion_record=record)
            decision = "promoted"
        else:
            reason = "无 champion 可对比：候选已留档 staging，待人工评审后手动晋升"
            decision = "staged"
        evaluation = {
            "eval_days": eval_days,
            "window": None,
            "candidate": None,
            "champion": None,
            "decision": decision,
            "reason": reason,
            "decided_at": decided_at,
            "candidate_id": staging_id,
            "champion_id": None,
        }
        model_store.write_evaluation(staging_dir, evaluation)
        return {
            "staging_id": staging_id,
            "staging_dir": staging_dir,
            "champion_id": None,
            "candidate": None,
            "champion": None,
            "champion_recipe": None,
            "window": None,
            "eval_days": eval_days,
            "should_promote": False,
            "reason": reason,
            "promoted": auto_promote,
            "evaluation": evaluation,
        }

    champ_snapshot = champ_manifest.get("config_snapshot") or {}
    champ_params = _recipe_params(champ_snapshot)
    champ_cols, champ_missing = _champion_feat_cols(panel, champ_snapshot, feat_cols)
    if champ_missing:
        logger.warning(
            "champion 训练列中有 %d 列已不在当前面板（%s），冠军回放只用剩余 %d 列",
            len(champ_missing),
            ",".join(champ_missing[:5]),
            len(champ_cols),
        )
    evaluation = evaluate_pair(
        panel,
        feat_cols,
        cand_params,
        champ_params,
        eval_days=eval_days,
        decision_params=decision_params,
        progress=progress,
        progress_cb=progress_cb,
        champ_feat_cols=champ_cols,
    )
    evaluation["champion_missing_cols"] = champ_missing
    ok, reason = should_promote(
        evaluation["candidate"],
        evaluation["champion"],
        cand_ret=evaluation["candidate_ret"],
        champ_ret=evaluation["champion_ret"],
    )
    evaluation.update(
        {
            "decision": "promoted" if (ok and auto_promote) else ("staged" if ok else "rejected"),
            "reason": reason,
            "decided_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_id": staging_id,
            "champion_id": champ_manifest.get("model_id"),
        }
    )
    model_store.write_evaluation(
        staging_dir,
        {key: value for key, value in evaluation.items() if not key.endswith("_ret")},
    )
    promoted = False
    if ok and auto_promote:
        model_store.promote(
            staging_id,
            promotion_record={
                "reason": reason,
                "decided_at": evaluation["decided_at"],
                "eval_window": evaluation["window"],
                "candidate_metrics": {
                    k: v for k, v in evaluation["candidate"].items() if k != "params"
                },
                "champion_metrics": {
                    k: v for k, v in evaluation["champion"].items() if k != "params"
                },
                "vs_model_id": champ_manifest.get("model_id"),
            },
        )
        promoted = True
    return {
        "staging_id": staging_id,
        "staging_dir": staging_dir,
        "champion_id": champ_manifest.get("model_id"),
        "candidate": evaluation["candidate"],
        "champion": evaluation["champion"],
        "champion_recipe": champ_params,
        "window": evaluation["window"],
        "eval_days": eval_days,
        "should_promote": ok,
        "reason": reason,
        "promoted": promoted,
        "evaluation": evaluation,
    }


def run_gate(
    panel: pd.DataFrame,
    feat_cols: list[str],
    candidate_ens,
    cand_params: dict,
    decision_params: dict | None = None,
    eval_days: int | None = None,
    progress: bool = True,
    progress_cb=None,
) -> dict:
    """门控编排：指纹比较 → （必要时）staging 落盘 + 双重放评估 + 晋升/留档。

    返回结论 dict：
      action               : bootstrap / refreshed / promoted / rejected
      reason               : 人类可读原因
      candidate_id         : 候选 model_id（refreshed 时为 champion id）
      champion_id          : 门控后 champion 的 model_id
      use_candidate_for_live : 今晚订单能否直接用本次训练的候选（True），
                               还是必须按 champion 配方重训（False，rejected）
      evaluation           : 双方指标 + 窗口（refreshed 时 None）
    """
    eval_days = int(eval_days or getattr(C, "PROMOTE_EVAL_DAYS", 120))
    fp = model_store.fingerprint_of(feat_cols, cand_params)
    champ_manifest = model_store.load_champion_manifest()

    # 首次引导：无 champion，候选直接晋升
    if champ_manifest is None:
        staging_dir, manifest = model_store.stage_candidate(candidate_ens, feat_cols, cand_params)
        record = {
            "reason": "bootstrap（首个 champion，未经对比直接晋升）",
            "decided_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            record["candidate_metrics"] = _bootstrap_metrics(
                panel,
                feat_cols,
                cand_params,
                eval_days,
                decision_params,
                progress,
                progress_cb,
            )
        except Exception as exc:
            logger.warning("bootstrap 单配方回测失败，仍按原流程晋升：%s", exc)
        model_store.promote(manifest["model_id"], promotion_record=record)
        return {
            "action": "bootstrap",
            "reason": record["reason"],
            "candidate_id": manifest["model_id"],
            "champion_id": manifest["model_id"],
            "use_candidate_for_live": True,
            "evaluation": None,
        }

    # 同一配方：滚动重训，原位刷新工件，不评估
    if champ_manifest.get("fingerprint") == fp:
        model_store.refresh_champion(candidate_ens, feat_cols, cand_params)
        return {
            "action": "refreshed",
            "reason": "配方指纹与 champion 相同，原位刷新工件（无需评估）",
            "candidate_id": champ_manifest.get("model_id"),
            "champion_id": champ_manifest.get("model_id"),
            "use_candidate_for_live": True,
            "evaluation": None,
        }

    # 配方不同：完整门控（staging 落盘 + 双回放评估 + 通过则晋升），
    # 编排复用 stage_and_evaluate（auto_promote=True：线上门控语义）。
    res = stage_and_evaluate(
        candidate_ens,
        panel,
        feat_cols,
        cand_params,
        eval_days=eval_days,
        auto_promote=True,
        decision_params=decision_params,
        progress=progress,
        progress_cb=progress_cb,
    )
    ok = res["promoted"]
    return {
        "action": "promoted" if ok else "rejected",
        "reason": res["reason"],
        "candidate_id": res["staging_id"],
        "champion_id": (res["staging_id"] if ok else res["champion_id"]),
        "champion_recipe": (None if ok else res["champion_recipe"]),
        "use_candidate_for_live": ok,
        "evaluation": res["evaluation"],
    }
