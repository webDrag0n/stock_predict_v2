"""模型晋升门控：候选（staging）vs 冠军（champion）的公平回测对比与判定。

触发时机：每晚 predict_tomorrow / webapp train_predict 训练完候选后，比较
候选配方指纹与 champion 指纹（model_store.fingerprint_of）：

  · 无 champion        → 首次引导，候选直接晋升（reason="bootstrap"）；
  · 指纹相同           → 同一配方的滚动重训，原位刷新 champion 工件，不评估；
  · 指纹不同           → 完整门控：
      1. 候选落盘 models/staging/<model_id>/；
      2. evaluate_pair：两个配方各自跑一遍最近 PROMOTE_EVAL_DAYS 个交易日的
         完整每日重放（run_daily_replay 正常逐日重训路径，非冻结模型——冻结
         artifact 回放会让训练窗更晚的候选在窗口前段处于样本内，不诚实）；
      3. should_promote：候选 Sharpe > champion Sharpe + PROMOTE_SHARPE_MARGIN
         且候选最大回撤不劣于 champion 回撤 − PROMOTE_DD_TOLERANCE；
      4. 通过 → promote（旧 champion 归档 archive/）；不通过 → 候选留档
         staging + evaluation.json，champion 不变。

两遍重放只有 ensemble 配方参数（腿/超参）不同，决策/成本/风控参数完全一致，
且各自只用自己的成熟标签逐日重训——与生产同分布、无泄露。

另提供 stage_and_evaluate：候选落盘 staging + 双回放评估的独立编排，
auto_promote=False 时绝不晋升（pipeline train 模式用，晋升交人工）；
run_gate 的"配方不同"分支复用它（auto_promote=True）。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

import config as C
import model_store
from backtest_daily import run_daily_replay
from perf_metrics import compute_performance, scalar_metrics

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
    """单配方完整重放 + 标量指标。"""
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
    return scalar_metrics(perf)


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
) -> dict:
    """候选配方 vs 冠军配方：同一评估窗口各跑一遍完整每日重放，返回双方指标。

    两遍只有配方参数（ensemble 腿/超参）不同；决策/成本/风控参数、初始资金、
    评估窗口完全一致。窗口起点按两边较大的 train_window 预热，保证两个配方
    的决策日集合相同。
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
    for tag, recipe in (("candidate", cand_params), ("champion", champ_params)):
        # warmup_days 显式统一为两边较大值：两个配方的决策日集合完全一致
        p = {"warmup_days": warmup, **dec, **_recipe_params(recipe)}
        if progress_cb is not None:
            p["progress_cb"] = progress_cb
        out[tag] = _replay_metrics(
            panel, feat_cols, p, start_date, end_date, initial_cash, lags, progress
        )
        out[tag]["params"] = _recipe_params(recipe)
    return out


def should_promote(
    cand_metrics: dict,
    champ_metrics: dict,
    sharpe_margin: float | None = None,
    dd_tolerance: float | None = None,
) -> tuple[bool, str]:
    """门控判定：Sharpe 更高（+margin）且最大回撤不劣于 champion − tolerance。

    max_drawdown 为百分数（如 -12.3，越小越差）；dd_tolerance 为分数（0.02=2pp）。
    """
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
    if ok_sharpe and ok_dd:
        reasons.append(
            f"Sharpe {cand_metrics['sharpe']:+.2f} > "
            f"{champ_metrics['sharpe']:+.2f}"
            + (f" + 门槛 {margin:.2f}" if margin > 0 else "")
            + f"，回撤 {cand_metrics['max_drawdown']:+.1f}% vs "
            f"{champ_metrics['max_drawdown']:+.1f}% 在容忍内"
        )
    return ok_sharpe and ok_dd, "；".join(reasons)


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
            model_store.promote(
                staging_id, promotion_record={"reason": reason, "decided_at": decided_at}
            )
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

    champ_params = _recipe_params(champ_manifest.get("config_snapshot") or {})
    evaluation = evaluate_pair(
        panel,
        feat_cols,
        cand_params,
        champ_params,
        eval_days=eval_days,
        decision_params=decision_params,
        progress=progress,
        progress_cb=progress_cb,
    )
    ok, reason = should_promote(evaluation["candidate"], evaluation["champion"])
    evaluation.update(
        {
            "decision": "promoted" if (ok and auto_promote) else ("staged" if ok else "rejected"),
            "reason": reason,
            "decided_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_id": staging_id,
            "champion_id": champ_manifest.get("model_id"),
        }
    )
    model_store.write_evaluation(staging_dir, evaluation)
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
