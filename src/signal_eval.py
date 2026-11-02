"""信号评估（modeleval）：滚动/冻结模型的样本外预测质量度量，不做撮合回测。

与 walk-forward 回测（src/walkforward.py）的区别：这里只回答"信号本身好不
好"——PredIC / ICIR / 分位数分层收益 / 多空收益，不经过 trading.decide_orders
与逐笔撮合，因此比完整回测快一个数量级，适合快速比较模型配方或检查冻结
champion 的信号是否仍有效。

两条打分路径（model_ref）：
  "rolling"      滚动重训：窗口切分与 train_predict 复用 walkforward 的同一切
                 分方式（train_w + embargo 起点、while start+test_w<=n 顺推、
                 step 步长），每窗口训练后只对测试段出 prediction，跳过撮合。
  "champion"     冻结 champion 工件（model_store.load_champion）。
  "staging:<id>" 冻结 staging 候选（model_store.load_ensemble(staging_dir_for(id))）。
                 冻结路径对评估区间每个交易日截面打分（ens.predict + ens.combine），
                 不刷新权重、不训练。

防泄露说明：
  - rolling 路径的 embargo 断言与 walkforward 一致（EMBARGO_DAYS >= FORWARD_WINDOW），
    训练段尾部与测试段之间剔除 embargo 天，杜绝 N 日标签跨界泄露。
  - 冻结路径打分是逐日自包含的：截面归一化（cross_sectional_norm）每天只用当日
    横截面，冻结模型本身不含未来数据。但【评估日 <= 模型 train_end_date 时是
    在样本内打分】，结果会虚高——返回的 model_info 带 train_end_date，调用方
    （eval_report）负责给出醒目提示。
  - PredIC/分位收益用的 responder_6 是历史已实现标签，评估口径与
    walkforward._pred_ic 完全相同。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as C
from factor_eval import daily_rank_ic, _layered_returns
from walkforward import train_predict, _pred_ic

logger = logging.getLogger(__name__)


def _emit(progress_cb, phase: str, current: int, total: int, message: str = "", **extras) -> None:
    """进度上报包装：可选、异常吞掉（progress 永远不能影响主流程）。"""
    if progress_cb is None:
        return
    try:
        progress_cb(phase=phase, current=current, total=total, message=message, **extras)
    except Exception:
        pass


def _ic_stats(daily_ic: pd.Series) -> tuple[float, float, float]:
    """逐日 RankIC 序列 → (PredIC 均值, ICIR, IC>0 占比)。口径同 walkforward._pred_ic。"""
    ic = daily_ic.dropna()
    if len(ic) == 0:
        return float("nan"), float("nan"), float("nan")
    icir = float(ic.mean() / ic.std()) if ic.std() > 1e-12 else float("nan")
    return float(ic.mean()), icir, float((ic > 0).mean())


def _truncate_panel(panel: pd.DataFrame, need_days: int) -> pd.DataFrame:
    """把面板截到最后 need_days 个交易日（控制评估耗时；训练只需尾部窗口）。"""
    dates = np.sort(panel["date"].unique())
    if need_days > 0 and len(dates) > need_days:
        panel = panel[panel["date"] >= dates[-need_days]].reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# rolling 路径：滚动重训（复用 walkforward 的切分与 train_predict，跳过撮合）
# ---------------------------------------------------------------------------
def _rolling_oos(
    panel: pd.DataFrame,
    feat_cols: list[str],
    lookback_days: int,
    train_window: int,
    test_window: int,
    wf_step: int,
    progress_cb,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """滚动窗口切分 + train_predict，返回 (拼接 OOS 帧, 逐窗口指标帧)。

    窗口切分与 walkforward.run_walkforward 保持一致（start = train_w + embargo
    起点、while start + test_w <= n 顺推、wf_step 步长），只是窗口长度由参数
    传入而非读 config；每窗口只做训练+预测+PredIC，不跑每日重放撮合。
    """
    embargo = C.EMBARGO_DAYS
    # 与 walkforward 同一防泄露断言：embargo 不足时训练段尾部标签（shift(-fw)
    # 远期收益）会偷看测试段价格。
    assert (
        embargo >= C.FORWARD_WINDOW
    ), f"EMBARGO_DAYS={embargo} < FORWARD_WINDOW={C.FORWARD_WINDOW}（前视泄露）"

    panel = _truncate_panel(panel, train_window + embargo + lookback_days)
    panel = panel.sort_values("date").reset_index(drop=True)
    # weight 列保障（load_factor_panel 不含 weight；与 run_walkforward 同样补等权，
    # 保证 ret_weighted 目标的样本权重口径一致）
    if C.WEIGHT not in panel.columns:
        panel[C.WEIGHT] = np.float32(1.0)
    dates = np.sort(panel["date"].unique())
    n_dates = len(dates)

    train_w, test_w, step = int(train_window), int(test_window), int(wf_step)
    n_windows_total = max((n_dates - train_w - embargo - test_w) // step + 1, 0)
    if n_windows_total <= 0:
        raise ValueError(
            f"数据不足以构成任何滚动窗口：面板 {n_dates} 天 < "
            f"train_window({train_w}) + embargo({embargo}) + test_window({test_w})"
        )
    logger.info(
        f"信号评估（rolling）：共 {n_windows_total} 个窗口 "
        f"(train={train_w} test={test_w} step={step} embargo={embargo})"
    )

    oos_frames = []
    win_rows = []
    start = train_w + embargo
    win_id = 0
    while start + test_w <= n_dates:
        tr_dates = dates[start - embargo - train_w : start - embargo]
        te_dates = dates[start : start + test_w]
        train = panel[panel["date"].isin(tr_dates)]
        test = panel[panel["date"].isin(te_dates)].copy()

        pred = train_predict(train, test, feat_cols)
        test["prediction"] = pred
        w_ic, _ = _pred_ic(test)
        win_rows.append(
            {
                "window": win_id,
                "train": f"{pd.Timestamp(tr_dates[0]).date()}~{pd.Timestamp(tr_dates[-1]).date()}",
                "test": f"{pd.Timestamp(te_dates[0]).date()}~{pd.Timestamp(te_dates[-1]).date()}",
                "n_feat": len(feat_cols),
                "PredIC": w_ic,
            }
        )
        keep = [c for c in ("date", "symbol", C.TARGET, "prediction") if c in test.columns]
        oos_frames.append(test[keep])
        _emit(
            progress_cb,
            "滚动预测中",
            win_id + 1,
            n_windows_total,
            f"窗口 {win_id + 1}/{n_windows_total} "
            f"测试{pd.Timestamp(te_dates[0]).date()}~{pd.Timestamp(te_dates[-1]).date()}",
            window_pred_ic=w_ic,
        )
        logger.info(
            f"  窗口 {win_id + 1}/{n_windows_total}: 测试"
            f"{pd.Timestamp(te_dates[0]).date()}~{pd.Timestamp(te_dates[-1]).date()} "
            f"PredIC={w_ic:+.4f}"
        )
        start += step
        win_id += 1

    return pd.concat(oos_frames, ignore_index=True), pd.DataFrame(win_rows)


# ---------------------------------------------------------------------------
# 冻结模型路径：champion / staging:<id>，对评估区间逐日截面打分
# ---------------------------------------------------------------------------
def _load_frozen(model_ref: str):
    """按 model_ref 加载冻结 ensemble。返回 (ens, manifest)。"""
    import model_store

    if model_ref == "champion":
        loaded = model_store.load_champion()
        if loaded is None:
            raise ValueError("无 champion 工件可加载（models/champion/ 不存在或已损坏）")
        return loaded
    if model_ref.startswith("staging:"):
        model_id = model_ref.split(":", 1)[1].strip()
        if not model_id:
            raise ValueError("model_ref='staging:' 缺 model_id")
        model_dir = model_store.staging_dir_for(model_id)
        return model_store.load_ensemble(model_dir)
    raise ValueError(f"未知 model_ref={model_ref!r}（支持 rolling / champion / staging:<id>）")


def _frozen_oos(
    panel: pd.DataFrame,
    model_ref: str,
    lookback_days: int,
    progress_cb,
) -> tuple[pd.DataFrame, dict]:
    """冻结 ensemble 对最后 lookback_days 个交易日逐日截面打分。

    用轻量内部打分（ens.predict + ens.combine）而非 live.predict_latest_with_ens：
    后者会 refresh_weights（从 ic_hist 重算/平滑权重，改变冻结工件保存的权重）
    且面向"最新日"语义；信号评估要的是【按工件保存的权重原样打分】，逐日切片
    自包含（截面归一化只用当日横截面，无泄露），也不需要把整面板截断到目标日。
    """
    ens, manifest = _load_frozen(model_ref)
    dates = np.sort(panel["date"].unique())
    eval_dates = dates[-int(lookback_days) :] if lookback_days > 0 else dates
    n = len(eval_dates)
    logger.info(
        f"信号评估（{model_ref}）：{n} 个交易日逐日打分 "
        f"（模型训练截止 {manifest.get('train_end_date')}）"
    )

    frames = []
    for i, d in enumerate(eval_dates):
        today = panel[panel["date"] == d].copy()
        if today.empty:
            _emit(progress_cb, "逐日打分", i + 1, n, f"{pd.Timestamp(d).date()} 无截面")
            continue
        preds = ens.predict(today)
        score = ens.combine(preds, today["date"].to_numpy())
        today["prediction"] = score
        keep = [c for c in ("date", "symbol", C.TARGET, "prediction") if c in today.columns]
        frames.append(today[keep])
        _emit(progress_cb, "逐日打分", i + 1, n, str(pd.Timestamp(d).date()))

    if not frames:
        raise ValueError("评估区间内无可打分的交易日")
    oos = pd.concat(frames, ignore_index=True)
    model_info = {
        "model_id": manifest.get("model_id"),
        "train_end_date": manifest.get("train_end_date"),
        "weights": manifest.get("weights"),
        "models": list(ens.models),
    }
    return oos, model_info


# ---------------------------------------------------------------------------
def run_signal_eval(
    panel: pd.DataFrame,
    feat_cols: list[str],
    model_ref: str = "rolling",
    lookback_days: int = 120,
    train_window: int = 500,
    test_window: int = 60,
    wf_step: int = 60,
    n_quantiles: int = 5,
    progress_cb=None,
) -> dict:
    """信号质量评估：产出 OOS 预测的 PredIC/ICIR/分位数分层收益（不做撮合回测）。

    Args:
        panel: 命名因子面板（load_factor_panel；含 date/symbol/因子/responder_6）。
        feat_cols: 参与训练的因子列。
        model_ref: "rolling"（滚动重训）/ "champion" / "staging:<model_id>"（冻结工件）。
        lookback_days: 评估区间长度（交易日，取面板尾部）。
        train_window/test_window/wf_step: rolling 路径的滚动窗口配置
            （切分方式与 walkforward 一致，仅参数化）。
        n_quantiles: 分位数分层数（prediction 当因子列喂 _layered_returns）。
        progress_cb: 可选进度回调 cb(phase=, current=, total=, message=, **extras)，
            rolling 按窗口上报、冻结模型按交易日上报；异常吞掉。

    返回 dict：
        model_ref / eval_start / eval_end / n_dates / n_rows
        pred_ic_mean / pred_icir / ic_positive_ratio   整体 PredIC 指标
        daily_ic        DataFrame(date, rank_ic)       逐日截面 RankIC 序列
        windows         DataFrame(window/train/test/PredIC)（rolling；冻结为 None）
        n_windows       窗口数（rolling；冻结为 None）
        layer_returns   list[float]                    Q1..Qn 各层平均下期收益
        long_short_return / monotonicity               多空收益与分层单调性
        n_quantiles
        model_info      dict（冻结路径：model_id/train_end_date/weights；rolling 为 None）
    """
    model_ref = str(model_ref or "rolling")
    if model_ref == "rolling":
        oos, windows = _rolling_oos(
            panel,
            feat_cols,
            lookback_days=lookback_days,
            train_window=train_window,
            test_window=test_window,
            wf_step=wf_step,
            progress_cb=progress_cb,
        )
        model_info = None
    else:
        oos, model_info = _frozen_oos(panel, model_ref, lookback_days, progress_cb)
        windows = None

    # 汇总指标（与 walkforward._pred_ic 同口径的逐日截面 RankIC）
    daily_ic = daily_rank_ic(oos, "prediction", C.TARGET)
    pred_ic_mean, pred_icir, ic_pos = _ic_stats(daily_ic)
    daily_ic_df = daily_ic.rename("rank_ic").reset_index()

    # 分位数分层收益：prediction 当因子列
    layer_ret, ls_ret, mono = _layered_returns(oos, "prediction", C.TARGET, n_quantiles)

    eval_dates = np.sort(oos["date"].unique())
    _emit(progress_cb, "汇总指标", 1, 1, "完成")
    result = {
        "model_ref": model_ref,
        "eval_start": str(pd.Timestamp(eval_dates[0]).date()),
        "eval_end": str(pd.Timestamp(eval_dates[-1]).date()),
        "n_dates": int(len(eval_dates)),
        "n_rows": int(len(oos)),
        "pred_ic_mean": pred_ic_mean,
        "pred_icir": pred_icir,
        "ic_positive_ratio": ic_pos,
        "daily_ic": daily_ic_df,
        "windows": windows,
        "n_windows": (int(len(windows)) if windows is not None else None),
        "layer_returns": [float(v) if v == v else None for v in layer_ret],
        "long_short_return": (float(ls_ret) if ls_ret == ls_ret else None),
        "monotonicity": float(mono),
        "n_quantiles": int(n_quantiles),
        "model_info": model_info,
    }
    logger.info(
        f"信号评估完成：{model_ref} · {len(eval_dates)} 天 · "
        f"PredIC={pred_ic_mean:+.4f} ICIR={pred_icir:+.3f} 多空={ls_ret:+.4f}"
    )
    return result
