"""实盘推理核心：把"截至今天的数据"变成"明日可执行的目标持仓 / 订单"。

设计原则
--------
1. 训练/实盘同分布：实盘用 IncrementalEnsemble（ensemble_predict_latest），
   与每日重放回测（backtest_daily.run_daily_replay）逐项同口径——同样的多模型
   腿、截面归一化、滚动 IC 定权与五层风控（src/risk.py）。
2. 信号只用于排序：long_topk 只看预测排名，不把 prediction 当涨幅，与回测一致。
3. 无未来信息：只用 date<=今天 的特征；最新一天的特征是当日收盘后才完整的，
   因此信号当晚算出、次日执行。

纯函数为主，便于单测；CLI 编排在 predict_tomorrow.py。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config as C
from ensemble import IncrementalEnsemble
from walkforward import train_predict

logger = logging.getLogger(__name__)


def live_panel_start_date(train_window: int | None = None, buffer_trading_days: int = 80) -> str:
    """计算实盘/次日任务所需的最早 start_date。

    覆盖训练窗口 + forward_window + embargo + IC/风险滚动窗口 + 缓冲。
    """
    tw = int(train_window or getattr(C, "TRAIN_WINDOW", 500))
    fw = int(getattr(C, "FORWARD_WINDOW", 20))
    emb = int(getattr(C, "EMBARGO_DAYS", fw))
    needed = tw + fw + emb + buffer_trading_days
    # 交易日 → 自然日：约 1.4 倍
    calendar_days = int(needed * 1.4) + 30
    from datetime import datetime, timedelta

    return (datetime.now() - timedelta(days=calendar_days)).strftime("%Y-%m-%d")


def load_live_panel(
    train_window: int | None = None,
    *,
    gate_on: bool = False,
    eval_days: int | None = None,
    verbose: bool = True,
    loader=None,
) -> tuple[pd.DataFrame, list[str]]:
    """次日下单流程的数据加载段：加载含【最新无标签日】+ 交易列的实盘面板。

    predict_tomorrow.py（CLI）与 pipeline/steps_nextday.py（webapp 会话）共用。
    窗口用 live_panel_start_date 收敛到最小必要范围；门控启用时 buffer 需额外
    覆盖评估窗口（双回放要有 eval_days 个决策日）。

    Args:
        loader: 可注入的面板加载函数（默认 data.load_factor_panel），测试可替换。
    """
    from data import load_factor_panel, list_factor_columns

    if gate_on:
        ed = int(eval_days if eval_days is not None else getattr(C, "PROMOTE_EVAL_DAYS", 120))
        buffer = max(80, ed + 10)
    else:
        buffer = 80
    start_date = live_panel_start_date(train_window=train_window, buffer_trading_days=buffer)
    load = loader if loader is not None else load_factor_panel
    panel = load(keep_unlabeled=True, keep_trade_cols=True, start_date=start_date, verbose=verbose)
    factors = list_factor_columns(panel)
    # 因子列 float64→float32：截面归一化（cross_sectional_norm）输出本就是
    # float32，GBM 又把特征离散化到 ≤255 bin，f32 的精度损失（~1e-7 相对）
    # 对训练/预测可忽略；常驻内存减半（全市场面板因子块 ~2.1GB→~1.1GB）。
    # 价格/标签/成交等 meta 列（list_factor_columns 之外）保持 float64。
    for c in factors:
        if panel[c].dtype == np.float64:
            panel[c] = panel[c].astype(np.float32)
    return panel, factors


def train_and_predict_latest(panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """实盘训练+预测，**直接复用** walkforward.train_predict，与已验证回测同分布。

    注意：当前无生产调用方（生产路径是 ensemble_predict_latest，与每日重放
    回测同口径），保留供对照/调试。训练集=全部有标签行，预测对象=最新
    无标签日，于是 MODEL_OBJECTIVE(ret_weighted)、按日截面 z-score+MAD-winsorize、
    多 seed、正则化超参全部与回测一致。无未来信息：最新日的特征当晚已完整。

    返回 today 子表（含 close、prediction、rank_pct），按 prediction 降序。
    """
    latest = panel["date"].max()
    train = panel.dropna(subset=[C.TARGET]).copy()
    today = panel[panel["date"] == latest].copy()
    if today.empty or train.empty:
        raise ValueError("无最新无标签日或无有标签训练数据")

    # train_predict 期望 train/test 含 'date' 列（按日截面归一化按 date 分组）。
    today["prediction"] = train_predict(train, today, feat_cols)
    today["rank_pct"] = today["prediction"].rank(ascending=False, pct=True)
    return today.sort_values("prediction", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 新路径：多模型增量集成 + 共享 decide_orders（与每日重放回测同分布）
# ---------------------------------------------------------------------------
def _emit_fn(progress_cb):
    def _emit(phase, current, total, message="", **extras):
        if progress_cb is None:
            return
        try:
            progress_cb(phase=phase, current=current, total=total, message=message, **extras)
        except Exception:
            pass

    return _emit


def predict_latest_with_ens(
    ens, panel: pd.DataFrame, pred_hist: pd.DataFrame | None = None, progress_cb=None
) -> tuple[pd.DataFrame, dict]:
    """训练无关的预测尾段：observe pred_hist → 滚动 IC 定权 → 预测最新日。

    供两条路径共用：ensemble_predict_latest（全量重训后）与
    champion_predict_latest（冻结 champion 工件加载后）。ens 必须已具备预测
    能力（已训练或已从 state_dict 恢复）。

    滚动 IC 定权用【持久化的真 OOS 预测历史】补 observe 后 refresh_weights：
    只给 d + FORWARD_WINDOW ≤ latest 的成熟日打分，无泄露。
    """
    _emit = _emit_fn(progress_cb)
    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    if today.empty:
        raise ValueError("无最新日横截面")

    _emit(
        "滚动 IC 定权",
        2,
        3,
        n_hist_days=(
            int(pred_hist["date"].nunique()) if pred_hist is not None and len(pred_hist) else 0
        ),
    )
    if pred_hist is not None and len(pred_hist):
        for d in sorted(pred_hist["date"].unique()):
            sub = pred_hist[pred_hist["date"] == d]
            preds_d = {
                c[len("pred_") :]: sub[c].to_numpy(dtype=np.float64)
                for c in sub.columns
                if c.startswith("pred_") and c[len("pred_") :] in ens.models
            }
            if not preds_d:
                continue
            ens.observe(np.datetime64(d), sub[["symbol"]].reset_index(drop=True), preds_d)
        ens.refresh_weights(panel, asof_date=latest)
        # 已成熟且成功打分的天数（各腿最小值）；不足窗口时 ensemble 内部
        # _recompute_weights 已保证退化为等权，这里记录说明避免误以为已定权。
        n_scored = min((len(ens._ic_hist[m]) for m in ens.models), default=0)
        if n_scored < ens.ic_window:
            logger.info(
                f"[提示] 实盘 OOS IC 定权历史不足：已成熟打分 {n_scored} 天 "
                f"（窗口 {ens.ic_window} 天，<3 天时完全等权）；"
                f"每日运行 predict_tomorrow 后会逐日积累。"
            )
    else:
        logger.info(
            "[提示] 无持久化 OOS 预测历史（pred_hist 为空），本次模型权重为等权；"
            "每日运行 predict_tomorrow 后会逐日积累。"
        )
        ens.refresh_weights(panel, asof_date=latest)  # 无历史 → 等权

    _emit("预测最新日", 3, 3, n_today_rows=len(today))
    preds = ens.predict(today)
    score = ens.combine(preds, today["date"].to_numpy())
    today = today.copy()
    for m, pr in preds.items():
        today[f"pred_{m}"] = pr
    today["score"] = score
    today["prediction"] = score  # decide_orders/排序统一用 score
    today["rank_pct"] = pd.Series(score, index=today.index).rank(ascending=False, pct=True)
    today = today.sort_values("score", ascending=False).reset_index(drop=True)
    info = {
        "weights": dict(ens.weights),
        "ic_means": ens.ic_means(),
        "models": list(ens.models),
        "latest_date": latest,
        # 集成本体：模型门控（model_gate）需要把候选落盘/对比。
        # 仅内存引用，不随 info 落盘。
        "ensemble": ens,
    }
    return today, info


def train_candidate_ensemble(
    panel: pd.DataFrame,
    feat_cols: list[str],
    params: dict | None = None,
    progress_cb=None,
) -> tuple[IncrementalEnsemble, dict]:
    """训练候选 ensemble：有同配方 champion 时在其工件上增量续训，否则全量重建。

    从 ensemble_predict_latest 抽出的训练段，供两条路径共用：
    实盘门控（run_gated_prediction 经 ensemble_predict_latest）与 pipeline
    train 模式（steps_train.train_candidate）。训练/定权口径与每日重放回测
    逐项一致：同样的多模型腿、截面归一化、滚动 IC 定权窗口。

    增量分支：champion 存在且配方指纹（model_store.fingerprint_of：特征列 +
    fw/embargo + ensemble 腿与训练超参）与当前一致时，load_champion 恢复工件
    后 update(force_full=False)——只在新成熟标签日上续训 INCR_TREES×新成熟天数
    棵树（REBUILD_DAYS 周期重建/树数上限等安全网仍由 update 内部保证）；
    无 champion 或配方变化时退回全量重建（force_full=True）。

    防泄露：只用 date <= latest - (FORWARD_WINDOW + EMBARGO_DAYS) 的已成熟
    标签行训练；最新无标签日不参与训练。

    Args:
        panel: 命名因子面板（含最新无标签日）。
        feat_cols: 因子列。
        params: 集成超参（ensemble_models/n_seeds/incr_trees/rebuild_days/
                train_window/ic_window），缺省项回落 config。
        progress_cb: 可选进度回调 cb(phase=..., current=..., total=..., **extras)，
                     异常吞掉，绝不影响主流程。

    返回 (ens, recipe)：
      ens   : 训练好的 IncrementalEnsemble（已具备预测能力）。
      recipe: 生效配方 dict（六个配方键按 config 默认值归一后的实际值，
              另附 train_mode=incremental/full 训练路径标记），
              可直接作 model_store 配方指纹 / manifest config_snapshot 的
              params 输入（两者均按键白名单取），以及门控评估的候选配方。
    """
    import model_store

    p = dict(params or {})
    fw = int(getattr(C, "FORWARD_WINDOW", 20))
    embargo = int(getattr(C, "EMBARGO_DAYS", fw))
    _emit = _emit_fn(progress_cb)

    latest = panel["date"].max()
    _emit("初始化集成", 0, 3, latest_date=str(pd.Timestamp(latest).date()))

    ens = IncrementalEnsemble(
        feat_cols,
        models=p.get("ensemble_models") or None,
        n_seeds=p.get("n_seeds"),
        incr_trees=p.get("incr_trees"),
        rebuild_days=p.get("rebuild_days"),
        train_window=p.get("train_window"),
        ic_window=p.get("ic_window"),
    )
    recipe = {
        "ensemble_models": ",".join(ens.models),
        "n_seeds": ens.n_seeds,
        "incr_trees": ens.incr_trees,
        "rebuild_days": ens.rebuild_days,
        "train_window": ens.train_window,
        "ic_window": ens.ic_window,
    }

    # 增量分支：champion 配方指纹一致 → 恢复工件续训；否则（无 champion /
    # 配方变化 / 工件损坏）从全新集成全量重建。指纹用与 run_gate 相同的
    # 原始 params 计算（fingerprint_of 内部对缺省键回落 config 默认值），
    # 避免归一化差异导致与 champion manifest 指纹错位、每晚静默退回全量。
    force_full = True
    champ = model_store.load_champion()
    if champ is not None:
        champ_ens, champ_manifest = champ
        fp = model_store.fingerprint_of(feat_cols, p)
        if champ_manifest.get("fingerprint") == fp:
            ens = champ_ens
            force_full = False

    # 已成熟标签：date <= latest - (fw + embargo)。只切【尾部 train_window 个
    # 成熟日 × 训练所需列】一份拷贝——update 内部本就只取尾部窗口训练，
    # 预裁剪避免全量成熟帧（~3.2M 行 × 全列，数 GB）在 matured/dropna/win
    # 各环节被反复整体拷贝（nextday OOM 的直接死因）。增量续训的新成熟日
    # 必落在尾部窗口内：champion 过旧（新成熟日 > FULL_TREES 安全网）会在
    # update 内走 tree-cap 全量重建，同样只需窗口帧。
    all_dates = np.sort(panel["date"].unique())
    di = int(np.searchsorted(all_dates, latest))
    mat_cut = all_dates[max(0, di - fw - embargo)]
    matured_mask = (panel["date"].to_numpy() <= mat_cut) & panel[C.TARGET].notna().to_numpy()
    if not matured_mask.any():
        raise ValueError("无已成熟标签可训练")
    matured_dates = np.sort(panel.loc[matured_mask, "date"].unique())
    win_start = matured_dates[-ens.train_window] if len(matured_dates) > ens.train_window else None
    if win_start is not None:
        matured_mask &= panel["date"].to_numpy() >= win_start
    keep = ["date", "symbol", C.TARGET, C.WEIGHT, "ln_mv", *feat_cols]
    keep = [c for c in dict.fromkeys(keep) if c in panel.columns]
    matured = panel.loc[matured_mask, keep]
    if force_full:
        _emit("全量重建中", 1, 3, n_train_rows=len(matured), models=",".join(ens.models))
    else:
        _emit(
            "champion 续训中",
            1,
            3,
            n_train_rows=len(matured),
            models=",".join(ens.models),
            champion_id=champ_manifest.get("model_id"),
        )
        logger.info(
            f"[训练] 配方指纹与 champion（{champ_manifest.get('model_id')}）一致，"
            f"增量续训（训练截止 {champ_manifest.get('train_end_date')} → 新成熟日）"
        )
    train_info = ens.update(matured, asof_date=latest, force_full=force_full)
    logger.info(f"[训练] update: {train_info}")

    recipe["train_mode"] = "full" if force_full else "incremental"
    return ens, recipe


def ensemble_predict_latest(
    panel: pd.DataFrame, feat_cols: list[str], params: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """用 IncrementalEnsemble 训练全部成熟历史、预测最新（无标签）日。

    与 backtest_daily.run_daily_replay 的训练/定权口径**逐项一致**：同样的多模型
    （gbm_l2 / gbm_ret_weighted / ic_linear）、同样的截面归一化、同样的滚动 IC 定权。
    实盘只跑一次：训练段委托 train_candidate_ensemble（有同配方 champion 时增量
    续训，否则全量重建把尾部 TRAIN_WINDOW 训到位），再对最新无标签日预测（预测
    尾段与冻结路径共用 predict_latest_with_ens）。

    返回 (today, info)：
      today: 最新日横截面，附 ensemble 各模型分 + 融合 score + prediction + rank_pct，
             并保留 close/bb_*/rsi_14 等 decide_orders 需要的 meta 列。
      info : {"weights":..., "ic_means":..., "models":..., "latest_date":...,
              "ensemble": 训练好的集成本体}
    """
    p = dict(params or {})
    progress_cb = p.pop("progress_cb", None)

    today = panel[panel["date"] == panel["date"].max()]
    if today.empty:
        raise ValueError("无最新日横截面")

    ens, _recipe = train_candidate_ensemble(panel, feat_cols, params=p, progress_cb=progress_cb)
    return predict_latest_with_ens(
        ens, panel, pred_hist=p.get("pred_hist"), progress_cb=progress_cb
    )


def champion_predict_latest(
    panel: pd.DataFrame, pred_hist: pd.DataFrame | None = None, progress_cb=None
) -> tuple[pd.DataFrame, dict]:
    """冻结路径：加载 champion 工件直接预测（不训练），定权状态刷新后回存。

    用于自适应重训的"未触发"分支与门控 rejected 分支——champion 冻结工件即
    生产模型。refresh_weights 用 pred_hist 把新成熟日打分入 _ic_hist（定权
    照常保鲜），随后 refresh_champion 回存工件：weights/ic_hist 更新，
    train_end_date 与 promotion 记录不变（模型本身未重训）。
    """
    import model_store

    _emit = _emit_fn(progress_cb)
    loaded = model_store.load_champion()
    if loaded is None:
        raise ValueError("无 champion 工件可加载（models/champion/ 不存在或已损坏）")
    ens, manifest = loaded
    _emit(
        "加载 champion 工件",
        1,
        3,
        model_id=manifest.get("model_id"),
        train_end_date=manifest.get("train_end_date"),
    )
    today, info = predict_latest_with_ens(ens, panel, pred_hist=pred_hist, progress_cb=progress_cb)
    snap = manifest.get("config_snapshot") or {}
    model_store.refresh_champion(ens, snap.get("feat_cols") or [], snap)
    info["champion_id"] = manifest.get("model_id")
    info["action"] = "frozen"
    return today, info


def run_gated_prediction(
    panel: pd.DataFrame,
    factors: list[str],
    *,
    ensemble_params: dict | None = None,
    decision_params: dict | None = None,
    eval_days: int | None = None,
    gate_on: bool = True,
    adaptive: bool | None = None,
    force_retrain: bool = False,
    pred_hist: pd.DataFrame | None = None,
    risk_state_path: str | None = None,
    gate_progress: bool = True,
    progress_cb=None,
    train_fn=None,
    log=print,
) -> dict:
    """次日下单流程的"训练/冻结 + 门控 + 预测"主体，CLI 与流水线步骤共用。

    自适应重训触发（IC 退化/收益落后/超龄，或 force_retrain）：
      触发   → train_fn（默认 ensemble_predict_latest）训练（同配方 champion
               存在时增量续训，否则全量）并预测最新日；
               gate_on 且产出集成本体时再跑 model_gate 门控（候选 vs champion
               双回放），被拒则改用冻结 champion 出信号。
      未触发 → champion_predict_latest 冻结预测。

    Args:
        ensemble_params: 集成超参（n_seeds/incr_trees/rebuild_days/train_window/
                         ic_window），同时作为门控的候选配方指纹输入。
        decision_params: 下单参数（门控双回放需与正式下单同口径）。
        eval_days: 门控评估窗口交易日数（默认 C.PROMOTE_EVAL_DAYS）。
        gate_on: 是否启用模型门控（USE_MODEL_GATE 且用户未关闭）。
        adaptive: 是否走自适应触发；None → gate_on and C.USE_ADAPTIVE_RETRAIN。
        pred_hist: 持久化真 OOS 预测历史（滚动 IC 定权 + 触发指标的数据源）。
        risk_state_path: 风控状态 JSON 路径（触发指标的收益落后判定用）。
        gate_progress: 门控双回放是否打印进度（CLI True / 会话 False）。
        progress_cb: 会话进度回调，透传给训练/冻结/门控。
        train_fn: 可注入的训练预测函数（默认 ensemble_predict_latest），测试可替换。
        log: 人类可读日志回调（CLI 用 print；会话传 None 静默，结论走返回值）。

    返回 dict：
      today/info : 最新日横截面与预测摘要（同 ensemble_predict_latest 的返回）
      gate       : 门控结论 dict（未跑门控为 None）
      action     : frozen / trained / bootstrap / refreshed / promoted / rejected
      fired      : 是否触发了重训
      reasons    : 触发/未触发原因列表
      ic_stats   : 成熟 OOS IC 统计（仅 adaptive 路径计算，否则 None）
    """
    from live_tracking import matured_ic_stats, nav_stats
    from retrain_trigger import should_retrain, model_age_days

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    if adaptive is None:
        adaptive = gate_on and bool(getattr(C, "USE_ADAPTIVE_RETRAIN", False))

    p = dict(ensemble_params or {})
    if pred_hist is not None:
        p["pred_hist"] = pred_hist
    if progress_cb is not None:
        p["progress_cb"] = progress_cb

    ic_stats = None
    fire, reasons, gate = True, [], None
    if adaptive:
        import model_store

        champ = model_store.load_champion()
        champ_manifest = champ[1] if champ else None
        ic_stats = matured_ic_stats(panel, pred_hist)
        fire, reasons = should_retrain(
            champ_manifest, ic_stats, nav_stats(risk_state_path, panel), panel, force=force_retrain
        )
        if fire:
            _log("[重训触发] " + "；".join(reasons))
        else:
            age = model_age_days(champ_manifest, panel)
            _log(
                f"[重训触发] 未触发：今晚用冻结 champion 预测"
                f"（{champ_manifest.get('model_id')}，年龄 {age} 个交易日）"
            )
    elif gate_on:
        reasons = ["自适应重训已关闭（USE_ADAPTIVE_RETRAIN=0），每晚重训"]

    latest = panel["date"].max()
    train = train_fn if train_fn is not None else ensemble_predict_latest
    if not fire:
        _emit = _emit_fn(progress_cb)
        _emit("冻结 champion 预测", 1, 3, "未触发重训")
        today, info = champion_predict_latest(panel, pred_hist=pred_hist, progress_cb=progress_cb)
        action = "frozen"
        _log(
            f"冻结 champion 已预测 {pd.Timestamp(latest).date()} 的 "
            f"{len(today)} 只股票（champion={info.get('champion_id')}）"
        )
    else:
        _log("训练并预测（多模型增量集成 + 滚动 IC 定权）...")
        today, info = train(panel, factors, params=p)
        action = "trained"
        _log(f"已预测 {pd.Timestamp(latest).date()} 的 {len(today)} 只股票")

        # 模型门控：候选配方指纹 vs champion。相同 → 原位刷新工件；不同 →
        # staging 落盘 + 双方最近 eval_days 天完整重放对比，Sharpe 更高且
        # 回撤不劣才晋升；被拒时今晚信号用冻结 champion（订单永远出自 champion）。
        if gate_on and info.get("ensemble") is not None:
            from model_gate import run_gate

            gate = run_gate(
                panel,
                factors,
                info["ensemble"],
                dict(ensemble_params or {}),
                decision_params=decision_params,
                eval_days=eval_days,
                progress=gate_progress,
                progress_cb=progress_cb,
            )
            action = gate["action"]
            if not gate["use_candidate_for_live"]:
                _log(f"候选被拒，改用冻结 champion 出信号" f"（champion={gate['champion_id']}）...")
                _emit = _emit_fn(progress_cb)
                _emit("冻结 champion 预测", 2, 3, f"候选被拒，champion={gate['champion_id']}")
                today, info = champion_predict_latest(
                    panel, pred_hist=pred_hist, progress_cb=progress_cb
                )
                action = "rejected"

    return {
        "today": today,
        "info": info,
        "gate": gate,
        "action": action,
        "fired": fire,
        "reasons": reasons,
        "ic_stats": ic_stats,
    }


def make_orders_from_state(
    today: pd.DataFrame,
    score: np.ndarray,
    current: dict[str, int],
    capital: float,
    params: dict | None = None,
    price_col: str = "close",
    risk_state_path: str | None = None,
    panel: pd.DataFrame | None = None,
    cash: float | None = None,
) -> pd.DataFrame:
    """用共享 decide_orders 生成"明日可执行"逐笔订单（与回测同一决策代码）。

    把当前持仓 + 现金组成 PortfolioState，交给
    trading.decide_orders：ML 选池 + 布林/RSI 择时 → 精确到 (代码,方向,股数,挂单价)
    的订单，附触发原因。返回 trading.decide_orders 的订单表（实盘可直接照单执行）。

    Args:
        today: 最新日横截面，需含 symbol、price_col、bb_*、rsi_14 等。
        score: 与 today 对齐的融合排序分。
        current: 当前持仓 {代码: 股数}。
        capital: 组合总资金兜底（未提供 cash 时用，反推现金 = capital − 持仓市值）。
        params: 决策参数（topk、max_positions、max_weight、order_price_mode 等）。
        price_col: 用于估值与目标金额计算的价列，默认 "close"（金钱计算必须用
                   真实市价；adj_close 绝对水平随加载窗口漂移，只用于因子）。
        risk_state_path: 风控状态 JSON 路径；若提供则读取历史波动/回撤/IC，
                         计算 risk_scale 并更新状态文件。
        panel: 完整历史面板（多日期），用于 regime/IV/成熟 IC 计算；
               缺省用 today 退化为单日线。
        cash: 账户真实现金（account.csv）。提供后总估值 = cash + Σ(持仓×最新
              原始 close)，如实入账驱动 vol/dd 两层；未提供时记录醒目警告
              （logger.warning）并跳过 vol/dd 层的状态更新（不再静默喂固定
              capital 假数据）。
    """
    from trading import PortfolioState, decide_orders
    from live_state import (
        load_risk_state,
        save_risk_state,
        compute_risk_scale,
        update_risk_state,
        pred_hist_path_for,
        load_pred_hist,
    )

    p = dict(params or {})
    # dynamic_topk 只在回测（backtest_daily）实现；live 显式拒绝而非静默忽略，
    # 避免"回测开了、实盘没开"的隐性口径分叉。
    if bool(p.get("use_dynamic_topk", False)) or getattr(C, "USE_DYNAMIC_TOPK", False):
        raise ValueError(
            "USE_DYNAMIC_TOPK=1 目前仅回测支持；实盘(live/predict_tomorrow)未实现，"
            "请关闭后重跑（避免静默忽略造成回测≠实盘）。"
        )

    # 估当前持仓市值（用真实市价列），反推/采用现金（与回测的 total_value 口径一致）
    sym = today["symbol"].to_numpy()
    px_col = price_col if price_col in today.columns else "close"
    close = today[px_col].to_numpy(dtype=float)
    price_map = {sym[i]: close[i] for i in range(len(sym)) if close[i] == close[i] and close[i] > 0}
    held_mv = sum(price_map.get(s, 0.0) * sh for s, sh in current.items())
    if cash is not None:
        cash = float(cash)
        total_value = cash + held_mv  # 真实现金 + 真实市值 = 真实总估值
        total_value_known = True
    else:
        logger.warning(
            "\n" + "=" * 60 + "\n[警告] 未提供账户现金（account.csv 缺失或无 cash 列）！"
            "\n       本次用 --capital 兜底反推现金，并【跳过 vol/回撤两层风控"
            "\n       的状态更新】（不喂假净值）。请在 account.csv 中维护 cash 列。"
            "\n" + "=" * 60
        )
        cash = max(float(capital) - held_mv, 0.0)
        total_value = float(capital)
        total_value_known = False
    state = PortfolioState(cash=cash, holdings=dict(current))

    # 风控缩放：复用回测一致的五层风控（唯一实现在 src/risk.py）
    if risk_state_path is not None:
        rstate = load_risk_state(risk_state_path)
        latest = today["date"].max()
        risk_panel = panel if panel is not None else today
        # 持久化预测历史：供 IC 降仓层把新成熟 OOS 预测日打分入队
        pred_hist = load_pred_hist(pred_hist_path_for(risk_state_path))
        risk_scale, rstate = compute_risk_scale(
            risk_panel, latest, state=rstate, params=p, pred_hist=pred_hist
        )
        p["gross_leverage"] = (
            float(p.get("gross_leverage", getattr(C, "GROSS_LEVERAGE", 1.0))) * risk_scale
        )
        if total_value_known:
            # 真实总估值入账，驱动 vol/dd 两层
            rstate = update_risk_state(rstate, latest, total_value)
        else:
            rstate.last_update = str(pd.Timestamp(latest).date())
        save_risk_state(risk_state_path, rstate)

    return decide_orders(
        state,
        today,
        np.asarray(score, dtype=float),
        params=p,
        asof_date=today["date"].max(),
        price_col=px_col,
    )
