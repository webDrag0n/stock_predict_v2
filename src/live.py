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

import numpy as np
import pandas as pd

import config as C


def live_panel_start_date(train_window: int | None = None,
                          buffer_trading_days: int = 80) -> str:
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


def train_and_predict_latest(panel: pd.DataFrame,
                             feat_cols: list[str]) -> pd.DataFrame:
    """实盘训练+预测，**直接复用** walkforward._train_predict，与已验证回测同分布。

    注意：当前无生产调用方（生产路径是 ensemble_predict_latest，与每日重放
    回测同口径），保留供对照/调试。训练集=全部有标签行，预测对象=最新
    无标签日，于是 MODEL_OBJECTIVE(ret_weighted)、按日截面 z-score+MAD-winsorize、
    多 seed、正则化超参全部与回测一致。无未来信息：最新日的特征当晚已完整。

    返回 today 子表（含 close、prediction、rank_pct），按 prediction 降序。
    """
    import sys, os
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from walkforward import _train_predict

    latest = panel["date"].max()
    train = panel.dropna(subset=[C.TARGET]).copy()
    today = panel[panel["date"] == latest].copy()
    if today.empty or train.empty:
        raise ValueError("无最新无标签日或无有标签训练数据")

    # _train_predict 期望 train/test 含 'date' 列（按日截面归一化按 date 分组）。
    today["prediction"] = _train_predict(train, today, feat_cols)
    today["rank_pct"] = today["prediction"].rank(ascending=False, pct=True)
    return today.sort_values("prediction", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 新路径：多模型增量集成 + 共享 decide_orders（与每日重放回测同分布）
# ---------------------------------------------------------------------------
def _emit_fn(progress_cb):
    def _emit(phase, current, total, **extras):
        if progress_cb is None:
            return
        try:
            progress_cb(phase=phase, current=current, total=total, **extras)
        except Exception:
            pass
    return _emit


def predict_latest_with_ens(ens, panel: pd.DataFrame,
                            pred_hist: pd.DataFrame | None = None,
                            progress_cb=None) -> tuple[pd.DataFrame, dict]:
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

    _emit("滚动 IC 定权", 2, 3,
          n_hist_days=(int(pred_hist["date"].nunique())
                       if pred_hist is not None and len(pred_hist) else 0))
    if pred_hist is not None and len(pred_hist):
        for d in sorted(pred_hist["date"].unique()):
            sub = pred_hist[pred_hist["date"] == d]
            preds_d = {c[len("pred_"):]: sub[c].to_numpy(dtype=np.float64)
                       for c in sub.columns
                       if c.startswith("pred_") and c[len("pred_"):] in ens.models}
            if not preds_d:
                continue
            ens.observe(np.datetime64(d), sub[["symbol"]].reset_index(drop=True),
                        preds_d)
        ens.refresh_weights(panel, asof_date=latest)
        # 已成熟且成功打分的天数（各腿最小值）；不足窗口时 ensemble 内部
        # _recompute_weights 已保证退化为等权，这里打印说明避免误以为已定权。
        n_scored = min((len(ens._ic_hist[m]) for m in ens.models), default=0)
        if n_scored < ens.ic_window:
            print(f"[提示] 实盘 OOS IC 定权历史不足：已成熟打分 {n_scored} 天 "
                  f"（窗口 {ens.ic_window} 天，<3 天时完全等权）；"
                  f"每日运行 predict_tomorrow 后会逐日积累。")
    else:
        print("[提示] 无持久化 OOS 预测历史（pred_hist 为空），本次模型权重为等权；"
              "每日运行 predict_tomorrow 后会逐日积累。")
        ens.refresh_weights(panel, asof_date=latest)   # 无历史 → 等权

    _emit("预测最新日", 3, 3, n_today_rows=len(today))
    preds = ens.predict(today)
    score = ens.combine(preds, today["date"].to_numpy())
    today = today.copy()
    for m, pr in preds.items():
        today[f"pred_{m}"] = pr
    today["score"] = score
    today["prediction"] = score          # decide_orders/排序统一用 score
    today["rank_pct"] = pd.Series(score, index=today.index).rank(
        ascending=False, pct=True)
    today = today.sort_values("score", ascending=False).reset_index(drop=True)
    info = {"weights": dict(ens.weights), "ic_means": ens.ic_means(),
            "models": list(ens.models), "latest_date": latest,
            # 集成本体：模型门控（model_gate）需要把候选落盘/对比。
            # 仅内存引用，不随 info 落盘。
            "ensemble": ens}
    return today, info


def ensemble_predict_latest(panel: pd.DataFrame, feat_cols: list[str],
                            params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """用 IncrementalEnsemble 训练全部成熟历史、预测最新（无标签）日。

    与 backtest_daily.run_daily_replay 的训练/定权口径**逐项一致**：同样的多模型
    （gbm_l2 / gbm_ret_weighted / ic_linear）、同样的截面归一化、同样的滚动 IC 定权。
    实盘只跑一次：全量重建（force_full）一次性把尾部 TRAIN_WINDOW 训到位，再对最新
    无标签日预测（预测尾段与冻结路径共用 predict_latest_with_ens）。

    返回 (today, info)：
      today: 最新日横截面，附 ensemble 各模型分 + 融合 score + prediction + rank_pct，
             并保留 close/bb_*/rsi_14 等 decide_orders 需要的 meta 列。
      info : {"weights":..., "ic_means":..., "models":..., "latest_date":...,
              "ensemble": 训练好的集成本体}
    """
    import os
    import sys
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from ensemble import IncrementalEnsemble

    p = dict(params or {})
    fw = int(getattr(C, "FORWARD_WINDOW", 20))
    embargo = int(getattr(C, "EMBARGO_DAYS", fw))
    progress_cb = p.pop("progress_cb", None)
    _emit = _emit_fn(progress_cb)

    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    if today.empty:
        raise ValueError("无最新日横截面")
    _emit("初始化集成", 0, 3, latest_date=str(pd.Timestamp(latest).date()))

    ens = IncrementalEnsemble(
        feat_cols,
        models=p.get("ensemble_models") or None,
        n_seeds=p.get("n_seeds"), incr_trees=p.get("incr_trees"),
        rebuild_days=p.get("rebuild_days"), train_window=p.get("train_window"),
        ic_window=p.get("ic_window"))

    # 已成熟标签：date <= latest - (fw + embargo)。一次性全量重建。
    all_dates = np.sort(panel["date"].unique())
    di = int(np.searchsorted(all_dates, latest))
    mat_cut = all_dates[max(0, di - fw - embargo)]
    matured = panel[panel["date"] <= mat_cut]
    if matured.dropna(subset=[C.TARGET]).empty:
        raise ValueError("无已成熟标签可训练")
    _emit("全量重建中", 1, 3, n_train_rows=len(matured),
          models=",".join(ens.models))
    ens.update(matured, asof_date=latest, force_full=True)

    return predict_latest_with_ens(ens, panel, pred_hist=p.get("pred_hist"),
                                   progress_cb=progress_cb)


def champion_predict_latest(panel: pd.DataFrame,
                            pred_hist: pd.DataFrame | None = None,
                            progress_cb=None) -> tuple[pd.DataFrame, dict]:
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
    _emit("加载 champion 工件", 1, 3, model_id=manifest.get("model_id"),
          train_end_date=manifest.get("train_end_date"))
    today, info = predict_latest_with_ens(ens, panel, pred_hist=pred_hist,
                                          progress_cb=progress_cb)
    snap = manifest.get("config_snapshot") or {}
    model_store.refresh_champion(ens, snap.get("feat_cols") or [], snap)
    info["champion_id"] = manifest.get("model_id")
    info["action"] = "frozen"
    return today, info


def make_orders_from_state(today: pd.DataFrame, score: np.ndarray,
                           current: dict[str, int], capital: float,
                           params: dict | None = None,
                           price_col: str = "close",
                           risk_state_path: str | None = None,
                           panel: pd.DataFrame | None = None,
                           cash: float | None = None) -> pd.DataFrame:
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
              原始 close)，如实入账驱动 vol/dd 两层；未提供时打印醒目警告并
              跳过 vol/dd 层的状态更新（不再静默喂固定 capital 假数据）。
    """
    from trading import PortfolioState, decide_orders
    from live_state import (load_risk_state, save_risk_state, compute_risk_scale,
                            update_risk_state, pred_hist_path_for, load_pred_hist)

    p = dict(params or {})
    # dynamic_topk 只在回测（backtest_daily）实现；live 显式拒绝而非静默忽略，
    # 避免"回测开了、实盘没开"的隐性口径分叉。
    if bool(p.get("use_dynamic_topk", False)) or getattr(C, "USE_DYNAMIC_TOPK", False):
        raise ValueError(
            "USE_DYNAMIC_TOPK=1 目前仅回测支持；实盘(live/predict_tomorrow)未实现，"
            "请关闭后重跑（避免静默忽略造成回测≠实盘）。")

    # 估当前持仓市值（用真实市价列），反推/采用现金（与回测的 total_value 口径一致）
    sym = today["symbol"].to_numpy()
    px_col = price_col if price_col in today.columns else "close"
    close = today[px_col].to_numpy(dtype=float)
    price_map = {sym[i]: close[i] for i in range(len(sym))
                 if close[i] == close[i] and close[i] > 0}
    held_mv = sum(price_map.get(s, 0.0) * sh for s, sh in current.items())
    if cash is not None:
        cash = float(cash)
        total_value = cash + held_mv      # 真实现金 + 真实市值 = 真实总估值
        total_value_known = True
    else:
        print("=" * 60)
        print("[警告] 未提供账户现金（account.csv 缺失或无 cash 列）！")
        print("       本次用 --capital 兜底反推现金，并【跳过 vol/回撤两层风控")
        print("       的状态更新】（不喂假净值）。请在 account.csv 中维护 cash 列。")
        print("=" * 60)
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
            risk_panel, latest, state=rstate, params=p, pred_hist=pred_hist)
        p["gross_leverage"] = float(p.get("gross_leverage",
                                          getattr(C, "GROSS_LEVERAGE", 1.0))) * risk_scale
        if total_value_known:
            # 真实总估值入账，驱动 vol/dd 两层
            rstate = update_risk_state(rstate, latest, total_value)
        else:
            rstate.last_update = str(pd.Timestamp(latest).date())
        save_risk_state(risk_state_path, rstate)

    return decide_orders(state, today, np.asarray(score, dtype=float),
                         params=p, asof_date=today["date"].max(),
                         price_col=px_col)


