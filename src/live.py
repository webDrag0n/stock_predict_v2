"""实盘推理核心：把"截至今天的数据"变成"明日可执行的目标持仓 / 订单"。

设计原则
--------
1. 训练/实盘同分布：实盘**直接复用** walkforward._train_predict（见
   train_and_predict_latest），因此 MODEL_OBJECTIVE(ret_weighted)、按日截面
   z-score+MAD-winsorize、多 seed、正则化超参与已验证回测**逐项一致**。训练集
   换成"截至最新有标签日的全部历史"，预测对象换成"最新无标签日"。
2. 信号只用于排序：long_topk 只看预测排名，不把 prediction 当涨幅，与回测一致。
3. 换手缓冲：建仓阈值 topk，平仓阈值放宽到 topk+buffer，中间地带继续持有不新买，
   显著压低换手与交易成本（FW=5 的成本痛点）。
4. 无未来信息：只用 date<=今天 的特征；最新一天的特征是当日收盘后才完整的，
   因此信号当晚算出、次日执行。

train_full_model + predict_latest 是旧路径（朴素 l2、原始特征 nan→0、单 seed），
在 ret_weighted/全市场结论之前写的，已被 train_and_predict_latest 取代。保留仅
为向后兼容/单测，生产勿用。

纯函数为主，便于单测；CLI 编排在 predict_tomorrow.py。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C

# A股最小交易单位：1 手 = 100 股
LOT = 100


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


def train_full_model(panel: pd.DataFrame, feat_cols: list[str]) -> lgb.LGBMRegressor:
    """用 panel 中所有【有标签】行训练一个 LGB（参数同 walkforward）。"""
    train = panel.dropna(subset=[C.TARGET])
    Xtr = train[feat_cols].to_numpy(np.float32, copy=True)
    np.nan_to_num(Xtr, copy=False, nan=0.0)
    ytr = train[C.TARGET].to_numpy()
    wtr = train[C.WEIGHT].to_numpy() if C.WEIGHT in train else None

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        reg_lambda=0.1, objective="l2",
        random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(Xtr, ytr, sample_weight=wtr)
    return model


def predict_latest(model: lgb.LGBMRegressor, panel: pd.DataFrame,
                   feat_cols: list[str]) -> pd.DataFrame:
    """对 panel 中最新交易日的全 universe 行做预测。

    返回 today 子表（含 close、prediction、rank_pct），按 prediction 降序。
    rank_pct=0 是预测最高（最看好）的股票。
    """
    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    X = today[feat_cols].to_numpy(np.float32, copy=True)
    np.nan_to_num(X, copy=False, nan=0.0)
    today["prediction"] = np.asarray(model.predict(X), dtype=np.float64).ravel()
    # rank_pct: 0=最看好, 1=最不看好（百分位）
    today["rank_pct"] = today["prediction"].rank(ascending=False, pct=True)
    return today.sort_values("prediction", ascending=False).reset_index(drop=True)


def train_and_predict_latest(panel: pd.DataFrame,
                             feat_cols: list[str]) -> pd.DataFrame:
    """实盘训练+预测，**直接复用** walkforward._train_predict，与已验证回测同分布。

    取代 train_full_model + predict_latest 的旧路径（朴素 l2、原始特征 nan→0、
    单 seed、超参不一致）——那套是 ret_weighted/全市场结论之前写的，已漂移。
    这里把 _train_predict 当作唯一真相源：训练集=全部有标签行，预测对象=最新
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
def ensemble_predict_latest(panel: pd.DataFrame, feat_cols: list[str],
                            params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """用 IncrementalEnsemble 训练全部成熟历史、预测最新（无标签）日。

    与 backtest_daily.run_daily_replay 的训练/定权口径**逐项一致**：同样的多模型
    （gbm_l2 / gbm_ret_weighted / ic_linear）、同样的截面归一化、同样的滚动 IC 定权。
    实盘只跑一次：全量重建（force_full）一次性把尾部 TRAIN_WINDOW 训到位，再对最新
    无标签日预测；滚动 IC 权重用全部已成熟历史预测日打分（无泄露）。

    返回 (today, info)：
      today: 最新日横截面，附 ensemble 各模型分 + 融合 score + prediction + rank_pct，
             并保留 close/bb_*/rsi_14 等 decide_orders 需要的 meta 列。
      info : {"weights":..., "ic_means":..., "models":..., "latest_date":...}
    """
    import sys, os
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from ensemble import IncrementalEnsemble

    p = dict(params or {})
    fw = int(getattr(C, "FORWARD_WINDOW", 20))
    embargo = int(getattr(C, "EMBARGO_DAYS", fw))
    progress_cb = p.pop("progress_cb", None)

    def _emit(phase, current, total, **extras):
        if progress_cb is None:
            return
        try:
            progress_cb(phase=phase, current=current, total=total, **extras)
        except Exception:
            pass

    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    if today.empty:
        raise ValueError("无最新日横截面")
    _emit("初始化集成", 0, 3, latest_date=str(pd.Timestamp(latest).date()))

    ens = IncrementalEnsemble(
        feat_cols,
        models=tuple(p["ensemble_models"]) if p.get("ensemble_models") else None,
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

    # 滚动 IC 定权：把所有已成熟历史预测日的预测补记后打分。为效率，仅在尾部
    # IC_WINDOW + fw 天的横截面上各跑一次预测并 observe，再 refresh。
    ic_win = ens.ic_window
    score_dates = all_dates[all_dates <= mat_cut][-(ic_win + 1):]
    _emit("滚动 IC 定权", 2, 3, n_score_days=len(score_dates))
    for d in score_dates:
        dframe = panel[panel["date"] == d]
        if dframe.empty:
            continue
        dpreds = ens.predict(dframe)
        ens.observe(d, dframe, dpreds)
    ens.refresh_weights(panel, asof_date=latest)

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
            "models": list(ens.models), "latest_date": latest}
    return today, info


def make_orders_from_state(today: pd.DataFrame, score: np.ndarray,
                           current: dict[str, int], capital: float,
                           params: dict | None = None,
                           price_col: str = "adj_close",
                           risk_state_path: str | None = None,
                           panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """用共享 decide_orders 生成"明日可执行"逐笔订单（与回测同一决策代码）。

    把当前持仓 + 现金组成 PortfolioState（现金 = capital − 持仓市值估算），交给
    trading.decide_orders：ML 选池 + 布林/RSI 择时 → 精确到 (代码,方向,股数,挂单价)
    的订单，附触发原因。返回 trading.decide_orders 的订单表（实盘可直接照单执行）。

    Args:
        today: 最新日横截面，需含 symbol、price_col、bb_*、rsi_14 等。
        score: 与 today 对齐的融合排序分。
        current: 当前持仓 {代码: 股数}。
        capital: 组合总资金（不是现金）。
        params: 决策参数（topk、max_positions、max_weight、order_price_mode 等）。
        price_col: 用于估值与目标金额计算的价列，默认 "adj_close"。
        risk_state_path: 风控状态 JSON 路径；若提供则读取历史波动/回撤/IC，
                         计算 risk_scale 并更新状态文件。
        panel: 完整历史面板（多日期），用于 regime/IV 计算；缺省用 today 退化为单日线。
    """
    from trading import PortfolioState, decide_orders
    from live_state import (load_risk_state, save_risk_state,
                            compute_risk_scale, update_risk_state)

    p = dict(params or {})
    # 估当前持仓市值，反推现金（与回测的 total_value 口径一致）
    sym = today["symbol"].to_numpy()
    px_col = price_col if price_col in today.columns else "close"
    close = today[px_col].to_numpy(dtype=float)
    price_map = {sym[i]: close[i] for i in range(len(sym))
                 if close[i] == close[i] and close[i] > 0}
    held_mv = sum(price_map.get(s, 0.0) * sh for s, sh in current.items())
    cash = max(float(capital) - held_mv, 0.0)
    state = PortfolioState(cash=cash, holdings=dict(current))

    # 风控缩放：复用回测一致的五层风控
    if risk_state_path is not None:
        rstate = load_risk_state(risk_state_path)
        latest = today["date"].max()
        risk_panel = panel if panel is not None else today
        risk_scale, rstate = compute_risk_scale(
            risk_panel, latest, state=rstate, params=p)
        p["gross_leverage"] = float(p.get("gross_leverage",
                                          getattr(C, "GROSS_LEVERAGE", 1.0))) * risk_scale
        # 用 capital 作为今日总估值（保守），更新状态供次日使用
        rstate = update_risk_state(rstate, latest, float(capital))
        save_risk_state(risk_state_path, rstate)

    return decide_orders(state, today, np.asarray(score, dtype=float),
                         params=p, asof_date=today["date"].max(),
                         price_col=px_col)



def target_weights(today: pd.DataFrame) -> np.ndarray:
    """目标权重（和=GROSS_LEVERAGE）。

    复刻 strategy._long_topk 的"选 top-K%、组内按排名比例加权"逻辑，但因实盘
    永远只有【一个交易日】，直接向量化计算，规避 groupby.apply 在单组下返回
    二维结果的 pandas 行为（见 pandas3 gotchas）。结果与回测口径一致。
    """
    pred = today["prediction"].to_numpy()
    rank_pct = pd.Series(pred).rank(pct=True, method="first").to_numpy()
    selected = rank_pct > (1.0 - C.TOPK_RATIO)
    w = np.where(selected, rank_pct, 0.0)
    total = w.sum()
    if total <= 1e-12:
        return np.zeros_like(w, dtype=np.float64)
    return (C.GROSS_LEVERAGE * w / total).astype(np.float64)


def weights_to_shares(weights: np.ndarray, close: np.ndarray,
                      capital: float) -> np.ndarray:
    """目标权重 → 目标股数，向下取整到整手（100股）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = capital * weights / close
    raw = np.where(np.isfinite(raw), raw, 0.0)
    return (np.floor(raw / LOT) * LOT).astype(np.int64)


def apply_turnover_buffer(today: pd.DataFrame, target_shares: np.ndarray,
                          current: dict[str, int], topk: float,
                          buffer: float) -> np.ndarray:
    """换手缓冲：减少不必要的买卖。返回调整后的目标股数。

    规则（按预测排名 rank_pct，0=最看好）：
      - 已持有的票：只要还在 topk+buffer 缓冲区内 → 维持当前股数（不卖、不加仓）；
        跌出缓冲区 → 目标置 0（卖出）。
      - 未持有的票：进入 topk → 按 target_shares 买入；
        在缓冲带(topk~topk+buffer) → 不建仓（保持 0）。
    这样只有"新进 topk"和"跌出缓冲区"才产生交易，中间地带零换手。
    """
    out = target_shares.copy()
    sym = today["symbol"].to_numpy()
    rank_pct = today["rank_pct"].to_numpy()
    keep_band = topk + buffer
    for i in range(len(today)):
        held = current.get(sym[i], 0)
        in_topk = rank_pct[i] <= topk
        in_buffer = rank_pct[i] <= keep_band
        if held > 0:
            # 已持有：缓冲区内维持现状，跌出则清仓
            out[i] = held if in_buffer else 0
        else:
            # 未持有：仅 topk 内建仓，缓冲带不动
            out[i] = out[i] if in_topk else 0
    return out


def make_orders(today: pd.DataFrame, target_shares: np.ndarray,
                current: dict[str, int], min_notional: float = 0.0) -> pd.DataFrame:
    """目标股数 − 当前持仓 = 差额订单，附涨跌停过滤与碎单过滤。

    涨跌停约束（已有 limit_flag 特征，T日涨停=不能买入、跌停=不能卖出）：
      - delta>0(买) 且 当日涨停 → 拦截（方向标记为 '涨停拦截'）
      - delta<0(卖) 且 当日跌停 → 拦截（'跌停拦截'）
    min_notional：差额金额低于此阈值的碎单过滤为'持有'，进一步压成本。
    """
    sym = today["symbol"].to_numpy()
    name = today["name"].to_numpy() if "name" in today.columns else sym
    close = today["close"].to_numpy()
    limit = (today["limit_flag"].to_numpy() if "limit_flag" in today.columns
             else np.zeros(len(today)))
    pct_chg = (today["pct_chg"].to_numpy() if "pct_chg" in today.columns
               else np.zeros(len(today)))
    rank_pct = today["rank_pct"].to_numpy()

    rows = []
    for i in range(len(today)):
        cur = current.get(sym[i], 0)
        tgt = int(target_shares[i])
        delta = tgt - cur
        notional = abs(delta) * close[i]
        action, blocked = "持有", ""
        if delta != 0 and notional >= min_notional:
            # 涨跌停方向限制：limit_flag=1 表示当日触及±9.9%
            up_limit = limit[i] and pct_chg[i] > 0
            down_limit = limit[i] and pct_chg[i] < 0
            if delta > 0 and up_limit:
                action, blocked = "买(涨停拦截)", "涨停不可买"
            elif delta < 0 and down_limit:
                action, blocked = "卖(跌停拦截)", "跌停不可卖"
            else:
                action = "买入" if delta > 0 else "卖出"
        elif delta != 0:
            action = "持有(碎单过滤)"
            delta = 0
        # 只保留有持仓变动或当前/目标非零的行
        if tgt == 0 and cur == 0:
            continue
        rows.append({
            "代码": sym[i], "名称": name[i], "方向": action,
            "目标股数": tgt, "当前股数": cur, "差额股数": delta,
            "现价": round(float(close[i]), 2),
            "预计金额": round(abs(delta) * float(close[i]), 2),
            "预测排名%": round(float(rank_pct[i]) * 100, 1),
            "备注": blocked,
        })
    orders = pd.DataFrame(rows)
    if len(orders):
        # 卖在前、买在后（先腾出资金），组内按预测排名
        order_key = {"卖出": 0, "卖(跌停拦截)": 0, "买入": 1, "买(涨停拦截)": 1,
                     "持有": 2, "持有(碎单过滤)": 2}
        orders["_k"] = orders["方向"].map(lambda a: order_key.get(a, 3))
        orders = orders.sort_values(["_k", "预测排名%"]).drop(columns="_k").reset_index(drop=True)
    return orders


def estimate_cost(orders: pd.DataFrame) -> dict:
    """估算订单的交易成本（佣金双边 + 印花税卖出单边）。"""
    if not len(orders):
        return {"buy_notional": 0.0, "sell_notional": 0.0, "commission": 0.0,
                "stamp_duty": 0.0, "total": 0.0}
    buys = orders[orders["方向"] == "买入"]["预计金额"].sum()
    sells = orders[orders["方向"] == "卖出"]["预计金额"].sum()
    commission = (buys + sells) * C.COMMISSION_RATE
    stamp = sells * C.STAMP_DUTY_RATE
    return {"buy_notional": float(buys), "sell_notional": float(sells),
            "commission": float(commission), "stamp_duty": float(stamp),
            "total": float(commission + stamp)}
