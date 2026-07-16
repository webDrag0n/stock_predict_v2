"""每日重放回测引擎 —— 把共享每日交易决策(trading.py)在历史上一天天重放。

与旧的 run_backtest_periodic（固定周期、向量化、只算组合收益率）不同，这里是
**事件级逐笔模拟**，完全对应需求：

  每个交易日 t（按时间顺序遍历）：
    1. 增量更新 ensemble（只用 ≤ t−FORWARD_WINDOW 的已成熟标签行续训/重建）；
    2. 滚动 IC 定权（只用已成熟历史预测日打分）；
    3. 对 t 的横截面预测 → 融合排序分；
    4. decide_orders：ML 选池 + 布林/RSI 择时 → 精确逐笔订单（决策在 t 收盘后）；
    5. simulate_fills：用 t 的【下一交易日】OHLC + 涨跌停带撮合每笔订单，
       更新 现金/持仓（PortfolioState）；
    6. 记录：当日逐笔成交、收盘组合状态（现金 / 股票市值 / 总值 / 持仓数）。

防泄露：第 4 步只用 ≤ t 的特征；第 5 步用 t 的下一日行情。决策→执行天然 1 日
滞后。模型训练只见 ≤ t−FORWARD_WINDOW 的成熟标签。

关键不变量：每日 现金 + 持仓市值 = 总资产，逐日连续。本引擎在 STRICT_CHECK 下
对每个成交日校验"现金变动 == −买入额 −成本 +卖出额"，发现偏差即报错。

产出（供 pipeline 步骤画图）：
  · daily   : 每日 [date, cash, stock_value, total_value, n_positions, gross_ret,
              turnover, cost, n_fill, n_reject]
  · trades  : 全部逐笔成交明细（date, symbol, side, fill_shares, fill_price,
              notional, cost, reason, status）
  · weights : 每个更新点的各模型 IC 权重（供"模型权重随时间"图）
  · 最终 PortfolioState
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import config as C
from ensemble import IncrementalEnsemble
from trading import PortfolioState, decide_orders, simulate_fills


# 撮合 / 估值所需的次日行情列
_BAR_COLS = ("symbol", "open", "high", "low", "close", "amount",
             "pre_close", "limit_flag", "pct_chg")


def run_daily_replay(panel: pd.DataFrame, feat_cols: list[str],
                     initial_cash: float = None,
                     initial_holdings: dict[str, int] | None = None,
                     params: dict | None = None,
                     start_date=None, end_date=None,
                     progress: bool = True) -> dict:
    """逐日重放回测。panel 需为 keep_trade_cols=True 的面板（含 OHLC + 布林带 meta）。

    Args:
        panel            : 命名因子面板（含 date, symbol, 因子.., responder_6,
                           open/high/low/close, bb_upper_20/bb_lower_20, rsi_14,
                           return_1d/pct_chg, amount, limit_flag）。
        feat_cols        : 喂给 ensemble 的因子列（list_factor_columns 的结果，
                           已排除 OHLC/布林带 meta）。
        initial_cash     : 初始现金（默认 C.INITIAL_CAPITAL）。
        initial_holdings : 初始持仓 {代码: 股数}（默认空仓）。
        params           : 决策/成交参数覆盖（见 trading.decide_orders/simulate_fills）。
        start_date/end_date : 限定回测【交易】区间（训练仍可用更早历史）。None=全程。

    Returns:
        dict(daily, trades, weights, state, meta)
    """
    p = dict(params or {})
    initial_cash = float(initial_cash if initial_cash is not None
                         else C.INITIAL_CAPITAL)
    fw = int(getattr(C, "FORWARD_WINDOW", 20))
    embargo = int(getattr(C, "EMBARGO_DAYS", fw))
    warmup = int(p.get("warmup_days", getattr(C, "TRAIN_WINDOW", 500)))
    strict = bool(p.get("strict_check", True))

    # 参数扫描加速：模型训练/预测与"决策+撮合"参数（min_notional/缓冲带/挂单价/
    # 布林RSI阈值）完全解耦。score_cache（{date -> {symbol: score}}）一旦提供，
    # 本引擎跳过全部模型工作，直接复用缓存分做决策+撮合 —— 一次训练、多次重放。
    #   score_cache=None  且 collect_scores=True : 正常训练，并在返回 dict 里附带
    #                      "score_cache" 供后续重放复用。
    #   score_cache=<dict>: 跳过模型，按缓存分重放（缓存缺某日则该日空仓）。
    score_cache = p.get("score_cache")
    collect_scores = bool(p.get("collect_scores", False))
    use_cache = score_cache is not None
    collected: dict = {} if collect_scores else None
    # 进度回调：run_session / pipeline 注入。签名 cb(phase, current, total, **extras)。
    # 进度回调是"尽力而为"——失败绝不能影响主回测。
    progress_cb = p.pop("progress_cb", None)

    # 风险控制层参数（默认开波动目标化）。target_vol=年化目标波动；高于它则去杠杆。
    # dd_brake：回撤超 dd_threshold 时再乘 dd_scale 进一步降暴露。
    use_vol_target = bool(p.get("use_vol_target", getattr(C, "USE_VOL_TARGET", True)))
    target_vol = float(p.get("target_vol", getattr(C, "TARGET_VOL", 0.15)))
    vol_window = int(p.get("vol_window", getattr(C, "VOL_WINDOW", 20)))
    use_dd_brake = bool(p.get("use_dd_brake", getattr(C, "USE_DD_BRAKE", True)))
    dd_threshold = float(p.get("dd_threshold", getattr(C, "DD_THRESHOLD", 0.20)))
    dd_scale = float(p.get("dd_scale", getattr(C, "DD_SCALE", 0.5)))
    # IC信号失效降仓：当近期OOS rank-IC滑入负区则减少暴露。
    # 防泄露：只用【已成熟】预测日的IC (d_prev + FW ≤ d)。
    # scale_ic = linear interp: IC≥ic_full_scale→1.0, IC≤ic_zero_scale→ic_min_scale
    use_ic_scale = bool(p.get("use_ic_scale", getattr(C, "USE_IC_SCALE", True)))
    ic_scale_window = int(p.get("ic_scale_window", getattr(C, "IC_SCALE_WINDOW", 20)))
    ic_full_scale = float(p.get("ic_full_scale", getattr(C, "IC_FULL_SCALE", 0.02)))
    ic_zero_scale = float(p.get("ic_zero_scale", getattr(C, "IC_ZERO_SCALE", -0.01)))
    ic_min_scale = float(p.get("ic_min_scale", getattr(C, "IC_MIN_SCALE", 0.20)))
    # 动态 topk 集中度：根据近期 OOS rank-IC 调整持仓只数。IC 高时集中吃头部 edge，
    # IC 低时分散降低噪音与换手。与 IC 降仓共用同一个成熟 IC 队列，无泄露。
    use_dynamic_topk = bool(p.get("use_dynamic_topk", getattr(C, "USE_DYNAMIC_TOPK", False)))
    dynamic_topk_ic_full = float(p.get("dynamic_topk_ic_full", getattr(C, "DYNAMIC_TOPK_IC_FULL", 0.02)))
    dynamic_topk_ic_zero = float(p.get("dynamic_topk_ic_zero", getattr(C, "DYNAMIC_TOPK_IC_ZERO", -0.01)))
    dynamic_topk_low = float(p.get("dynamic_topk_low", getattr(C, "DYNAMIC_TOPK_LOW", 0.10)))
    dynamic_topk_high = float(p.get("dynamic_topk_high", getattr(C, "DYNAMIC_TOPK_HIGH", 0.25)))
    # Regime 择时空仓避险（最果断的一层）：当【大盘趋势】转空（等权市场跌破均线 /
    # 趋势窗收益为负）时整体降到 regime_min（默认 0 = 全空仓避险），而非渐进降仓。
    # 区别于 IC 降仓(渐进、滞后)——这是对"市场方向"的果断进出。仅用已实现行情，防泄露。
    # 注意:这层只防"市场下行型"敌对(如2024崩盘)，对"市场上行但选股失效"型无能为力
    # (后者市场在涨、趋势滤网不触发)——诚实地各管各的。
    use_regime = bool(p.get("use_regime", getattr(C, "USE_REGIME_FILTER", True)))
    regime_window = int(p.get("regime_window", getattr(C, "REGIME_WINDOW", 20)))
    regime_floor_ret = float(p.get("regime_floor_ret", getattr(C, "REGIME_FLOOR_RET", 0.0)))
    regime_min = float(p.get("regime_min", getattr(C, "REGIME_MIN", 0.0)))
    # IV-based gross leverage scaling: high implied vol → de-risk.
    use_iv_scale = bool(p.get("use_iv_scale", getattr(C, "USE_IV_SCALE", False)))
    iv_scale_window = int(p.get("iv_scale_window", getattr(C, "IV_SCALE_WINDOW", 60)))
    iv_feature = p.get("iv_feature_for_scale", getattr(C, "IV_FEATURE_FOR_SCALE", "opt_vix_like"))
    iv_min_scale = float(p.get("iv_min_scale", getattr(C, "IV_MIN_SCALE", 0.5)))
    iv_mid_scale = float(p.get("iv_mid_scale", getattr(C, "IV_MID_SCALE", 1.0)))

    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Pre-compute trailing betas if beta constraint is enabled.
    # Beta = slope of stock return_1d vs equal-weight market return_1d over the
    # trailing BETA_LOOKBACK days per symbol. Leak-free: uses only past returns.
    use_beta_constraint = bool(p.get("use_beta_constraint",
                                     getattr(C, "USE_BETA_CONSTRAINT", False)))
    max_portfolio_beta = float(p.get("max_portfolio_beta",
                                      getattr(C, "MAX_PORTFOLIO_BETA", 1.10)))
    beta_lookback = int(p.get("beta_lookback", getattr(C, "BETA_LOOKBACK", 60)))
    if use_beta_constraint and "return_1d" in panel.columns:
        panel = _add_trailing_beta(panel, lookback=beta_lookback)

    all_dates = np.sort(panel["date"].unique())

    # 决策日区间：默认从"有足够训练历史 + 标签成熟"的第一天开始，到倒数第二天
    # （最后一天没有"下一交易日"可撮合）。
    first_decision_i = warmup + embargo
    decision_dates = all_dates[first_decision_i:-1]
    if start_date is not None:
        decision_dates = decision_dates[decision_dates >= np.datetime64(start_date)]
    if end_date is not None:
        decision_dates = decision_dates[decision_dates <= np.datetime64(end_date)]
    if len(decision_dates) == 0:
        raise ValueError("决策日为空：历史不足以预热，或日期区间过窄。")

    # 预切：把面板按日期切成 (start, end) 行索引。避免把每个交易日存成一个独立
    # DataFrame（会复制/引用大量内存），也避免每天做整表布尔索引。
    row_dates = panel["date"].to_numpy()
    _uniq_dates, _starts, _counts = np.unique(
        row_dates, return_index=True, return_counts=True)
    _date_start = {d: int(s) for d, s in zip(_uniq_dates, _starts)}
    _date_end = {d: int(s + c) for d, s, c in zip(_uniq_dates, _starts, _counts)}
    date_to_next = {all_dates[i]: all_dates[i + 1]
                    for i in range(len(all_dates) - 1)}

    def _slice_date(d):
        return panel.iloc[_date_start[d]:_date_end[d]]

    # Regime 择时：预计算每个交易日的【大盘 regime 在险标志】（仅用 ≤ 当日行情，防泄露）。
    # 等权市场日收益 → 过去 regime_window 日累计收益；≤ regime_floor_ret 视为下行 regime。
    _regime_risk_on = {}
    if use_regime:
        _rc = ("return_1d" if "return_1d" in panel.columns
               else ("pct_chg" if "pct_chg" in panel.columns else None))
        if _rc is not None:
            _mkt = panel.groupby("date")[_rc].mean().sort_index()
            if _rc == "pct_chg":
                _mkt = _mkt / 100.0
            _mkt_dates = _mkt.index.to_numpy()
            _mkt_v = _mkt.to_numpy(dtype=np.float64)
            for i, dd_ in enumerate(_mkt_dates):
                lo = max(0, i - regime_window + 1)
                trail = float(np.prod(1.0 + _mkt_v[lo:i + 1]) - 1.0)
                _regime_risk_on[dd_] = (trail > regime_floor_ret)

    # IV-based leverage scaling: pre-compute per-date scale from trailing history.
    # The IV feature is already lagged by 1 trading day in the data loader, so it
    # is known at decision time without look-ahead.
    _iv_scale = {}
    if use_iv_scale and iv_feature in panel.columns:
        _iv_series = panel.groupby("date")[iv_feature].first().sort_index()
        _iv_dates = _iv_series.index.to_numpy()
        _iv_v = _iv_series.to_numpy(dtype=np.float64)
        for i, dd_ in enumerate(_iv_dates):
            lo = max(0, i - iv_scale_window + 1)
            trail = _iv_v[lo:i]
            if len(trail) < iv_scale_window // 3 or not np.isfinite(_iv_v[i]):
                _iv_scale[dd_] = 1.0
                continue
            lo_q, hi_q = np.nanpercentile(trail, [33.0, 66.0])
            v = _iv_v[i]
            if v >= hi_q:
                _iv_scale[dd_] = iv_min_scale
            elif v <= lo_q:
                _iv_scale[dd_] = 1.0
            else:
                # linear interp between mid and high tercile
                t = (v - lo_q) / (hi_q - lo_q + 1e-12)
                _iv_scale[dd_] = 1.0 + t * (iv_min_scale - 1.0)

    ens = None
    if not use_cache:
        ens = IncrementalEnsemble(feat_cols,
                                  models=tuple(p["ensemble_models"]) if p.get("ensemble_models") else None,
                                  n_seeds=p.get("n_seeds"),
                                  incr_trees=p.get("incr_trees"),
                                  rebuild_days=p.get("rebuild_days"),
                                  train_window=p.get("train_window"),
                                  ic_window=p.get("ic_window"))

    state = PortfolioState(cash=initial_cash,
                           holdings=dict(initial_holdings or {}))

    daily_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    weight_rows: list[dict] = []
    # IC 信号失效降仓：维护一个 (pred_date -> {sym: score}) 的预测历史，
    # 和一个已成熟 IC 的滑动队列。pred_date 的 IC 在 pred_date + FW 日后才能计算。
    from collections import deque
    _pred_hist_ic: dict = {}    # date -> {sym: score}（仅用于 IC 打分，与 ensemble 的无关）
    _ic_queue: deque = deque(maxlen=ic_scale_window)   # 最近 ic_scale_window 个已成熟 IC
    _scored_ic: set = set()

    t0 = time.time()
    n_dec = len(decision_dates)
    if progress_cb is not None:
        try:
            progress_cb(phase="回放启动", current=0, total=max(n_dec, 1),
                        n_decision_days=int(n_dec),
                        date_range=(f"{pd.Timestamp(all_dates[first_decision_i]).date()} ~ "
                                    f"{pd.Timestamp(decision_dates[-1]).date()}"))
        except Exception:
            pass

    for k, d in enumerate(decision_dates):
        today = _slice_date(d)
        sym_today = today["symbol"].to_numpy()

        if use_cache:
            # 复用缓存分：跳过全部模型工作。缺某日 → 全 0 分（决策层等同空仓）。
            sc_map = score_cache.get(d, {})
            score = np.array([sc_map.get(s, np.nan) for s in sym_today],
                             dtype=np.float64)
            if not np.isfinite(score).any():
                score = np.zeros(len(today), dtype=np.float64)
            else:
                score = np.nan_to_num(score, nan=np.nanmin(score))
            w = None
        else:
            # --- 1) 已成熟标签：date <= d - (fw + embargo) ---
            di = np.searchsorted(all_dates, d)
            mat_cut_i = max(0, di - fw - embargo)
            mat_cut = all_dates[mat_cut_i]
            matured = panel.iloc[:_date_end[mat_cut]]

            # --- 1b) 增量更新 + 滚动 IC 定权 ---
            ens.update(matured, asof_date=d)
            ens.refresh_weights(panel, asof_date=d)
            w = ens.weights
            weight_rows.append({"date": d, **{f"w_{m}": w.get(m, 0.0) for m in ens.models},
                                **{f"ic_{m}": v for m, v in ens.ic_means().items()}})

            # --- 2) 决策日横截面预测 ---
            preds = ens.predict(today)
            score = ens.combine(preds, today["date"].to_numpy())
            ens.observe(d, today, preds)   # 存预测供日后成熟定权
            if collected is not None:
                collected[d] = {sym_today[i]: float(score[i])
                                for i in range(len(sym_today))}

        # --- IC 信号失效降仓：记录今日预测分，供 FW 天后成熟打分 ---
        if use_ic_scale:
            _pred_hist_ic[d] = {sym_today[i]: float(score[i]) for i in range(len(sym_today))}

        # --- 3) 决策（收盘后）---
        # 风险控制层（仅用【已实现】历史，无未来信息）：
        #   A. 波动目标化：按 trailing 实现波动缩放 gross（高波动去杠杆）
        #   B. 回撤刹车：超阈值后再乘 dd_scale
        #   C. IC 信号失效降仓：近期 OOS rank-IC 衰入负区时降暴露（最直接针对
        #      "近窗所有 leg 失效仍持仓挨亏"的问题，IC 只用已成熟预测，防泄露）
        risk_scale = 1.0
        ic_scale = 1.0
        if use_ic_scale:
            # 把新成熟的预测日打 IC（date + FW ≤ d = 标签已成熟）
            from scipy.stats import spearmanr
            di = np.searchsorted(all_dates, d)
            for d_prev in list(_pred_hist_ic):
                if d_prev in _scored_ic:
                    continue
                # 只用真正成熟的标签
                d_prev_idx = int(np.searchsorted(all_dates, d_prev))
                mature_idx = d_prev_idx + fw
                if mature_idx >= len(all_dates) or all_dates[mature_idx] > d:
                    continue
                sub = _slice_date(d_prev) if d_prev in _date_start else None
                if sub is None:
                    _scored_ic.add(d_prev); continue
                pm = _pred_hist_ic[d_prev]
                syms = [s for s in pm if s in sub["symbol"].to_numpy()]
                if len(syms) < 30:
                    _scored_ic.add(d_prev); continue
                sy = sub.set_index("symbol")
                tgt = sy.loc[syms, C.TARGET].to_numpy(float)
                sc_ = np.array([pm[s] for s in syms], float)
                ok = np.isfinite(tgt) & np.isfinite(sc_)
                if ok.sum() < 20:
                    _scored_ic.add(d_prev); continue
                ic_val = spearmanr(sc_[ok], tgt[ok]).correlation
                if ic_val == ic_val:
                    _ic_queue.append(float(ic_val))
                _scored_ic.add(d_prev)
                del _pred_hist_ic[d_prev]   # 释放内存
            # IC 缩放：线性插值 [ic_zero_scale, ic_full_scale] → [ic_min_scale, 1.0]
            roll_ic = None
            if len(_ic_queue) >= max(3, ic_scale_window // 4):
                roll_ic = float(np.mean(_ic_queue))
                t = ((roll_ic - ic_zero_scale) /
                     (ic_full_scale - ic_zero_scale + 1e-9))
                t = max(0.0, min(1.0, t))
                ic_scale = ic_min_scale + t * (1.0 - ic_min_scale)
        if (use_vol_target or use_dd_brake) and len(daily_rows) >= 6:
            tvs = np.array([r["total_value"] for r in daily_rows], dtype=np.float64)
            rets = tvs[1:] / tvs[:-1] - 1.0
            if use_vol_target and len(rets) >= 5:
                rv = float(np.std(rets[-vol_window:])) + 1e-12
                target_daily = target_vol / np.sqrt(252.0)
                risk_scale = min(1.0, target_daily / rv)
            if use_dd_brake:
                dd = tvs[-1] / tvs.max() - 1.0
                if dd < -dd_threshold:
                    risk_scale *= dd_scale
        risk_scale *= ic_scale   # 三层叠加：vol × dd × IC
        # 第四层 regime 择时：大盘下行 regime 时整体降到 regime_min（默认 0=空仓）。
        # 用决策日 d 的【已实现】趋势标志（仅 ≤ d 行情），次日执行，防泄露。
        if use_regime and not _regime_risk_on.get(d, True):
            risk_scale *= regime_min
        # 第五层 IV regime 降杠杆：高隐含波动率时降低 gross exposure。
        if use_iv_scale:
            risk_scale *= _iv_scale.get(d, 1.0)
        day_p = dict(p)
        day_p["gross_leverage"] = float(p.get("gross_leverage",
                                              getattr(C, "GROSS_LEVERAGE", 1.0))) * risk_scale

        # 动态 topk：根据近期 IC 调整集中度。IC 高->集中，IC 低->分散。
        if use_dynamic_topk and roll_ic is not None:
            t = ((roll_ic - dynamic_topk_ic_zero) /
                 (dynamic_topk_ic_full - dynamic_topk_ic_zero + 1e-9))
            t = max(0.0, min(1.0, t))
            # t=1 at IC_FULL -> low topk; t=0 at IC_ZERO -> high topk
            dyn_topk = dynamic_topk_high - t * (dynamic_topk_high - dynamic_topk_low)
            dyn_topk = max(0.02, min(0.50, dyn_topk))
            day_p["topk"] = float(dyn_topk)
            # preserve buffer relative to original hold_ratio
            base_topk = float(p.get("topk", getattr(C, "TOPK_RATIO", 0.15)))
            base_hold = float(p.get("hold_ratio", getattr(C, "HOLD_RATIO", base_topk)))
            buffer = max(0.0, base_hold - base_topk)
            day_p["hold_ratio"] = min(0.55, dyn_topk + buffer)

        # rebalance_every: 只在每第 N 个决策日重新决策，其余日持有不动（降频闸门）。
        rebal_every = int(p.get("rebalance_every", getattr(C, "REBALANCE_EVERY", 1)))
        if rebal_every > 1 and (k % rebal_every) != 0:
            orders = decide_orders(state, today.iloc[0:0], np.array([]), params=day_p,
                                   asof_date=d)
        else:
            orders = decide_orders(state, today, score, params=day_p, asof_date=d)

        # --- 4) 次日撮合 ---
        nd = date_to_next.get(d)
        next_bars = _slice_date(nd) if nd in _date_start else None
        if next_bars is not None:
            keep = [c for c in _BAR_COLS if c in next_bars.columns]
            next_bars = next_bars[keep]
        cash_before = state.cash
        if len(orders) and next_bars is not None:
            fill_res = simulate_fills(state, orders, next_bars, params=p, t1_lock=True)
        else:
            fill_res = {"fills": pd.DataFrame(), "n_fill": 0, "n_reject": len(orders),
                        "buy_notional": 0.0, "sell_notional": 0.0,
                        "commission": 0.0, "stamp": 0.0, "impact": 0.0,
                        "total_cost": 0.0}

        # --- 5) 收盘估值：用【次日】close 给收盘后的持仓估值（持仓是次日成交后的）---
        val_bars = next_bars if next_bars is not None else today
        price_map = _price_map(val_bars)
        stock_val = state.market_value(price_map)
        total_val = state.cash + stock_val

        # --- 守恒校验 ---
        if strict:
            expected = (cash_before - fill_res["buy_notional"]
                        + fill_res["sell_notional"] - fill_res["total_cost"])
            if abs(expected - state.cash) > max(1.0, abs(expected) * 1e-6):
                raise AssertionError(
                    f"现金守恒被破坏 @ {pd.Timestamp(d).date()}: "
                    f"期望 {expected:.2f} 实际 {state.cash:.2f} "
                    f"(买{fill_res['buy_notional']:.2f} 卖{fill_res['sell_notional']:.2f} "
                    f"成本{fill_res['total_cost']:.2f})")

        turnover = (fill_res["buy_notional"] + fill_res["sell_notional"])
        daily_rows.append({
            "date": nd,                       # 成交发生在下一交易日，归到该日
            "decision_date": d,
            "cash": state.cash,
            "stock_value": stock_val,
            "total_value": total_val,
            "n_positions": len(state.held_symbols()),
            "turnover": turnover,
            "cost": fill_res["total_cost"],
            "commission": fill_res["commission"],
            "stamp": fill_res["stamp"],
            "impact": fill_res["impact"],
            "n_fill": fill_res["n_fill"],
            "n_reject": fill_res["n_reject"],
            "risk_scale": risk_scale,
        })
        fdf = fill_res["fills"]
        if len(fdf):
            fdf = fdf.copy()
            fdf.insert(0, "date", nd)
            trade_frames.append(fdf)

        if progress and (k % 5 == 0 or k == n_dec - 1):
            el = time.time() - t0
            eta = el / (k + 1) * (n_dec - k - 1)
            wtxt = ("缓存分" if use_cache
                    else str({m: round(w.get(m, 0), 2) for m in ens.models}))
            print(f"  [{k+1}/{n_dec}] {pd.Timestamp(d).date()} "
                  f"总值¥{total_val:,.0f} 持仓{len(state.held_symbols())} "
                  f"权重{wtxt} "
                  f"(elapsed {el:.0f}s eta {eta:.0f}s)", flush=True)
            if progress_cb is not None:
                try:
                    progress_cb(
                        phase="回放决策日",
                        current=k + 1,
                        total=n_dec,
                        total_value=round(float(total_val), 0),
                        n_positions=int(len(state.held_symbols())),
                        n_fill=int(fill_res["n_fill"]),
                        n_reject=int(fill_res["n_reject"]),
                        date=str(pd.Timestamp(d).date()),
                        weights=wtxt,
                    )
                except Exception:
                    pass

    if progress_cb is not None:
        try:
            progress_cb(phase="回放完成", current=n_dec, total=max(n_dec, 1),
                        n_decision_days=int(n_dec))
        except Exception:
            pass

    daily = pd.DataFrame(daily_rows)
    trades = (pd.concat(trade_frames, ignore_index=True)
              if trade_frames else pd.DataFrame())
    weights = pd.DataFrame(weight_rows)

    # 期末以最后持仓的最后可得价格估值已在 daily 最后一行；补充总收益
    meta = {
        "initial_cash": initial_cash,
        "initial_holdings": dict(initial_holdings or {}),
        "n_decision_days": int(n_dec),
        "elapsed_sec": round(time.time() - t0, 1),
        "models": (list(ens.models) if ens is not None else ["<cached>"]),
        "final_weights": (dict(ens.weights) if ens is not None else {}),
        "used_cache": use_cache,
    }
    out = {"daily": daily, "trades": trades, "weights": weights,
           "state": state, "meta": meta}
    if collected is not None:
        out["score_cache"] = collected
    return out


def _price_map(bars: pd.DataFrame) -> dict[str, float]:
    """symbol -> close（估值用）。"""
    if "close" not in bars.columns:
        return {}
    sym = bars["symbol"].to_numpy()
    close = bars["close"].to_numpy(dtype=np.float64)
    return {sym[i]: close[i] for i in range(len(sym))
            if close[i] == close[i] and close[i] > 0}


def compute_trade_stats(trades: pd.DataFrame) -> dict:
    """从逐笔成交计算交易级统计：买卖笔数、成交率、按代码配对的已实现盈亏。

    用 FIFO 把每只股的买入与后续卖出配对，估算每笔卖出的已实现收益（不含跨段
    复杂分摊，足够给出胜率/平均盈亏/盈亏比的画像）。
    """
    if trades is None or len(trades) == 0:
        return {"n_buy": 0, "n_sell": 0, "n_round_trips": 0, "win_rate": float("nan"),
                "avg_win": float("nan"), "avg_loss": float("nan"),
                "profit_factor": float("nan"), "realized_pnl": 0.0,
                "round_trips": pd.DataFrame()}
    f = trades[trades["filled"]] if "filled" in trades.columns else trades
    f = f.sort_values("date")
    n_buy = int((f["side"] == "buy").sum())
    n_sell = int((f["side"] == "sell").sum())

    # FIFO 配对（每股一队列）
    from collections import defaultdict, deque
    lots: dict[str, deque] = defaultdict(deque)
    rt_rows = []
    for r in f.itertuples(index=False):
        if r.fill_shares <= 0 or r.fill_price != r.fill_price:
            continue
        if r.side == "buy":
            lots[r.symbol].append([r.fill_shares, float(r.fill_price), r.date])
        else:
            remain = r.fill_shares
            sell_px = float(r.fill_price)
            while remain > 0 and lots[r.symbol]:
                lot = lots[r.symbol][0]
                take = min(remain, lot[0])
                pnl = (sell_px - lot[1]) * take
                rt_rows.append({"symbol": r.symbol, "buy_date": lot[2],
                                "sell_date": r.date, "shares": take,
                                "buy_price": lot[1], "sell_price": sell_px,
                                "pnl": pnl,
                                "ret": (sell_px / lot[1] - 1.0) if lot[1] else np.nan})
                lot[0] -= take
                remain -= take
                if lot[0] <= 0:
                    lots[r.symbol].popleft()

    rt = pd.DataFrame(rt_rows)
    if len(rt):
        wins = rt[rt["pnl"] > 0]["pnl"]
        losses = rt[rt["pnl"] < 0]["pnl"]
        win_rate = float((rt["pnl"] > 0).mean())
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")
        realized = float(rt["pnl"].sum())
    else:
        win_rate = avg_win = avg_loss = float("nan")
        pf = float("nan"); realized = 0.0

    return {"n_buy": n_buy, "n_sell": n_sell, "n_round_trips": int(len(rt)),
            "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
            "profit_factor": pf, "realized_pnl": realized, "round_trips": rt}


def _add_trailing_beta(panel: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Add a trailing-beta column to the panel.

    For each symbol, regress its return_1d on the equal-weight market return_1d
    over the trailing `lookback` days. The market return on each date is the
    cross-sectional mean of return_1d. Leak-free: beta at date t uses only
    returns from dates <= t.
    """
    if "return_1d" not in panel.columns:
        return panel
    df = panel.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    mkt = df.groupby("date")["return_1d"].transform("mean")
    df["_mkt_ret"] = mkt

    min_periods = max(10, lookback // 3)
    g = df.groupby("symbol", group_keys=False)
    x = df["_mkt_ret"]
    y = df["return_1d"]
    x_mean = g["_mkt_ret"].transform(lambda s: s.rolling(lookback, min_periods=min_periods).mean())
    y_mean = g["return_1d"].transform(lambda s: s.rolling(lookback, min_periods=min_periods).mean())
    xy_mean = g.apply(lambda sub: (sub["_mkt_ret"] * sub["return_1d"]).rolling(lookback, min_periods=min_periods).mean())
    x2_mean = g["_mkt_ret"].transform(lambda s: (s * s).rolling(lookback, min_periods=min_periods).mean())
    cov = xy_mean - x_mean * y_mean
    var = x2_mean - x_mean ** 2
    beta = cov / var
    beta[(var <= 1e-12) | (~np.isfinite(var))] = np.nan
    df["beta"] = beta
    df = df.drop(columns=["_mkt_ret"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)
