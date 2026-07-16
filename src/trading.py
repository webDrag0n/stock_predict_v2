"""共享每日交易决策核心 —— 实盘(次日)与回测重放共用同一套代码。

设计哲学（与项目既有 live.py 复用 walkforward 的思路一致）：决策逻辑只写一份，
回测 = 把这套每日决策在历史上一天天重放，并用【次日真实行情】模拟成交。这样
回测与实盘天然同分布，杜绝"回测一套、实盘另一套"的偏差。

三层职责分离（对应需求里"ML预测 + 布林带/RSI择时 + 精确到每笔的买卖"）：

  1. ML 选池（cross-sectional ranking，项目已验证的唯一稳健 edge）：
     ensemble 融合排序分 → 取当日 top TOPK_RATIO 进入【候选池】，是"买什么"。
  2. 指标择时（布林带 + RSI）：在候选池内决定"何时/以什么价买卖"——
     - 买入：close 接近/跌破布林下轨(超卖) 且 RSI 不过热；
     - 卖出：持仓 close 接近/突破布林上轨(超买)、RSI 过热、或跌出 ML 持有带(衰减/止损)。
  3. 逐笔订单 + 成交模拟：decide_orders 产出精确到 (代码,方向,股数,挂单价) 的订单；
     simulate_fills 用【次日 OHLC + 涨跌停带】判定每笔是否成交、以什么价成交，
     更新现金/持仓，按 A股 佣金+印花税+冲击 计成本。

防泄露：decide_orders 只用 ≤ 决策日(asof) 的特征（收盘后当晚完整）；成交用
asof 的【下一交易日】行情。决策→执行天然 1 日滞后，无未来信息。

关键不变量：现金 + 持仓市值 = 总资产，逐日连续（卖出加现金、买入扣现金、各扣成本），
回测重放每日校验该守恒。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C

LOT = 100   # A股最小交易单位：1 手 = 100 股


# ---------------------------------------------------------------------------
@dataclass
class PortfolioState:
    """组合状态：现金 + 持仓(代码→股数) + 建仓信息(用于持有天数/止损)。

    market_value / total_value 需要传入"当日各股价格"才能估值（持仓本身只记股数）。
    """
    cash: float
    holdings: dict[str, int] = field(default_factory=dict)        # 代码 -> 股数
    cost_basis: dict[str, float] = field(default_factory=dict)    # 代码 -> 建仓均价
    entry_date: dict[str, object] = field(default_factory=dict)   # 代码 -> 建仓日

    def market_value(self, prices: dict[str, float]) -> float:
        """持仓市值 = Σ 股数 × 当日价（缺价的股按 0 计，调用方应保证覆盖）。"""
        mv = 0.0
        for sym, sh in self.holdings.items():
            px = prices.get(sym)
            if px is not None and px == px and px > 0:
                mv += sh * px
        return float(mv)

    def total_value(self, prices: dict[str, float]) -> float:
        return float(self.cash + self.market_value(prices))

    def held_symbols(self) -> set:
        return {s for s, sh in self.holdings.items() if sh > 0}

    def copy(self) -> "PortfolioState":
        return PortfolioState(self.cash, dict(self.holdings),
                              dict(self.cost_basis), dict(self.entry_date))


# ---------------------------------------------------------------------------
def _board_limit_pct(symbol: str, is_st: bool | None = None) -> float:
    """根据代码前缀与 ST 状态返回涨跌停幅度（小数）。"""
    if is_st:
        return 0.05
    s = str(symbol).split(".")[0] if "." in str(symbol) else str(symbol)
    # 北交所：8/4/9 开头（新三板精选层/北交所）
    if s.startswith(("8", "4", "9")):
        return 0.30
    # 创业板 30 开头 / 科创板 68 开头
    if s.startswith(("30", "68")):
        return 0.20
    # 主板 / 默认
    return 0.10


def detect_board_limit(symbol: str, pct_chg: float | None,
                       high: float | None, low: float | None,
                       pre_close: float | None,
                       is_st: bool | None = None) -> tuple[bool, bool, bool]:
    """判断个股是否涨停/跌停/一字板。

    Returns:
        (up_board, down_board, one_word)
    """
    if pct_chg is None and pre_close is None:
        return False, False, False
    limit_pct = _board_limit_pct(symbol, is_st)
    # 用 pct_chg 优先；没有 pct_chg 时用 pre_close 兜底
    if pct_chg is not None and pct_chg == pct_chg:
        up = pct_chg >= (limit_pct * 100.0 - 0.5)   # 留 0.5% 容差
        down = pct_chg <= -(limit_pct * 100.0 - 0.5)
    elif pre_close is not None and pre_close > 0 and high is not None and low is not None:
        up = high >= pre_close * (1.0 + limit_pct) * 0.999
        down = low <= pre_close * (1.0 - limit_pct) * 1.001
    else:
        up = down = False
    one_word = (high is not None and low is not None
                and high == low and high > 0)
    return up, down, one_word


# ---------------------------------------------------------------------------
def parse_initial_holdings(text: str) -> dict[str, int]:
    """解析前端"初始持仓"文本 → {代码: 股数}。

    每行一条，格式 `代码:股数` 或 `代码,股数`（股数自动向下取整到整手）。
    空行/注释(#)忽略。容错：解析失败的行跳过。例::

        000001.SZ:1000
        600519.SH, 200
    """
    out: dict[str, int] = {}
    if not text:
        return out
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = ":" if ":" in line else ("," if "," in line else None)
        if sep is None:
            continue
        code, _, qty = line.partition(sep)
        code = code.strip()
        try:
            sh = int(float(qty.strip()))
        except ValueError:
            continue
        sh = (sh // LOT) * LOT
        if code and sh > 0:
            out[code] = out.get(code, 0) + sh
    return out


# ---------------------------------------------------------------------------
def _col(today: pd.DataFrame, name: str) -> np.ndarray | None:
    return today[name].to_numpy() if name in today.columns else None


def _apply_beta_constraint(pool_idx: list[int], beta: np.ndarray,
                           rank_pct: np.ndarray,
                           max_portfolio_beta: float) -> list[int]:
    """Greedy beta constraint: replace highest-beta stocks with lower-beta candidates.

    Args:
        pool_idx: currently selected stock indices (sorted by rank_pct ascending,
                  i.e. best first).
        beta: beta array aligned with today rows.
        rank_pct: rank percentile array (0 = best).
        max_portfolio_beta: maximum allowed equal-weight portfolio beta.

    Returns:
        Adjusted pool_idx satisfying the constraint (or original if impossible).
    """
    if len(pool_idx) == 0:
        return pool_idx
    pool = list(pool_idx)
    # candidates = all rows with valid beta, sorted by score (rank_pct ascending)
    all_idx = sorted(range(len(beta)), key=lambda i: rank_pct[i])
    all_idx = [i for i in all_idx if np.isfinite(beta[i])]

    def _portfolio_beta(idxs):
        b = beta[np.array(idxs, dtype=int)]
        return float(np.nanmean(b)) if len(b) else float("inf")

    # Iteratively remove the highest-beta stock in the pool and replace it with
    # the best-scoring candidate not currently in the pool that lowers beta.
    for _ in range(len(pool) * 2):
        pb = _portfolio_beta(pool)
        if pb <= max_portfolio_beta:
            break
        # highest-beta stock in current pool
        hi = max(pool, key=lambda i: beta[i] if np.isfinite(beta[i]) else -np.inf)
        # best alternative not in pool and with lower beta
        alternatives = [i for i in all_idx if i not in pool and beta[i] < beta[hi]]
        if not alternatives:
            break
        # choose the alternative with best score (lowest rank_pct) among those
        # that bring the portfolio beta below target, if any; otherwise the lowest-beta
        best = None
        for alt in alternatives:
            new_pool = [i for i in pool if i != hi] + [alt]
            if _portfolio_beta(new_pool) <= max_portfolio_beta:
                best = alt
                break
        if best is None:
            best = min(alternatives, key=lambda i: beta[i])
        pool = [i for i in pool if i != hi] + [best]
    return sorted(pool, key=lambda i: rank_pct[i])


def decide_orders(state: PortfolioState, today: pd.DataFrame, score: np.ndarray,
                  params: dict | None = None,
                  asof_date: object | None = None,
                  price_col: str = "close") -> pd.DataFrame:
    """单日决策：ML 融合排序分 + 布林带/RSI 择时 → 精确逐笔订单（决策日收盘后）。

    Args:
        state    : 当前组合状态（现金 + 持仓），决策【前】的状态。
        today    : 决策日 asof 的横截面（每行一只股），需含 symbol、close/adj_close，
                   择时还需 bb_upper_20/bb_lower_20/rsi_14/limit_flag/pct_chg（缺则跳过对应门）。
        score    : 与 today 行对齐的 ensemble 融合排序分（越大越看好）。
        params   : 覆盖 config 默认（topk/max_positions/bb/rsi/价格模式等）。
        asof_date: 决策日（用于 T+1 锁仓的建仓日标记）。
        price_col: 用于估值/目标金额计算的价列，实盘建议传 "adj_close"。

    Returns:
        订单 DataFrame，每行一笔意图：
          symbol, name, side(buy/sell), shares, limit_price, ref_close,
          reason(触发原因), rank_pct, bb_pos, rsi, blocked, asof_date
        side=buy 的 limit_price 是买入挂单价；sell 同理。shares 为正整手。
        注意：这是【意图】，最终是否成交由 simulate_fills 用次日行情判定。
    """
    p = params or {}
    topk = float(p.get("topk", C.TOPK_RATIO))
    hold_ratio = float(p.get("hold_ratio", getattr(C, "HOLD_RATIO", topk)))
    hold_ratio = max(hold_ratio, topk)
    max_pos = int(p.get("max_positions", C.MAX_POSITIONS))
    max_w = float(p.get("max_weight", C.MAX_WEIGHT_PER_NAME))
    use_bb = bool(p.get("use_bb_timing", C.USE_BB_TIMING))
    bb_buy_k = float(p.get("bb_buy_k", C.BB_BUY_K))
    bb_sell_k = float(p.get("bb_sell_k", C.BB_SELL_K))
    rsi_buy_max = float(p.get("rsi_buy_max", C.RSI_BUY_MAX))
    rsi_sell_min = float(p.get("rsi_sell_min", C.RSI_SELL_MIN))
    price_mode = str(p.get("order_price_mode", C.ORDER_PRICE_MODE))
    min_notional = float(p.get("min_notional", C.DAILY_MIN_NOTIONAL))
    gross = float(p.get("gross_leverage", C.GROSS_LEVERAGE))
    use_beta_constraint = bool(p.get("use_beta_constraint",
                                     getattr(C, "USE_BETA_CONSTRAINT", False)))
    max_portfolio_beta = float(p.get("max_portfolio_beta",
                                      getattr(C, "MAX_PORTFOLIO_BETA", 1.10)))

    sym = today["symbol"].to_numpy()
    name = _col(today, "name")
    if price_col not in today.columns:
        price_col = "close"
    close = today[price_col].to_numpy(dtype=np.float64)
    bb_up = _col(today, "bb_upper_20")
    bb_lo = _col(today, "bb_lower_20")
    rsi = _col(today, "rsi_14")
    limit_flag = _col(today, "limit_flag")
    pct_chg = _col(today, "pct_chg")
    is_st = _col(today, "is_st") if "is_st" in today.columns else None

    n = len(today)
    _EMPTY_COLS = ["symbol", "name", "side", "shares", "limit_price", "ref_close",
                   "reason", "rank_pct", "bb_pos", "rsi", "blocked", "asof_date"]
    if n == 0:                       # 持有日（rebalance_every 闸门）→ 无订单
        return pd.DataFrame(columns=_EMPTY_COLS)
    score = np.asarray(score, dtype=np.float64)
    # 当日百分位排名：rank_pct=0 最看好（top），=1 最差。与 live.py 口径一致。
    order = np.argsort(-score, kind="stable")
    rank_pct = np.empty(n, dtype=np.float64)
    rank_pct[order] = np.arange(n) / max(n - 1, 1)

    in_pool = rank_pct <= topk            # 候选池（买入资格）
    in_hold_band = rank_pct <= hold_ratio  # 持有带（跌出才因衰减卖出）
    held = state.held_symbols()

    # 估值用决策日 price_col 估当前总资产（决定单股目标金额）
    prices = {sym[i]: close[i] for i in range(n) if close[i] == close[i]}
    total_val = state.total_value(prices)
    # 每股目标金额：在持仓数上限与单股权重上限间取较紧者
    cap_by_count = total_val * gross / max(max_pos, 1) if max_pos > 0 else total_val * max_w
    target_notional = min(cap_by_count, total_val * max_w)

    def _bb_pos(i):
        if bb_up is None or bb_lo is None:
            return np.nan
        rng = bb_up[i] - bb_lo[i]
        return (close[i] - bb_lo[i]) / rng if rng and rng == rng and rng > 1e-9 else np.nan

    def _limit_price(i, side):
        """挂单价：next_open 用 ref close 占位（成交时以次日开盘价撮合）；
        limit_bb 买挂下轨、卖挂上轨（成交需次日价区触及）。"""
        if price_mode == "limit_bb" and bb_up is not None and bb_lo is not None:
            if side == "buy" and bb_lo[i] == bb_lo[i]:
                return float(bb_lo[i])
            if side == "sell" and bb_up[i] == bb_up[i]:
                return float(bb_up[i])
        return float(close[i])   # next_open 模式：占位，撮合时换次日开盘

    # 辅助：用代码前缀识别的板幅判定替代简单的 limit_flag+pct_chg
    def _limit_flags(i):
        """返回 (up_limit, down_limit) 布尔对。"""
        if limit_flag is None:
            return False, False
        # 优先复用面板 limit_flag，但用它触发时再做板幅校验
        if not bool(limit_flag[i]):
            return False, False
        pc = pct_chg[i] if pct_chg is not None else None
        up, down, _ = detect_board_limit(sym[i], pc, None, None, None,
                                         is_st=bool(is_st[i]) if is_st is not None else None)
        # limit_flag 只告诉"触及板"，方向由 pct_chg 符号决定
        if pc is not None and pc > 0:
            return up, False
        if pc is not None and pc < 0:
            return False, down
        # pct_chg 缺失时保守处理：按板幅阈值重新算
        return up, down

    # ============================================================== 满仓再平衡
    if not use_bb:
        # 目标池：score 最高的 min(max_pos, |in_pool|) 只
        pool_idx = [i for i in range(n) if in_pool[i]
                    and close[i] == close[i] and close[i] > 0]
        pool_idx.sort(key=lambda i: rank_pct[i])
        if max_pos > 0:
            pool_idx = pool_idx[:max_pos]

        # Beta 约束：等权组合 beta 过高时，用次优的低 beta 候选替换高 beta 股。
        beta = _col(today, "beta")
        if use_beta_constraint and beta is not None and len(pool_idx) > 0:
            pool_idx = _apply_beta_constraint(
                pool_idx, beta, rank_pct, max_portfolio_beta)

        n_target = len(pool_idx)
        if n_target == 0:
            return pd.DataFrame(columns=[
                "symbol", "name", "side", "shares", "limit_price", "ref_close",
                "reason", "rank_pct", "bb_pos", "rsi", "blocked", "asof_date"])
        # 等权目标金额（受单股权重上限约束），满仓部署
        per_name = min(total_val * gross / n_target, total_val * max_w)
        target_sh = {}
        for i in pool_idx:
            target_sh[sym[i]] = int(np.floor(per_name / close[i] / LOT) * LOT)
        idx_by_sym = {sym[i]: i for i in range(n)}
        rows = []
        # 1) 卖：当前持有但已不在目标池 → 清仓；仍在池但超配 → 减仓
        for s_held in list(held):
            i = idx_by_sym.get(s_held)
            cur = state.holdings.get(s_held, 0)
            tgt = target_sh.get(s_held, 0)
            if cur <= tgt:
                continue
            delta = cur - tgt
            if i is None:                       # 该名今日无行情 → 无法定价，跳过
                continue
            px = close[i]
            # 碎单过滤只作用于"减仓微调"（tgt>0 的小幅超配）；清仓(tgt==0)永远执行。
            if tgt > 0 and delta * px < min_notional:
                continue                        # 小幅超配不动（抑制 churn）
            reason = "再平衡:清仓出池" if tgt == 0 else "再平衡:减仓"
            _, dn = _limit_flags(i)
            rows.append(_row(s_held, name, i, "sell", delta, _limit_price(i, "sell"),
                             px, ("跌停拦截:" + reason) if dn else reason,
                             rank_pct[i], _bb_pos(i),
                             rsi[i] if rsi is not None else np.nan,
                             blocked=dn, asof_date=asof_date))
        # 2) 买：目标池内新建 + 低配补足
        for i in pool_idx:
            s2 = sym[i]
            cur = state.holdings.get(s2, 0)
            tgt = target_sh[s2]
            if tgt <= cur:
                continue
            delta = tgt - cur
            px = close[i]
            # 碎单过滤只挡"加仓微调"（cur>0 的小幅补足）；新建仓(cur==0)永远执行，
            # 否则在小 per_name / 资金缩水时会陷入"建不起仓→满现金"的死循环。
            if cur > 0 and delta * px < min_notional:
                continue
            up, _ = _limit_flags(i)
            reason = "再平衡:建仓" if cur == 0 else "再平衡:加仓"
            rows.append(_row(s2, name, i, "buy", delta, _limit_price(i, "buy"),
                             px, ("涨停拦截:" + reason) if up else reason,
                             rank_pct[i], _bb_pos(i),
                             rsi[i] if rsi is not None else np.nan,
                             blocked=up, asof_date=asof_date))
        orders = pd.DataFrame(rows, columns=[
            "symbol", "name", "side", "shares", "limit_price", "ref_close",
            "reason", "rank_pct", "bb_pos", "rsi", "blocked", "asof_date"])
        if len(orders):
            orders["_k"] = orders["side"].map({"sell": 0, "buy": 1})
            orders = orders.sort_values(["_k", "rank_pct"]).drop(columns="_k") \
                           .reset_index(drop=True)
        return orders

    rows = []
    for i in range(n):
        s = sym[i]
        cur = state.holdings.get(s, 0)
        px = close[i]
        if px != px or px <= 0:
            continue
        bbp = _bb_pos(i)
        r = rsi[i] if rsi is not None else np.nan
        up_limit, down_limit = _limit_flags(i)

        # ---- 持仓票：判断是否卖出 ----
        if cur > 0:
            reason = None
            if not in_hold_band[i]:
                reason = "跌出ML持有带"
            elif use_bb and rsi_sell_min and r == r and r >= rsi_sell_min:
                reason = f"RSI超买({r:.0f})"
            elif use_bb and bb_up is not None and bb_up[i] == bb_up[i] \
                    and px >= bb_up[i] * (1.0 - bb_sell_k):
                reason = "触布林上轨"
            if reason:
                if down_limit:
                    rows.append(_row(s, name, i, "sell", cur, _limit_price(i, "sell"),
                                     px, "跌停拦截:" + reason, rank_pct[i], bbp, r,
                                     blocked=True, asof_date=asof_date))
                else:
                    rows.append(_row(s, name, i, "sell", cur, _limit_price(i, "sell"),
                                     px, reason, rank_pct[i], bbp, r,
                                     blocked=False, asof_date=asof_date))
            # 否则继续持有，不产生订单
            continue

        # ---- 非持仓票：判断是否买入 ----
        if not in_pool[i]:
            continue
        if len(held) + sum(1 for x in rows if x["side"] == "buy") >= max_pos and max_pos > 0:
            continue  # 已达持仓数上限
        # 指标择时门控
        if use_bb:
            if rsi_buy_max and r == r and r > rsi_buy_max:
                continue   # RSI 过热不追
            if bb_lo is not None and bb_lo[i] == bb_lo[i]:
                if px > bb_lo[i] * (1.0 + bb_buy_k):
                    continue   # 未回踩到下轨附近，不买（等更好价位）
        # 目标股数：目标金额 / 价，向下取整手
        shares = int(np.floor(target_notional / px / LOT) * LOT)
        if shares <= 0 or shares * px < min_notional:
            continue
        reason = "布林下轨回踩" if use_bb else "进入ML候选池"
        if up_limit:
            rows.append(_row(s, name, i, "buy", shares, _limit_price(i, "buy"),
                             px, "涨停拦截:" + reason, rank_pct[i], bbp, r,
                             blocked=True, asof_date=asof_date))
        else:
            rows.append(_row(s, name, i, "buy", shares, _limit_price(i, "buy"),
                             px, reason, rank_pct[i], bbp, r,
                             blocked=False, asof_date=asof_date))

    cols = ["symbol", "name", "side", "shares", "limit_price", "ref_close",
            "reason", "rank_pct", "bb_pos", "rsi", "blocked", "asof_date"]
    orders = pd.DataFrame(rows, columns=cols)
    if len(orders):
        # 卖在前(腾现金)、买在后；组内按排名
        orders["_k"] = orders["side"].map({"sell": 0, "buy": 1})
        orders = orders.sort_values(["_k", "rank_pct"]).drop(columns="_k") \
                       .reset_index(drop=True)
    return orders


def _row(sym, name_arr, i, side, shares, limit_price, ref_close, reason,
         rank_pct, bb_pos, rsi, blocked=False, asof_date=None):
    return {
        "symbol": sym,
        "name": (name_arr[i] if name_arr is not None else sym),
        "side": side, "shares": int(shares),
        "limit_price": round(float(limit_price), 3),
        "ref_close": round(float(ref_close), 3),
        "reason": reason,
        "rank_pct": round(float(rank_pct), 4),
        "bb_pos": (round(float(bb_pos), 3) if bb_pos == bb_pos else None),
        "rsi": (round(float(rsi), 1) if rsi == rsi else None),
        "blocked": bool(blocked),
        "asof_date": str(pd.Timestamp(asof_date).date()) if asof_date is not None else None,
    }


# ---------------------------------------------------------------------------
def simulate_fills(state: PortfolioState, orders: pd.DataFrame,
                   next_bars: pd.DataFrame, params: dict | None = None,
                   t1_lock: bool = True) -> dict:
    """用【次日行情】模拟订单成交，更新 state（原地修改），返回成交明细 + 现金流。

    成交判定（A股规则）：
      · next_bars 缺该股次日数据（停牌） → 不成交。
      · 涨停/跌停：按代码前缀区分 10%/20%/30%/5%(ST)；一字板（high==low）时反向单无法成交。
      · next_open 模式：以次日 OPEN 成交（市价单），必成交（除非停牌/一字板反向/T+1锁）。
      · limit_bb 模式：买单需次日 low ≤ 买入挂单价（价区触及），成交价=min(挂单价, open)；
        卖单需次日 high ≥ 卖出挂单价，成交价=max(挂单价, open)。未触及→不成交。
      · T+1 锁：当日买入/加仓的票，次日决策日不能卖出（实盘 T+1 交割）。
      · 买入还需现金充足（含成本），否则按可用现金缩减到整手；不足 1 手则放弃。

    成本：佣金双边 COMMISSION_RATE；印花税卖出单边 STAMP_DUTY_RATE；
    可选平方根冲击成本（按成交额/ADV）。

    Returns:
        {"fills": DataFrame(逐笔成交), "n_fill", "n_reject", "buy_notional",
         "sell_notional", "commission", "stamp", "impact", "total_cost"}
    """
    p = params or {}
    price_mode = str(p.get("order_price_mode", C.ORDER_PRICE_MODE))
    commission = float(p.get("commission", C.COMMISSION_RATE))
    stamp = float(p.get("stamp_duty", C.STAMP_DUTY_RATE))
    use_impact = bool(p.get("use_impact", C.USE_IMPACT_COST))
    impact_k = float(p.get("impact_k", C.IMPACT_K))

    nb = next_bars.set_index("symbol") if "symbol" in next_bars.columns else next_bars
    has_open = "open" in nb.columns
    has_hl = "high" in nb.columns and "low" in nb.columns
    has_amt = "amount" in nb.columns

    fills = []
    tot_comm = tot_stamp = tot_impact = 0.0
    buy_notional = sell_notional = 0.0

    # 交易日序列：用于 T+1 锁仓的"至少隔 1 个交易日"判断
    trade_dates = None
    if t1_lock:
        # 从订单 asof_date 和面板推断；若无法推断，退化为同自然日锁
        pass

    for o in orders.itertuples(index=False):
        s = o.symbol
        side = o.side
        want = int(o.shares)
        asof = getattr(o, "asof_date", None)
        if want <= 0 or s not in nb.index:
            fills.append(_fill(o, 0, np.nan, "停牌/无次日行情" if s not in nb.index else "零股数",
                               filled=False))
            continue
        bar = nb.loc[s]
        nopen = float(bar["open"]) if has_open and bar["open"] == bar["open"] else np.nan
        nhigh = float(bar["high"]) if has_hl and bar["high"] == bar["high"] else nopen
        nlow = float(bar["low"]) if has_hl and bar["low"] == bar["low"] else nopen
        if nopen != nopen or nopen <= 0:
            fills.append(_fill(o, 0, np.nan, "次日无有效开盘价", filled=False))
            continue

        npct = (float(bar["pct_chg"]) if ("pct_chg" in bar.index
                and bar["pct_chg"] == bar["pct_chg"]) else None)
        pre_close = (float(bar["pre_close"]) if ("pre_close" in bar.index
                    and bar["pre_close"] == bar["pre_close"]) else None)
        is_st = (bool(bar["is_st"]) if "is_st" in bar.index
                 and bar["is_st"] == bar["is_st"] else None)

        up_board, down_board, one_word = detect_board_limit(
            s, npct, nhigh, nlow, pre_close, is_st=is_st)

        # T+1 锁仓检查：卖出时若建仓日就是 asof_date（同一决策日买入），实盘不能卖
        if t1_lock and side == "sell" and asof is not None:
            entry = state.entry_date.get(s)
            if entry is not None:
                entry_str = str(pd.Timestamp(entry).date())
                asof_str = str(pd.Timestamp(asof).date())
                if entry_str == asof_str:
                    fills.append(_fill(o, 0, np.nan, "T+1锁仓", filled=False))
                    continue

        # 成交价确定
        fill_px = np.nan
        reject = None
        if side == "buy":
            if one_word and up_board:
                reject = "一字涨停无法买入"
            elif price_mode == "limit_bb":
                lp = float(o.limit_price)
                if nlow <= lp:                      # 价区触及买入挂单价
                    fill_px = min(lp, nopen)        # 跳空低开则以开盘价更优成交
                else:
                    reject = "未触及买入挂单价"
            else:                                    # next_open 市价
                fill_px = nopen
        else:  # sell
            if one_word and down_board:
                reject = "一字跌停无法卖出"
            elif price_mode == "limit_bb":
                lp = float(o.limit_price)
                if nhigh >= lp:
                    fill_px = max(lp, nopen)
                else:
                    reject = "未触及卖出挂单价"
            else:
                fill_px = nopen

        if reject or fill_px != fill_px or fill_px <= 0:
            fills.append(_fill(o, 0, np.nan, reject or "无效成交价", filled=False))
            continue

        # ---- 买入：现金约束（含成本），不足则缩到可买整手 ----
        if side == "buy":
            adv = float(bar["amount"]) * 1000.0 if has_amt and bar["amount"] == bar["amount"] else np.inf
            shares = want
            # 估算单位股的总成本率（佣金 + 冲击），迭代一次确定可买量
            per_share_cost = fill_px * (1.0 + commission)
            affordable = int(np.floor(state.cash / per_share_cost / LOT) * LOT)
            shares = min(shares, max(affordable, 0))
            if shares <= 0:
                fills.append(_fill(o, 0, fill_px, "现金不足", filled=False))
                continue
            notional = shares * fill_px
            comm = notional * commission
            impact = (impact_k * np.sqrt(notional / adv) * notional
                      if use_impact and np.isfinite(adv) and adv > 0 else 0.0)
            total = notional + comm + impact
            if total > state.cash + 1e-6:           # 缩一手再试（冲击使其略超）
                shares -= LOT
                if shares <= 0:
                    fills.append(_fill(o, 0, fill_px, "现金不足", filled=False))
                    continue
                notional = shares * fill_px
                comm = notional * commission
                impact = (impact_k * np.sqrt(notional / adv) * notional
                          if use_impact and np.isfinite(adv) and adv > 0 else 0.0)
                total = notional + comm + impact
            state.cash -= total
            prev = state.holdings.get(s, 0)
            new_sh = prev + shares
            # 更新建仓均价
            prev_cost = state.cost_basis.get(s, 0.0) * prev
            state.cost_basis[s] = (prev_cost + notional) / new_sh if new_sh else fill_px
            state.holdings[s] = new_sh
            # T+1：记录本次决策日为建仓日（加仓也更新为最新决策日）
            state.entry_date[s] = asof
            tot_comm += comm; tot_impact += impact; buy_notional += notional
            fills.append(_fill(o, shares, fill_px, "成交", filled=True,
                               cost=comm + impact))
        # ---- 卖出 ----
        else:
            prev = state.holdings.get(s, 0)
            shares = min(want, prev)
            if shares <= 0:
                fills.append(_fill(o, 0, fill_px, "无持仓可卖", filled=False))
                continue
            adv = float(bar["amount"]) * 1000.0 if has_amt and bar["amount"] == bar["amount"] else np.inf
            notional = shares * fill_px
            comm = notional * commission
            st = notional * stamp
            impact = (impact_k * np.sqrt(notional / adv) * notional
                      if use_impact and np.isfinite(adv) and adv > 0 else 0.0)
            state.cash += notional - comm - st - impact
            new_sh = prev - shares
            if new_sh <= 0:
                state.holdings.pop(s, None)
                state.cost_basis.pop(s, None)
                state.entry_date.pop(s, None)
            else:
                state.holdings[s] = new_sh
            tot_comm += comm; tot_stamp += st; tot_impact += impact
            sell_notional += notional
            fills.append(_fill(o, shares, fill_px, "成交", filled=True,
                               cost=comm + st + impact))

    fdf = pd.DataFrame(fills)
    n_fill = int(fdf["filled"].sum()) if len(fdf) else 0
    return {
        "fills": fdf,
        "n_fill": n_fill,
        "n_reject": (len(fdf) - n_fill) if len(fdf) else 0,
        "buy_notional": float(buy_notional),
        "sell_notional": float(sell_notional),
        "commission": float(tot_comm),
        "stamp": float(tot_stamp),
        "impact": float(tot_impact),
        "total_cost": float(tot_comm + tot_stamp + tot_impact),
    }


def estimate_order_cost(orders: pd.DataFrame, params: dict | None = None) -> dict:
    """估算一批订单（decide_orders 产出）的交易成本（佣金双边 + 印花税卖出单边）。

    仅用于实盘订单展示的事前估算（按挂单价 × 股数 算名义额）。真实成交成本由
    simulate_fills 用次日价计；这里给用户一个下单前的成本预估。被拦截(blocked)
    的订单不计入。"""
    p = params or {}
    commission = float(p.get("commission", C.COMMISSION_RATE))
    stamp = float(p.get("stamp_duty", C.STAMP_DUTY_RATE))
    if orders is None or len(orders) == 0:
        return {"buy_notional": 0.0, "sell_notional": 0.0, "commission": 0.0,
                "stamp_duty": 0.0, "total": 0.0}
    o = orders[~orders["blocked"]] if "blocked" in orders.columns else orders
    notion = o["shares"].to_numpy(dtype=np.float64) * o["limit_price"].to_numpy(dtype=np.float64)
    is_buy = (o["side"] == "buy").to_numpy()
    buy = float(notion[is_buy].sum())
    sell = float(notion[~is_buy].sum())
    comm = (buy + sell) * commission
    st = sell * stamp
    return {"buy_notional": buy, "sell_notional": sell, "commission": float(comm),
            "stamp_duty": float(st), "total": float(comm + st)}


def _fill(o, shares, price, status, filled, cost=0.0):
    return {
        "symbol": o.symbol, "name": getattr(o, "name", o.symbol),
        "side": o.side, "want_shares": int(o.shares),
        "fill_shares": int(shares),
        "fill_price": (round(float(price), 3) if price == price else None),
        "notional": (round(float(shares * price), 2) if price == price else 0.0),
        "cost": round(float(cost), 2),
        "status": status, "filled": bool(filled),
        "reason": getattr(o, "reason", ""),
    }
