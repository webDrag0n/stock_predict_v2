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
def trade_costs(notional, side: str, adv=np.inf, params: dict | None = None):
    """单笔订单的 A股交易成本（唯一实现，回测两个引擎与实盘估算共用）。

    佣金 = max(notional × COMMISSION_RATE, MIN_COMMISSION)   —— 买卖双边
    印花税 = notional × STAMP_DUTY_RATE                       —— 仅卖出
    冲击 = notional × IMPACT_K × sqrt(notional / ADV)         —— 可选

    Args:
        notional : 成交额（元），标量或 numpy 数组（向量化时逐元素计算）。
        side     : "buy" | "sell"。
        adv      : 该股次日成交额（元），标量或与 notional 同形数组；
                   inf / NaN / <=0 → 冲击为 0。
        params   : 覆盖 config（commission/min_commission/stamp_duty/use_impact/impact_k）。

    Returns:
        (commission, stamp, impact)，与 notional 同形。
    """
    p = params or {}
    rate = float(p.get("commission", C.COMMISSION_RATE))
    min_comm = float(p.get("min_commission", getattr(C, "MIN_COMMISSION", 0.0)))
    stamp_rate = float(p.get("stamp_duty", C.STAMP_DUTY_RATE))
    use_impact = bool(p.get("use_impact", C.USE_IMPACT_COST))
    impact_k = float(p.get("impact_k", C.IMPACT_K))

    notional = np.asarray(notional, dtype=np.float64)
    adv = np.asarray(adv, dtype=np.float64)
    comm = notional * rate
    if min_comm > 0:
        comm = np.where(notional > 0, np.maximum(comm, min_comm), 0.0)
    stamp = notional * stamp_rate if side == "sell" else np.zeros_like(notional)
    ok_adv = np.isfinite(adv) & (adv > 0)
    ratio = np.divide(notional, adv, out=np.zeros_like(notional), where=ok_adv)
    impact = np.where(ok_adv & (notional > 0) & use_impact,
                      impact_k * np.sqrt(ratio) * notional, 0.0)
    return comm, stamp, impact


# ---------------------------------------------------------------------------
@dataclass
class PortfolioState:
    """组合状态：现金 + 持仓(代码→股数) + 建仓信息(用于持有天数/止损)。

    market_value / total_value 需要传入"当日各股价格"才能估值（持仓本身只记股数）。
    last_price 记录每股最近一次可得的原始价：当日缺价（停牌/跌出横截面）时用它
    兜底估值，不再按 0 计（消除停牌日净值假跌与风控刹车误触）。
    val_factor 是每股累计除权估值因子（由回测侧按 pre_close 链条维护，见
    backtest_daily._price_map），估值价 = 原始价 × val_factor，消除除权日假性盈亏。
    注意：last_price 假定原始价口径（price_col="close"），不要喂 adj_close。
    t1_locked 记录 T+1 锁定明细：代码 -> (锁定决策日, 当日买入的股数)。锁仓按
    【当日新增股数】粒度——老仓位不受新买入影响（真实 T+1 只锁当日买入部分）。
    """
    cash: float
    holdings: dict[str, int] = field(default_factory=dict)        # 代码 -> 股数
    cost_basis: dict[str, float] = field(default_factory=dict)    # 代码 -> 建仓均价
    entry_date: dict[str, object] = field(default_factory=dict)   # 代码 -> 建仓日
    last_price: dict[str, float] = field(default_factory=dict)    # 代码 -> 最近可得原始价
    val_factor: dict[str, float] = field(default_factory=dict)    # 代码 -> 累计除权估值因子
    t1_locked: dict[str, tuple] = field(default_factory=dict)     # 代码 -> (锁定决策日, 锁定股数)

    def market_value(self, prices: dict[str, float]) -> float:
        """持仓市值 = Σ 股数 × 估值价（原始价 × val_factor）。

        当日有价的股顺手刷新 last_price；缺价的股（停牌等）用 last_price 兜底，
        不再按 0 计。完全无历史价的持仓才按 0 计（刚建仓尚未估值的边缘情形）。
        """
        mv = 0.0
        for sym, sh in self.holdings.items():
            f = self.val_factor.get(sym, 1.0)
            px = prices.get(sym)
            if px is not None and px == px and px > 0:
                self.last_price[sym] = float(px)
                mv += sh * px * f
                continue
            lp = self.last_price.get(sym)
            if lp is not None and lp > 0:
                mv += sh * lp * f
        return float(mv)

    def total_value(self, prices: dict[str, float]) -> float:
        return float(self.cash + self.market_value(prices))

    def held_symbols(self) -> set:
        return {s for s, sh in self.holdings.items() if sh > 0}

    def copy(self) -> "PortfolioState":
        return PortfolioState(self.cash, dict(self.holdings),
                              dict(self.cost_basis), dict(self.entry_date),
                              dict(self.last_price), dict(self.val_factor),
                              dict(self.t1_locked))


# ---------------------------------------------------------------------------
def _board_limit_pct(symbol: str, is_st: bool | None = None) -> float:
    """根据代码前缀与 ST 状态返回涨跌停幅度（小数）。"""
    if is_st:
        return 0.05
    s = str(symbol).split(".")[0] if "." in str(symbol) else str(symbol)
    # 北交所：4/8 开头及 920 新号段（这里保留 "9" 前缀即按北交所 30% 处理）。
    # 注意沪市 B 股 900xxx / 深市 B 股 200xxx 是 10% 主板幅度，会被 "9" 误伤——
    # 因此 B 股已在 tushare.py 的 universe 构建处整体剔除，正常不会走到这里。
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
    # 一字板：high≈low。精确浮点相等会被分笔噪声/舍入漏判，改用
    # 容差 abs(h-l) <= 0.1% × pre_close（pre_close 缺失时退化为以 high 为尺度）。
    one_word = False
    if high is not None and low is not None and high > 0:
        scale = pre_close if (pre_close is not None and pre_close > 0) else high
        one_word = abs(high - low) <= 0.001 * scale
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
            if i is None:
                # 跌出当日决策横截面（停牌/被剔出样本）→ 不再卡死持有。
                # 有历史价则按 last_price 发市价清退单：next_open 模式下挂单价
                # 只是占位，成交与否/成交价由 simulate_fills 用次日行情决定
                # （次日仍停牌则自然不成交，下一天再发）；完全无历史价才继续持有。
                lp = state.last_price.get(s_held)
                if lp is None or lp <= 0:
                    continue
                rows.append(_row(s_held, None, None, "sell", delta, lp, lp,
                                 "跌出横截面:清退", np.nan, np.nan, np.nan,
                                 blocked=False, asof_date=asof_date))
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

    # 跌出当日横截面的持仓：上面的循环只遍历今日面板里的票，查不到持仓股——
    # 与满仓再平衡分支同样处理：有历史价则按 last_price 发市价清退单，
    # 由 simulate_fills 用次日行情决定成交；无历史价则继续持有等待。
    sym_set = set(sym.tolist())
    for s_held in held:
        if s_held in sym_set:
            continue
        lp = state.last_price.get(s_held)
        if lp is None or lp <= 0:
            continue
        rows.append(_row(s_held, None, None, "sell", state.holdings[s_held], lp, lp,
                         "跌出横截面:清退", np.nan, np.nan, np.nan,
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
      · T+1 锁：当日决策买入的【股数】不能在同决策日的卖出单中卖出（老仓可卖）。
      · 买入还需现金充足（含成本），否则按可用现金缩减到整手；不足 1 手则放弃。

    成本：佣金双边 COMMISSION_RATE（每笔最低 MIN_COMMISSION）；印花税卖出单边
    STAMP_DUTY_RATE；可选平方根冲击成本（按成交额/ADV）。三者由 trade_costs
    统一计算。可选参与率上限：单笔成交 ≤ MAX_PARTICIPATION × 次日成交额
    （冲击模型只计价不限量，没有上限时大单会不切实际地全额成交）。

    Returns:
        {"fills": DataFrame(逐笔成交), "n_fill", "n_reject", "buy_notional",
         "sell_notional", "commission", "stamp", "impact", "total_cost"}
    """
    p = params or {}
    price_mode = str(p.get("order_price_mode", C.ORDER_PRICE_MODE))
    max_participation = float(p.get("max_participation",
                                    getattr(C, "MAX_PARTICIPATION", 0.0)))

    nb = next_bars.set_index("symbol") if "symbol" in next_bars.columns else next_bars
    has_open = "open" in nb.columns
    has_hl = "high" in nb.columns and "low" in nb.columns
    has_amt = "amount" in nb.columns

    fills = []
    tot_comm = tot_stamp = tot_impact = 0.0
    buy_notional = sell_notional = 0.0

    # T+1 锁仓的实际判定在下方卖出分支：t1_locked 里锁定日 == 订单 asof_date 的
    # 股数不能卖（每日决策节奏下同一决策日买入的部分次日才能卖，无需额外交易日序列）。

    for o in orders.itertuples(index=False):
        s = o.symbol
        side = o.side
        want = int(o.shares)
        asof = getattr(o, "asof_date", None)
        # 决策层拦截（涨跌停 blocked=True）的订单记 0 成交直接跳过——与实盘
        # --commit 跳过 blocked 单的语义一致（此前回测会照常撮合，回测≠实盘）。
        if bool(getattr(o, "blocked", False)):
            fills.append(_fill(o, 0, np.nan, "决策层拦截", filled=False))
            continue
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

        # T+1 锁仓（按当日新增股数粒度）：当日决策买入的部分，同决策日的卖出单
        # 不能动；早于当日买入的老仓位可正常卖。真实 T+1 只锁当日买入的股数。
        t1_sellable_cap: int | None = None
        if t1_lock and side == "sell" and asof is not None:
            lock = state.t1_locked.get(s)
            if lock is not None:
                lock_date, lock_shares = lock
                if str(pd.Timestamp(lock_date).date()) == str(pd.Timestamp(asof).date()):
                    t1_sellable_cap = max(state.holdings.get(s, 0) - int(lock_shares), 0)

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

        # ---- 买入：参与率上限 → 现金约束（含成本），不足则缩到可买整手 ----
        if side == "buy":
            adv = float(bar["amount"]) * 1000.0 if has_amt and bar["amount"] == bar["amount"] else np.inf
            shares = want
            capped = False
            # 参与率上限：单笔不超过次日成交额的一定比例（冲击模型只计价不限量）
            if max_participation > 0 and np.isfinite(adv) and adv > 0:
                cap_shares = int(np.floor(max_participation * adv / fill_px / LOT) * LOT)
                if cap_shares < shares:
                    shares = max(cap_shares, 0)
                    capped = True
            # 估算单位股的总成本率（佣金），迭代确定可买量
            comm_rate = float(p.get("commission", C.COMMISSION_RATE))
            per_share_cost = fill_px * (1.0 + comm_rate)
            affordable = int(np.floor(state.cash / per_share_cost / LOT) * LOT)
            shares = min(shares, max(affordable, 0))
            if shares <= 0:
                fills.append(_fill(o, 0, fill_px,
                                   "流动性不足(参与率上限)" if capped else "现金不足",
                                   filled=False))
                continue
            notional = shares * fill_px
            comm, _, impact = trade_costs(notional, "buy", adv, p)
            total = notional + comm + impact
            # 最低佣金/冲击成本可能使总价略超现金：循环缩手直到买得起或股数归零
            # （修复前只缩一手，冲击大时仍会扣出负现金）。
            while shares > 0 and total > state.cash + 1e-6:
                shares -= LOT
                notional = shares * fill_px
                comm, _, impact = trade_costs(notional, "buy", adv, p)
                total = notional + comm + impact
            if shares <= 0:
                fills.append(_fill(o, 0, fill_px, "现金不足", filled=False))
                continue
            state.cash -= total
            prev = state.holdings.get(s, 0)
            new_sh = prev + shares
            # 更新建仓均价
            prev_cost = state.cost_basis.get(s, 0.0) * prev
            state.cost_basis[s] = (prev_cost + notional) / new_sh if new_sh else fill_px
            state.holdings[s] = new_sh
            # T+1：记录本次决策日为建仓日；锁定明细按当日新增股数累计（加仓不锁老仓）
            state.entry_date[s] = asof
            asof_str = str(pd.Timestamp(asof).date()) if asof is not None else None
            if asof_str is not None:
                lock = state.t1_locked.get(s)
                if lock is not None and str(pd.Timestamp(lock[0]).date()) == asof_str:
                    state.t1_locked[s] = (asof_str, int(lock[1]) + shares)
                else:
                    state.t1_locked[s] = (asof_str, shares)
            tot_comm += comm; tot_impact += impact; buy_notional += notional
            fills.append(_fill(o, shares, fill_px,
                               "成交(参与率限量)" if capped else "成交",
                               filled=True, cost=comm + impact))
        # ---- 卖出 ----
        else:
            prev = state.holdings.get(s, 0)
            shares = min(want, prev)
            # T+1 粒度锁：只能卖"非当日买入"的部分
            if t1_sellable_cap is not None:
                shares = min(shares, t1_sellable_cap)
            if shares <= 0:
                status = ("T+1锁仓" if (t1_sellable_cap is not None and prev > 0)
                          else "无持仓可卖")
                fills.append(_fill(o, 0, fill_px, status, filled=False))
                continue
            adv = float(bar["amount"]) * 1000.0 if has_amt and bar["amount"] == bar["amount"] else np.inf
            capped = False
            if max_participation > 0 and np.isfinite(adv) and adv > 0:
                cap_shares = int(np.floor(max_participation * adv / fill_px / LOT) * LOT)
                if cap_shares < shares:
                    shares = max(cap_shares, 0)
                    capped = True
            if shares <= 0:
                fills.append(_fill(o, 0, fill_px, "流动性不足(参与率上限)", filled=False))
                continue
            notional = shares * fill_px
            comm, st, impact = trade_costs(notional, "sell", adv, p)
            state.cash += notional - comm - st - impact
            new_sh = prev - shares
            if new_sh <= 0:
                state.holdings.pop(s, None)
                state.cost_basis.pop(s, None)
                state.entry_date.pop(s, None)
                # 清仓后估值缓存一并清掉，防止日后再次建仓时沿用陈旧的
                # last_price / 除权因子造成估值口径漂移
                state.last_price.pop(s, None)
                state.val_factor.pop(s, None)
                state.t1_locked.pop(s, None)
            else:
                state.holdings[s] = new_sh
            tot_comm += comm; tot_stamp += st; tot_impact += impact
            sell_notional += notional
            fills.append(_fill(o, shares, fill_px,
                               "成交(参与率限量)" if capped else "成交",
                               filled=True, cost=comm + st + impact))

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

    仅用于实盘订单展示的事前估算（按挂单价 × 股数 算名义额），成本公式与
    撮合侧完全一致（trade_costs，含每笔最低佣金；不含冲击——事前无法知道
    次日 ADV 口径的成交价）。真实成交成本由 simulate_fills 用次日价计。
    被拦截(blocked)的订单不计入。"""
    if orders is None or len(orders) == 0:
        return {"buy_notional": 0.0, "sell_notional": 0.0, "commission": 0.0,
                "stamp_duty": 0.0, "total": 0.0}
    o = orders[~orders["blocked"]] if "blocked" in orders.columns else orders
    notion = o["shares"].to_numpy(dtype=np.float64) * o["limit_price"].to_numpy(dtype=np.float64)
    is_buy = (o["side"] == "buy").to_numpy()
    buy = float(notion[is_buy].sum())
    sell = float(notion[~is_buy].sum())
    comm_buy, _, _ = trade_costs(notion[is_buy], "buy", params=params)
    comm_sell, st_sell, _ = trade_costs(notion[~is_buy], "sell", params=params)
    comm = float(comm_buy.sum() + comm_sell.sum())
    st = float(st_sell.sum())
    return {"buy_notional": buy, "sell_notional": sell, "commission": comm,
            "stamp_duty": st, "total": comm + st}


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
