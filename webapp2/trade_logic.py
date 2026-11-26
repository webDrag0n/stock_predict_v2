"""交易页的纯逻辑：订单统计、执行进度、流程四步状态、风控五层拆解、账户快照、目标 vs 当前。

不依赖 dash，便于单测。口径说明：
  · 风控乘数复用 src/risk.py::compute_risk_scale（回测=实盘唯一实现），这里只做只读拆层。
  · 订单是「当晚收盘后生成、次日开盘执行」的：订单日期早于行情缓存最新日 ⇒ 订单已过期。
    （仅读文件名与已落盘数据，不涉及任何未来信息。）
  · 执行勾选的键与 execution.load_rejected_rids / predict_tomorrow --commit 完全一致：
    "<side>-<symbol>" → "done" | "rejected"（旧版曾写 true 表示已执行，读取时兼容）。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

import config as C
from risk import compute_risk_scale


# ----------------------------------------------------------------------------- 基础
def truthy(v) -> bool:
    """csv 读回的 blocked 可能是 True/False 字符串，统一成 bool。"""
    return str(v).strip().lower() in ("true", "1")


def blocked_mask(orders: pd.DataFrame) -> pd.Series:
    if "blocked" not in orders.columns:
        return pd.Series(False, index=orders.index)
    return orders["blocked"].map(truthy)


def as_float(v) -> float | None:
    """松散数值解析：None/NaN/非法 → None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def order_price(limit, ref) -> float | None:
    """挂单价优先，缺失回落参考收盘；都无效返回 None。"""
    for v in (limit, ref):
        f = as_float(v)
        if f is not None and f > 0:
            return f
    return None


def order_key(side, symbol) -> str:
    return f"{side}-{symbol}"


def order_amounts(orders: pd.DataFrame) -> dict:
    """买/卖/拦截笔数与预计成交金额（剔除拦截单；价 = 挂单价，缺失回落参考收盘）。"""
    out = {"n_buy": 0, "n_sell": 0, "n_blocked": 0, "buy_amt": 0.0, "sell_amt": 0.0, "asof": None}
    if orders is None or orders.empty:
        return out
    if "asof_date" in orders.columns and orders["asof_date"].notna().any():
        out["asof"] = str(orders["asof_date"].dropna().iloc[-1])[:10]
    blk = blocked_mask(orders)
    out["n_blocked"] = int(blk.sum())
    for r in orders[~blk].itertuples(index=False):
        px = order_price(getattr(r, "limit_price", None), getattr(r, "ref_close", None))
        if px is None:
            continue
        amt = (as_float(getattr(r, "shares", 0)) or 0.0) * px
        side = getattr(r, "side", "")
        if side == "buy":
            out["n_buy"] += 1
            out["buy_amt"] += amt
        elif side == "sell":
            out["n_sell"] += 1
            out["sell_amt"] += amt
    return out


# ----------------------------------------------------------------------------- 执行勾选
def rejected_keys(state: dict | None) -> set[str]:
    return {k for k, v in (state or {}).items() if v == "rejected"}


def done_keys(state: dict | None) -> set[str]:
    return {k for k, v in (state or {}).items() if v in ("done", True)}


def exec_progress(orders: pd.DataFrame, state: dict | None) -> tuple[int, int, int]:
    """(已执行, 应执行, 已拒绝)。应执行 = 非拦截且未被拒绝的单。"""
    if orders is None or orders.empty:
        return 0, 0, 0
    blk = blocked_mask(orders)
    rej, done = rejected_keys(state), done_keys(state)
    keys = [order_key(r.side, r.symbol) for r in orders[~blk].itertuples(index=False)]
    n_rej = sum(1 for k in keys if k in rej)
    todo = [k for k in keys if k not in rej]
    return sum(1 for k in todo if k in done), len(todo), n_rej


def orders_fresh(orders_day: str | None, data_day: str | None) -> bool:
    """最新订单是否仍在当前执行周期：订单日期 ≥ 行情最新日（与总览「订单已过期」同口径）。

    订单是当晚收盘后生成、次日开盘执行的，所以订单日期等于行情最新日就是新鲜的。
    没有行情缓存时无从比较，视为新鲜（只会出现在没有数据的沙盒里）。
    """
    if not orders_day:
        return False
    if not data_day:
        return True
    return str(orders_day) >= str(data_day)


def _hhmm(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M")
    except (TypeError, ValueError, OSError):
        return ""


def flow_steps(
    *,
    orders_day: str | None,
    data_day: str | None,
    progress: tuple[int, int, int],
    has_orders: bool,
    orders_mtime: float | None,
    holdings_mtime: float | None,
    account: dict,
) -> list[dict]:
    """今日流程四步。state ∈ ok | warn | off。纯函数。"""
    fresh = orders_fresh(orders_day, data_day)
    x, n, n_rej = progress
    # ① 生成订单
    if orders_day and fresh:
        s1 = ("ok", f"订单 {orders_day}（次日开盘执行）")
    elif orders_day:
        s1 = ("warn", f"订单 {orders_day} 已过期，行情已到 {data_day}")
    else:
        s1 = ("warn", "还没有订单")
    # ② 照单执行
    suffix = f"（已拒绝 {n_rej} 笔）" if n_rej else ""
    if not has_orders or not fresh:
        s2 = ("off", "等待当前周期订单")
    elif n == 0:
        s2 = ("ok", f"应执行单已全部拒绝（{n_rej} 笔）") if n_rej else ("off", "没有需要执行的单")
    elif x >= n:
        s2 = ("ok", f"已全部执行 {n}/{n}{suffix}")
    else:
        s2 = ("warn", f"已执行 {x}/{n}{suffix}")
    # ③ 成交确认
    if not (has_orders and fresh):
        s3 = ("off", "需先有当前周期订单")
    elif holdings_mtime is not None and orders_mtime is not None and holdings_mtime >= orders_mtime:
        s3 = ("ok", f"已确认 {_hhmm(holdings_mtime)}")
    else:
        s3 = ("warn", "待确认（拦截/拒绝单跳过，其余假设成交）")
    # ④ 现金维护
    if not account.get("exists"):
        s4 = ("off", "未建账，输入现金保存即创建")
    elif account.get("stale"):
        s4 = ("warn", f"已 {account['age_days']:.0f} 天未更新（上限 {account['max_age_days']} 天）")
    else:
        s4 = ("ok", f"现金 ¥{account['cash']:,.0f}" if account.get("cash") is not None else "新鲜")
    return [
        {"key": "gen", "title": "生成订单", "state": s1[0], "text": s1[1]},
        {"key": "exec", "title": "照单执行", "state": s2[0], "text": s2[1]},
        {"key": "fill", "title": "成交确认", "state": s3[0], "text": s3[1]},
        {"key": "cash", "title": "现金维护", "state": s4[0], "text": s4[1]},
    ]


# ----------------------------------------------------------------------------- 风控五层
def risk_layers(state: dict | None, iv_scale: float | None = None, iv_status="unknown"):
    """从 live_risk_state 现算五层乘数（复用 risk.compute_risk_scale 的唯一实现）。

    逐层拆法：vol 层 = 关 dd/IC 后调用；vol×dd = 关 IC 后调用；dd 层 = (vol×dd)/vol；
    IC 层取完整调用返回的 ic_scale。regime 默认禁用；IV 层由调用方预计算传入。
    返回 {"total", "layers":[{key,name,scale,status,reason}]}，无净值历史 → None。
    status ∈ ok | warn | danger | unknown | disabled。
    """
    if not state:
        return None
    tvs = [float(v) for v in (state.get("total_values") or [])]
    icq = [float(x) for x in (state.get("ic_queue") or [])]
    if not tvs:
        return None
    total, ic_scale, roll_ic = compute_risk_scale(tvs, icq, iv_scale=iv_scale or 1.0)
    vol_only, _, _ = compute_risk_scale(
        tvs, icq, params={"use_dd_brake": False, "use_ic_scale": False}
    )
    vol_dd, _, _ = compute_risk_scale(tvs, icq, params={"use_ic_scale": False})
    dd_scale = vol_dd / vol_only if vol_only > 1e-12 else 1.0

    rv_ann = dd = None
    if len(tvs) >= 2:
        arr = np.asarray(tvs, dtype=float)
        rets = arr[1:] / arr[:-1] - 1.0
        rv_ann = float(np.std(rets[-C.VOL_WINDOW :]) * np.sqrt(252.0))
        dd = float(arr[-1] / arr.max() - 1.0)

    layers: list[dict] = []

    def put(key, name, scale, status, reason):
        layers.append(
            {"key": key, "name": name, "scale": scale, "status": status, "reason": reason}
        )

    if not C.USE_VOL_TARGET:
        put("vol", "波动目标", None, "disabled", "config 已关闭（USE_VOL_TARGET=0）")
    elif len(tvs) < 6:
        put("vol", "波动目标", None, "unknown", f"净值历史不足 6 日（当前 {len(tvs)} 日），未启用")
    else:
        st = "ok" if vol_only >= 0.999 else ("warn" if vol_only >= 0.5 else "danger")
        put(
            "vol",
            "波动目标",
            vol_only,
            st,
            f"近 {C.VOL_WINDOW} 日实现波动 {rv_ann:.0%}，目标 {C.TARGET_VOL:.0%}",
        )
    if not C.USE_DD_BRAKE:
        put("dd", "回撤刹车", None, "disabled", "config 已关闭（USE_DD_BRAKE=0）")
    elif len(tvs) < 6:
        put("dd", "回撤刹车", None, "unknown", f"净值历史不足 6 日（当前 {len(tvs)} 日），未启用")
    else:
        put(
            "dd",
            "回撤刹车",
            dd_scale,
            "danger" if dd_scale < 0.999 else "ok",
            f"当前回撤 {dd:.1%}（阈值 -{C.DD_THRESHOLD:.0%}）",
        )
    if not C.USE_IC_SCALE:
        put("ic", "IC 降仓", None, "disabled", "config 已关闭（USE_IC_SCALE=0）")
    elif roll_ic is None:
        put("ic", "IC 降仓", None, "unknown", f"成熟 OOS IC 样本不足（{len(icq)} 个），暂不降仓")
    else:
        st = "ok" if ic_scale >= 0.999 else ("danger" if ic_scale <= 0.35 else "warn")
        put(
            "ic",
            "IC 降仓",
            ic_scale,
            st,
            f"近 {C.IC_SCALE_WINDOW} 日成熟 OOS IC 均值 {roll_ic:+.4f}"
            f"（满仓线 {C.IC_FULL_SCALE:+.2f} / 清零线 {C.IC_ZERO_SCALE:+.2f}）",
        )
    if not C.USE_REGIME_FILTER:
        put("regime", "择时 regime", None, "disabled", "已禁用（USE_REGIME_FILTER=0）")
    else:
        put("regime", "择时 regime", 1.0, "ok", "已启用（决策端实时判定，本页不重复计算）")
    if iv_status == "disabled":
        put("iv", "IV 降仓", None, "disabled", "config 已关闭（USE_IV_SCALE=0）")
    elif iv_scale is None:
        put("iv", "IV 降仓", None, "unknown", "期权隐波数据不可得（opt_vix_like 缺失）")
    else:
        st = "ok" if iv_scale >= 0.999 else ("warn" if iv_scale >= 0.6 else "danger")
        reason = (
            f"opt_vix_like 处于近 {C.IV_SCALE_WINDOW} 日高三分位，已降仓"
            if iv_scale < 0.999
            else f"opt_vix_like 未处于近 {C.IV_SCALE_WINDOW} 日高三分位"
        )
        put("iv", "IV 降仓", iv_scale, st, reason)
    return {"total": float(total), "layers": layers}


# ----------------------------------------------------------------------------- 账户 / 目标
def account_snapshot(
    cash: float | None, holdings: pd.DataFrame, last_px: dict, prev_px: dict
) -> dict | None:
    """现金 + 持仓市值 + 净值 + 当日盈亏。现金未知 → None（空态）。"""
    if cash is None:
        return None
    mv = pnl = 0.0
    pnl_known = False
    if holdings is not None and not holdings.empty:
        for r in holdings.itertuples(index=False):
            sh = as_float(getattr(r, "shares", 0)) or 0.0
            if sh <= 0:
                continue
            px = last_px.get(str(r.symbol)) or as_float(getattr(r, "ref_close", None))
            if px is None:
                continue
            mv += sh * px
            ppx = prev_px.get(str(r.symbol))
            if ppx:
                pnl += sh * (px - ppx)
                pnl_known = True
    return {
        "cash": float(cash),
        "mv": mv,
        "nav": float(cash) + mv,
        "pnl": pnl if pnl_known else None,
    }


def target_vs_current(
    orders: pd.DataFrame,
    holdings: pd.DataFrame,
    last_px: dict,
    nav: float | None,
    skip_keys: set[str] | None = None,
) -> pd.DataFrame:
    """由订单推导目标权重 vs 当前权重（假设未拦截、未拒绝的单全部成交）。

    价格：最新收盘 > 订单价 > 持仓参考价。列：symbol/cur_shares/tgt_shares/cur_w/tgt_w。
    """
    skip_keys = skip_keys or set()
    cur: dict[str, float] = {}
    ref_px: dict[str, float] = {}
    if holdings is not None and not holdings.empty:
        for r in holdings.itertuples(index=False):
            sh = as_float(getattr(r, "shares", 0)) or 0.0
            if sh > 0:
                cur[str(r.symbol)] = sh
                ref = as_float(getattr(r, "ref_close", None))
                if ref:
                    ref_px[str(r.symbol)] = ref
    tgt = dict(cur)
    if orders is not None and not orders.empty:
        blk = blocked_mask(orders)
        for r in orders[~blk].itertuples(index=False):
            sym, side = str(r.symbol), str(r.side)
            if order_key(side, sym) in skip_keys:
                continue
            sh = as_float(getattr(r, "shares", 0)) or 0.0
            px = order_price(getattr(r, "limit_price", None), getattr(r, "ref_close", None))
            if px is not None:
                ref_px.setdefault(sym, px)
            if side == "buy":
                tgt[sym] = tgt.get(sym, 0.0) + sh
            elif side == "sell":
                tgt[sym] = max(0.0, tgt.get(sym, 0.0) - sh)
    total = float(nav) if nav and nav > 0 else None
    rows = []
    for sym in sorted(set(cur) | {s for s, v in tgt.items() if v > 0}):
        px = last_px.get(sym) or ref_px.get(sym)
        cw = tw = None
        if px is not None and total is not None:
            cw, tw = cur.get(sym, 0.0) * px / total, tgt.get(sym, 0.0) * px / total
        rows.append(
            {
                "symbol": sym,
                "cur_shares": cur.get(sym, 0.0),
                "tgt_shares": tgt.get(sym, 0.0),
                "cur_w": cw,
                "tgt_w": tw,
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "cur_shares", "tgt_shares", "cur_w", "tgt_w"])


# ----------------------------------------------------------------------------- 成交确认预览
def confirm_plan(orders: pd.DataFrame, state: dict | None, fresh: bool, confirmed: bool) -> dict:
    """成交确认前的二次确认信息：会写回哪些单、为什么不能确认。纯函数。

    can=False 时 why 说明原因。confirmed=True 表示 holdings.csv 已晚于订单（疑似确认过）。
    """
    out = {"n_sell": 0, "n_buy": 0, "n_blocked": 0, "n_rej": 0, "can": False, "why": ""}
    if orders is None or orders.empty:
        out["why"] = "没有订单"
        return out
    blk = blocked_mask(orders)
    rej = rejected_keys(state)
    out["n_blocked"] = int(blk.sum())
    for r in orders[~blk].itertuples(index=False):
        if order_key(r.side, r.symbol) in rej:
            out["n_rej"] += 1
        elif r.side == "sell":
            out["n_sell"] += 1
        elif r.side == "buy":
            out["n_buy"] += 1
    if not fresh:
        out["why"] = "订单已过期，不能确认成交"
    elif confirmed:
        out["why"] = "holdings.csv 已晚于订单，疑似已确认；重复确认会重复加仓"
    elif out["n_sell"] + out["n_buy"] == 0:
        out["why"] = "没有可确认的单（全部被拦截或拒绝）"
    else:
        out["can"] = True
    return out
