"""订单「为什么」：把一笔订单翻译成决策链白话（触发规则 → 择时 → 模型明细 → 参数 → 持仓）。

纯函数，不依赖 dash。reason 口径见 src/trading.py::decide_orders。
数据缺失的小节自动省略（不报错）；①④ 始终存在。
"""

from __future__ import annotations

import config as C
from model_display import model_blurb, model_label

from . import data
from .trade_logic import as_float, truthy


def explain_context(orders, weights: dict | None = None, params: dict | None = None) -> dict:
    """「分析」面板的外部数据：pred_hist 最新两日得分、模型权重、当日股票数、持仓。"""
    ctx = {"pred": {}, "n_stocks": None, "weights": weights, "holdings": {}, "params": params or {}}
    if orders is None or orders.empty:
        return ctx
    ph = data.pred_hist()
    if not ph.empty and {"date", "symbol", "score"} <= set(ph.columns):
        ph = ph.assign(date=ph["date"].astype(str), symbol=ph["symbol"].astype(str))
        dates = sorted(ph["date"].unique())
        last = ph[ph["date"] == dates[-1]]
        ctx["n_stocks"] = int(last["symbol"].nunique())
        legs = [c for c in ph.columns if c.startswith("pred_")]
        prev_score = {}
        if len(dates) > 1:
            prev = ph[ph["date"] == dates[-2]]
            prev_score = dict(zip(prev["symbol"], prev["score"]))
        want = set(orders["symbol"].astype(str))
        for r in last[last["symbol"].isin(want)].to_dict("records"):
            ctx["pred"][str(r["symbol"])] = {
                "score": float(r["score"]),
                "prev_score": prev_score.get(str(r["symbol"])),
                "legs": {c[5:]: as_float(r[c]) for c in legs if as_float(r.get(c)) is not None},
                "date": dates[-1],
            }
    h = data.holdings()
    if not h.empty and "symbol" in h.columns:
        for r in h.to_dict("records"):
            ctx["holdings"][str(r["symbol"])] = {
                "shares": int(as_float(r.get("shares")) or 0),
                "entry_date": str(r.get("entry_date") or "")[:10],
                "ref_close": as_float(r.get("ref_close")),
            }
    return ctx


def _trigger(base: str, rank_txt: str, rsi, bb_pos, topk: float, hold_ratio: float):
    if base.startswith("RSI超买"):
        rsi_txt = f"{rsi:.0f}" if rsi is not None else "?"
        return (
            "① 触发规则：RSI 超买（择时卖出）",
            f"RSI 升至 {rsi_txt}，≥ 超买线 {C.RSI_SELL_MIN:.0f}，短线过热，获利了结。",
        )
    if base == "触布林上轨":
        bb = f"布林位置 {bb_pos:.2f}" if bb_pos is not None else "收盘价"
        return (
            "① 触发规则：触布林上轨（择时卖出）",
            f"{bb}，逼近或突破布林上轨，短线超涨，获利了结。",
        )
    if base == "跌出ML持有带":
        head = f"模型排名跌至{rank_txt}，" if rank_txt else "模型排名"
        return (
            "① 触发规则：跌出ML持有带（排名衰减卖出）",
            head + f"跌出持有带（要求排名前 {hold_ratio:.0%} 以内），触发卖出。",
        )
    if base == "再平衡:清仓出池":
        return "① 触发规则：再平衡清仓（跌出目标池）", "该股已不在当日目标池，清仓。"
    if base == "再平衡:减仓":
        return "① 触发规则：再平衡减仓（超配回落）", "仍在目标池，但持仓超过目标权重，减仓至目标。"
    if base == "再平衡:建仓":
        tail = f"（{rank_txt}）" if rank_txt else ""
        return "① 触发规则：再平衡建仓（进入目标池）", f"进入当日目标池{tail}，按目标权重新建仓。"
    if base == "再平衡:加仓":
        return "① 触发规则：再平衡加仓（低配补足）", "在目标池内且当前低配，加仓至目标权重。"
    if base == "进入ML候选池":
        head = f"模型排名{rank_txt}，" if rank_txt else "模型排名"
        return (
            "① 触发规则：进入ML候选池（排名驱动）",
            head + f"进入候选池（买入资格线：前 {topk:.0%}）。",
        )
    if base == "布林下轨回踩":
        bb = f"布林位置 {bb_pos:.2f}" if bb_pos is not None else "股价"
        head = f"模型排名{rank_txt}进入候选池，且{bb}、" if rank_txt else f"{bb}，"
        return "① 触发规则：布林下轨回踩（排名+择时）", head + "回踩至布林下轨附近，择时买点。"
    if base == "跌出横截面:清退":
        return (
            "① 触发规则：跌出横截面（清退）",
            "当日决策样本中查不到该股（停牌或被剔除），按最近可得价发清退单；明日仍无法成交则继续持有。",
        )
    return "① 触发规则", base or "—"


def order_explain(row: dict, ctx: dict | None) -> list[dict]:
    """[{"title", "lines"}]：拦截说明（仅拦截单）→ ① 触发 → ② 择时 → ③ 模型 → ④ 参数 → ⑤ 持仓。"""
    ctx = ctx or {}
    sym = str(row.get("symbol") or "")
    side = str(row.get("side") or "")
    reason = str(row.get("reason") or "")
    blocked = truthy(row.get("blocked", False))
    rank_pct, bb_pos, rsi = (as_float(row.get(k)) for k in ("rank_pct", "bb_pos", "rsi"))
    limit, ref = as_float(row.get("limit_price")), as_float(row.get("ref_close"))
    shares = int(as_float(row.get("shares")) or 0)

    params = ctx.get("params") or {}
    topk = float(params.get("topk", C.TOPK_RATIO))
    hold_ratio = max(float(params.get("hold_ratio") or getattr(C, "HOLD_RATIO", topk)), topk)
    n = ctx.get("n_stocks")
    if rank_pct is not None and n:
        rank_txt = f"全市场第 {int(round(rank_pct * n)):,}/{n:,} 名（前 {rank_pct:.1%}）"
    elif rank_pct is not None:
        rank_txt = f"全市场前 {rank_pct:.1%}"
    else:
        rank_txt = ""

    base, kind = reason, ""
    for pre, k in (("涨停拦截:", "up"), ("跌停拦截:", "down")):
        if reason.startswith(pre):
            base, kind = reason[len(pre) :], k

    secs: list[dict] = []
    if blocked:
        txt = (
            "当日涨停：涨停板上的买单几乎无法成交，且追高风险大。该单被拦截，标记为【不执行】，"
            "成交确认会跳过它。"
            if kind == "up"
            else "当日跌停：跌停板上的卖单几乎无法成交。该单被拦截，标记为【不执行】；"
            "若明日仍持有，会按最新排名重新评估。"
        )
        secs.append({"title": "⛔ 拦截说明", "lines": [txt]})
    title, trig = _trigger(base, rank_txt, rsi, bb_pos, topk, hold_ratio)
    secs.append({"title": title, "lines": [trig]})

    timing = []
    if rsi is not None:
        note = (
            "超买区"
            if rsi >= C.RSI_SELL_MIN
            else "偏热" if rsi > C.RSI_BUY_MAX else "超卖区" if rsi <= 30 else "中性，未超买"
        )
        timing.append(f"RSI（>70 偏热、>80 超买）{rsi:.1f}：{note}")
    if bb_pos is not None:
        note = "贴近上轨" if bb_pos >= 0.8 else "贴近下轨" if bb_pos <= 0.2 else "带内运行"
        timing.append(f"布林位置（0=下轨，1=上轨）{bb_pos:.2f}：{note}")
    if timing:
        if side == "sell" and base in ("跌出ML持有带", "再平衡:清仓出池", "再平衡:减仓"):
            timing.append("指标未触发卖出，本单由排名/再平衡驱动，非择时卖出。")
        elif side == "buy" and base == "布林下轨回踩":
            timing.append("择时门槛通过（RSI 未过热、价格回踩至下轨可买区）。")
        elif side == "buy" and (base.startswith("再平衡") or base == "进入ML候选池"):
            timing.append("本次为满仓再平衡口径，未启用指标择时门控。")
        secs.append({"title": "② 择时检查", "lines": timing})

    pred = (ctx.get("pred") or {}).get(sym)
    if pred:
        lines = []
        legs = pred.get("legs") or {}
        if legs:
            lines.append(
                " · ".join(
                    f"{model_label(k)} {v:+.4f}（{'偏多' if v > 0 else '偏弱' if v < 0 else '中性'}）"
                    for k, v in legs.items()
                )
            )
            blurbs = [f"{model_label(k)}：{model_blurb(k)}" for k in legs if model_blurb(k)]
            if blurbs:
                lines.append("；".join(blurbs) + "。")
        w = ctx.get("weights") or {}
        w_txt = " · ".join(f"{model_label(k)} {v:.0%}" for k, v in w.items())
        score_txt = f"融合分 {pred['score']:.3f}"
        prev = as_float(pred.get("prev_score"))
        if prev is not None:
            d = pred["score"] - prev
            score_txt += (
                f"（前一日 {prev:.3f}，{'略升' if d > 1e-4 else '略降' if d < -1e-4 else '持平'}）"
            )
        lines.append((f"权重 {w_txt} → " if w_txt else "") + score_txt)
        secs.append({"title": "③ 模型明细（各子模型分别打分）", "lines": lines})

    hold = (ctx.get("holdings") or {}).get(sym)
    whole = "（全部持仓）" if side == "sell" and hold and hold.get("shares") == shares else ""
    if limit is not None and ref is not None and abs(limit - ref) < 1e-6:
        px_txt = f"市价单（次日开盘价撮合，参考收盘 {ref:.2f}）"
    elif limit is not None:
        px_txt = f"限价 {limit:.3f}" + (f"（参考收盘 {ref:.2f}）" if ref is not None else "")
        if str(params.get("order_price_mode", C.ORDER_PRICE_MODE)) == "limit_bb":
            rail = "布林上轨" if side == "sell" else "布林下轨"
            cmp_ = "成交价不会低于限价" if side == "sell" else "成交价不会高于限价"
            px_txt = f"限价 {limit:.3f}（挂在{rail}，{cmp_}" + (
                f"；参考收盘 {ref:.2f}）" if ref is not None else "）"
            )
    else:
        px_txt = ""
    px = limit or ref
    amt = f"，约 ¥{shares * px:,.0f}" if shares and px else ""
    side_zh = "卖出" if side == "sell" else "买入"
    secs.append(
        {
            "title": "④ 订单参数",
            "lines": [f"{side_zh} {shares:,} 股{whole}{(' @ ' + px_txt) if px_txt else ''}{amt}"],
        }
    )

    if hold:
        parts = []
        if hold.get("entry_date"):
            parts.append(f"建仓日 {hold['entry_date']}")
        cost = hold.get("ref_close")
        if cost:
            parts.append(f"成本 {cost:.2f}")
            if ref is not None:
                parts.append(f"按参考收盘浮盈 {ref / cost - 1.0:+.1%}")
        if parts:
            secs.append({"title": "⑤ 持仓信息", "lines": [" · ".join(parts)]})
    return secs
