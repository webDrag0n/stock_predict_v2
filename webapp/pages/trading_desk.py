"""交易台 Trading Desk（/）—— 开盘前执行单首页。

对应设计文档 docs/frontend_redesign.html 第 3 节（页面一 · 交易台）：
  今日流程引导卡（生成订单 → 照单执行 → 成交确认 → 现金维护，四步状态灯+动作）
  今日操作卡（最新 outputs/orders_*.csv：订单数/预计成交/风控总乘数）
  风控五层状态灯（live_risk_state.json → 现算各层乘数与触发原因）
  账户快照（account.csv 现金 + holdings.csv × 最新收盘）
  订单执行单（卖单区在上/买单区在下，blocked 置灰，点「执行」/「拒绝」大按钮持久化到
    outputs/execution_state_<日期>.json，刷新不丢；每行末尾「分析▾」可展开该单的
    决策链白话解释：触发规则/择时检查/模型明细/订单参数/持仓信息，数据缺失的小节省略）
  目标仓位 vs 当前持仓 diff（由订单推导目标权重）
  净值与偏差（tracking/live_daily.parquet + tracking_report.json 告警）
  持仓明细表（含现价/浮盈/持有天数，代码跳转 /market?symbol=<代码>）
  执行闭环说明（成交确认/现金维护均可在本页完成，CLI 为等价路径；默认折叠）

出单入口已并入本页：今日流程卡「立即生成」/ 今日操作卡「生成次日订单」按钮弹窗
（资金 + 可编辑持仓表格 + label；持仓预填 = 当前持仓 + 最新订单假设全部成交，
已成交确认过则不重复应用；其余参数取 registry.default_params("nextday") 默认值）
→ scheduler.create("nextday") 创建会话，dcc.Interval 2.5s 心跳轮询 session.json 的
current_progress 在今日操作卡状态区渲染进度（排队/百分比/phase/ETA/跳转
/runs/<sid>）。CLI（./daily_run.sh）仍是等价的定时出单路径；两者产物不互相覆盖
（会话订单落在会话目录 artifacts/）。

点「执行」/「拒绝」自动写盘持久化；成交确认（写回 holdings.csv）与现金维护
（account.csv 追加）均在本页完成，与 CLI ./daily_run.sh commit / 手编
account.csv 为等价路径（写回实现共用 src/execution.py）。

五层风控乘数复用 src/risk.py::compute_risk_scale（回测=实盘唯一实现）做
只读计算：用 params 覆盖逐层拆出 vol / 回撤 / IC 三层乘数；regime 默认禁用；
IV 层尝试从 wholemkt_panel 的 opt_vix_like 现算，不可得显示「未知」。
"""

from __future__ import annotations

import functools
import os
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import config as C
from execution import load_holdings_table, save_holdings
from risk import compute_risk_scale as _risk_scale
from trading import parse_initial_holdings

from pipeline import registry, scheduler

from .. import dataio
from ..components import model_blurb, model_label, page_header, term
from ..theme import PALETTE, base_fig, empty_fig, money

_ACCOUNT_TEMPLATE = "date,cash\n2026-10-22,100000.00"
_POLL_MS = 2500  # 出单进度心跳（与模型实验室 mlab-poll 同节奏）


# ---------------------------------------------------------------------------
# 纯函数：订单统计 / 风控五层 / 账户快照 / 目标仓位（单元测试锚点）
# ---------------------------------------------------------------------------
def _today_str() -> str:
    """今天 YYYYMMDD（独立成函数便于测试 monkeypatch）。"""
    return date.today().strftime("%Y%m%d")


def _blocked_mask(orders: pd.DataFrame) -> pd.Series:
    """blocked 列容错成 bool（csv 读回可能是 True/False 字符串）。"""
    if "blocked" not in orders.columns:
        return pd.Series(False, index=orders.index)
    col = orders["blocked"]
    if pd.api.types.is_bool_dtype(col):
        return col.fillna(False)
    return col.astype(str).str.lower().isin(["true", "1"])


def _order_price(row) -> float:
    """挂单价优先，缺失回落参考收盘；都无效返回 nan。"""
    for c in ("limit_price", "ref_close"):
        v = row.get(c)
        if v is not None and v == v and float(v) > 0:
            return float(v)
    return float("nan")


def order_amounts(orders: pd.DataFrame) -> dict:
    """订单统计：买/卖/拦截笔数 + 预计成交金额（剔除 blocked；价 = limit_price||ref_close）。

    预计成交金额口径：非拦截单的 shares × 价格 求和，买卖分列。
    """
    out = {"n_buy": 0, "n_sell": 0, "n_blocked": 0, "buy_amt": 0.0, "sell_amt": 0.0, "asof": None}
    if orders is None or orders.empty:
        return out
    df = orders
    if "asof_date" in df.columns and df["asof_date"].notna().any():
        out["asof"] = str(df["asof_date"].dropna().iloc[-1])[:10]
    blk = _blocked_mask(df)
    out["n_blocked"] = int(blk.sum())
    for row in df[~blk].itertuples(index=False):
        px = _order_price(row._asdict() if hasattr(row, "_asdict") else row)
        amt = float(getattr(row, "shares", 0) or 0) * px
        if not np.isfinite(amt):
            continue
        if getattr(row, "side", "") == "buy":
            out["n_buy"] += 1
            out["buy_amt"] += amt
        elif getattr(row, "side", "") == "sell":
            out["n_sell"] += 1
            out["sell_amt"] += amt
    return out


def compute_risk_layers(
    state: dict | None,
    iv_scale: float | None = None,
    iv_status: str = "unknown",
) -> dict | None:
    """从 live_risk_state dict 现算五层乘数（只读复用 src/risk.py 唯一实现）。

    逐层拆解方法：vol 层 = 关 dd/IC 后调用；vol×dd = 关 IC 后调用；
    dd 层 = (vol×dd) / vol；IC 层由完整调用的 ic_scale 返回值给出。
    regime 默认禁用（USE_REGIME_FILTER=0）；IV 层由调用方预计算传入。

    Returns:
        {"total": float, "layers": [ {key,name,scale,status,reason}, ... ]}
        status ∈ ok/warn/danger/unknown/disabled；state 无净值历史 → None。
    """
    if not state:
        return None
    tvs = [float(v) for v in (state.get("total_values") or [])]
    icq = [float(x) for x in (state.get("ic_queue") or [])]
    if not tvs:
        return None

    total, ic_scale, roll_ic = _risk_scale(tvs, icq, iv_scale=iv_scale or 1.0)
    vol_only, _, _ = _risk_scale(tvs, icq, params={"use_dd_brake": False, "use_ic_scale": False})
    vol_dd, _, _ = _risk_scale(tvs, icq, params={"use_ic_scale": False})
    dd_scale = vol_dd / vol_only if vol_only > 1e-12 else 1.0

    # 展示用原因量：实现波动（年化）与当前回撤
    rv_ann = dd = None
    if len(tvs) >= 2:
        arr = np.asarray(tvs, dtype=float)
        rets = arr[1:] / arr[:-1] - 1.0
        rv_ann = float(np.std(rets[-C.VOL_WINDOW :]) * np.sqrt(252.0))
        dd = float(arr[-1] / arr.max() - 1.0)

    layers = []

    def _put(key, name, scale, status, reason):
        layers.append(
            {"key": key, "name": name, "scale": scale, "status": status, "reason": reason}
        )

    # A. vol 目标
    if not C.USE_VOL_TARGET:
        _put("vol", "vol目标", None, "disabled", "已在 config 关闭（USE_VOL_TARGET=0）")
    elif len(tvs) < 6:
        _put("vol", "vol目标", None, "unknown", f"净值历史不足 6 日（当前 {len(tvs)} 日），未启用")
    else:
        status = "ok" if vol_only >= 0.999 else ("warn" if vol_only >= 0.5 else "danger")
        _put(
            "vol",
            "vol目标",
            vol_only,
            status,
            f"近{C.VOL_WINDOW}日实现波动 {rv_ann:.0%} vs 目标 {C.TARGET_VOL:.0%}",
        )
    # B. 回撤刹车
    if not C.USE_DD_BRAKE:
        _put("dd", "回撤刹车", None, "disabled", "已在 config 关闭（USE_DD_BRAKE=0）")
    elif len(tvs) < 6:
        _put("dd", "回撤刹车", None, "unknown", f"净值历史不足 6 日（当前 {len(tvs)} 日），未启用")
    else:
        status = "danger" if dd_scale < 0.999 else "ok"
        _put("dd", "回撤刹车", dd_scale, status, f"当前回撤 {dd:.1%}（阈值 -{C.DD_THRESHOLD:.0%}）")
    # C. IC 降仓
    if not C.USE_IC_SCALE:
        _put("ic", "IC 降仓", None, "disabled", "已在 config 关闭（USE_IC_SCALE=0）")
    elif roll_ic is None:
        _put(
            "ic",
            "IC 降仓",
            None,
            "unknown",
            f"成熟 OOS IC 样本不足（{len(icq)} 个），暂不降仓",
        )
    else:
        status = "ok" if ic_scale >= 0.999 else ("danger" if ic_scale <= 0.35 else "warn")
        _put(
            "ic",
            "IC 降仓",
            ic_scale,
            status,
            f"近{C.IC_SCALE_WINDOW}日成熟 OOS IC 均值 {roll_ic:+.4f}"
            f"（满仓线 {C.IC_FULL_SCALE:+.2f} / 清零线 {C.IC_ZERO_SCALE:+.2f}）",
        )
    # D. 择时 regime（默认关）
    if not C.USE_REGIME_FILTER:
        _put("regime", "择时 regime", None, "disabled", "已禁用（USE_REGIME_FILTER=0）")
    else:
        _put("regime", "择时 regime", 1.0, "ok", "已启用（决策端实时判定，本页不重复计算）")
    # E. IV 降仓
    if iv_status == "disabled":
        _put("iv", "IV 降仓", None, "disabled", "已在 config 关闭（USE_IV_SCALE=0）")
    elif iv_scale is None:
        _put("iv", "IV 降仓", None, "unknown", "期权隐波数据不可得（opt_vix_like 缺失）")
    else:
        status = "ok" if iv_scale >= 0.999 else ("warn" if iv_scale >= 0.6 else "danger")
        reason = (
            f"opt_vix_like 处于近{C.IV_SCALE_WINDOW}日高三分位，已降仓"
            if iv_scale < 0.999
            else f"opt_vix_like 未处于近{C.IV_SCALE_WINDOW}日高三分位"
        )
        _put("iv", "IV 降仓", iv_scale, status, reason)
    return {"total": float(total), "layers": layers}


def account_snapshot(
    account_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    last_px: dict,
    prev_px: dict,
) -> dict | None:
    """现金 + 持仓市值 + 净值 + 当日盈亏。account.csv 无有效 cash → None（空态）。"""
    if account_df is None or account_df.empty or "cash" not in account_df.columns:
        return None
    cash_ser = pd.to_numeric(account_df["cash"], errors="coerce").dropna()
    if cash_ser.empty:
        return None
    cash = float(cash_ser.iloc[-1])
    mv = 0.0
    pnl = 0.0
    pnl_known = False
    if holdings_df is not None and not holdings_df.empty:
        for r in holdings_df.itertuples(index=False):
            sh = float(getattr(r, "shares", 0) or 0)
            if sh <= 0:
                continue
            px = last_px.get(r.symbol)
            if px is None:
                ref = getattr(r, "ref_close", None)
                px = float(ref) if ref is not None and ref == ref else None
            if px is None:
                continue
            mv += sh * px
            ppx = prev_px.get(r.symbol)
            if ppx:
                pnl += sh * (px - ppx)
                pnl_known = True
    return {
        "cash": cash,
        "mv": mv,
        "nav": cash + mv,
        "pnl": pnl if pnl_known else None,
    }


def target_vs_current(
    orders: pd.DataFrame,
    holdings_df: pd.DataFrame,
    last_px: dict,
    nav: float | None,
) -> pd.DataFrame:
    """由订单推导目标持仓权重 vs 当前权重（假设非拦截单全部成交）。

    价格：最新收盘 > 订单挂单价 > 持仓 ref_close。返回列：
    symbol/cur_shares/tgt_shares/cur_w/tgt_w。无持仓且无订单 → 空表。
    """
    cur: dict[str, float] = {}
    ref_px: dict[str, float] = {}
    if holdings_df is not None and not holdings_df.empty:
        for r in holdings_df.itertuples(index=False):
            sh = float(getattr(r, "shares", 0) or 0)
            if sh > 0:
                cur[r.symbol] = sh
                ref = getattr(r, "ref_close", None)
                if ref is not None and ref == ref:
                    ref_px[r.symbol] = float(ref)
    tgt = dict(cur)
    if orders is not None and not orders.empty:
        blk = _blocked_mask(orders)
        for row in orders[~blk].itertuples(index=False):
            sh = float(getattr(row, "shares", 0) or 0)
            px = _order_price(row._asdict() if hasattr(row, "_asdict") else row)
            if np.isfinite(px):
                ref_px.setdefault(row.symbol, px)
            if row.side == "buy":
                tgt[row.symbol] = tgt.get(row.symbol, 0.0) + sh
            elif row.side == "sell":
                tgt[row.symbol] = max(0.0, tgt.get(row.symbol, 0.0) - sh)
    rows = []
    total = float(nav) if nav and nav > 0 else None
    for sym in sorted(set(cur) | {s for s, v in tgt.items() if v > 0}):
        px = last_px.get(sym) or ref_px.get(sym)
        if px is None or total is None:
            cw = tw = None
        else:
            cw = cur.get(sym, 0.0) * px / total
            tw = tgt.get(sym, 0.0) * px / total
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


# ---------------------------------------------------------------------------
# 页面专用 loader（lru_cache；数据新鲜度跟随进程生命周期，与 dataio 一致）
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _close_maps() -> tuple[dict, dict, str | None]:
    """最新两个 daily 缓存的收盘价：(prev_map, last_map, last_date)。

    只读尾部两个 daily_*.parquet（各约 5k 行），避免加载 6M 行合并面板。
    """
    files = dataio.daily_files()
    if not files:
        return {}, {}, None

    def _load(path: str) -> dict:
        try:
            df = pd.read_parquet(path, columns=["symbol", "close"])
        except Exception:
            return {}
        return {str(s): float(c) for s, c in zip(df["symbol"], df["close"]) if c == c and c > 0}

    last_date = os.path.basename(files[-1]).replace("daily_", "").replace(".parquet", "")
    prev = _load(files[-2]) if len(files) > 1 else {}
    return prev, _load(files[-1]), last_date


@functools.lru_cache(maxsize=1)
def _iv_scale_now() -> tuple[str, float | None]:
    """IV 降仓乘数：('disabled'|'unknown'|'ok', scale)。

    只读复用 src/live_state._iv_scale 的口径：面板 opt_vix_like 在最新日
    相对近 IV_SCALE_WINDOW 日的三分位插值。数据缺失/异常 → unknown。
    """
    if not getattr(C, "USE_IV_SCALE", False):
        return "disabled", None
    try:
        import pyarrow.parquet as pq

        path = os.path.join(dataio.CACHE, "wholemkt_panel.parquet")
        feat = getattr(C, "IV_FEATURE_FOR_SCALE", "opt_vix_like")
        if not os.path.exists(path) or feat not in pq.read_schema(path).names:
            return "unknown", None
        panel = pd.read_parquet(path, columns=["date", feat])
        if panel.empty or panel[feat].dropna().empty:
            return "unknown", None
        from live_state import _iv_scale

        v = _iv_scale(
            panel,
            panel["date"].max(),
            feat,
            int(getattr(C, "IV_SCALE_WINDOW", 60)),
            float(getattr(C, "IV_MIN_SCALE", 0.5)),
        )
        return "ok", float(v)
    except Exception:
        return "unknown", None


def _orders_fresh(orders_date: str | None) -> bool:
    """最新订单是否在当前执行周期内：订单日期 ≥ 最新日线数据日
    （当晚生成、次日开盘执行，次日早晨订单日期仍等于最新数据日）。
    无日线缓存时回落到「日期 = 今日」。"""
    if not orders_date:
        return False
    dates = dataio.trade_dates()
    if not dates:
        return orders_date == _today_str()
    return orders_date >= dates[-1]


# ---------------------------------------------------------------------------
# 今日流程卡：四步状态判定（纯函数，单元测试锚点）
# ---------------------------------------------------------------------------
def _today_iso() -> str:
    """今天 YYYY-MM-DD（由 _today_str 推导，monkeypatch 一致生效）。"""
    t = _today_str()
    return f"{t[:4]}-{t[4:6]}-{t[6:]}"


def _holdings_mtime() -> float | None:
    """outputs/holdings.csv 的 mtime；无文件 → None。"""
    try:
        return os.path.getmtime(os.path.join(dataio.OUTPUTS, "holdings.csv"))
    except OSError:
        return None


def _fmt_hhmm(ts) -> str:
    """epoch 秒 → 'HH:MM'，非法输入返回 ''。"""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M")
    except (TypeError, ValueError, OSError):
        return ""


def _rejected_rids(state: dict) -> set[str]:
    """执行状态中「已拒绝」的订单键（值 "rejected"；true 为已执行，向后兼容旧格式）。"""
    return {k for k, v in (state or {}).items() if v == "rejected"}


def _exec_progress(orders: pd.DataFrame, state: dict) -> tuple[int, int]:
    """执行勾选进度 (已勾选, 应执行笔数)；仅统计非拦截且未拒绝的单。"""
    if orders is None or orders.empty:
        return 0, 0
    blk = _blocked_mask(orders)
    rej = _rejected_rids(state)
    keys = [
        f"{r.side}-{r.symbol}"
        for r in orders[~blk].itertuples(index=False)
        if f"{r.side}-{r.symbol}" not in rej
    ]
    return sum(1 for k in keys if state.get(k) is True), len(keys)


def _step1_state(orders_date, orders_mtime, source, today, is_trading, fresh) -> tuple[str, str]:
    """① 生成订单：最新订单在当前执行周期内 → 绿；非交易日 → 灰；否则黄。"""
    if is_trading is False:
        return "disabled", "今日非交易日"
    if orders_date and fresh:
        if orders_date == today:
            txt = f"已完成 {_fmt_hhmm(orders_mtime)}".rstrip()
        else:
            txt = f"订单 {orders_date}（早盘待执行）"
        if source and source != "cli":
            txt += f" · {source}"
        return "ok", txt
    if is_trading is None:
        return "warn", "待生成（日历未知）"
    return "warn", "待生成"


def _step2_state(x: int, n: int, has_orders: bool, n_rej: int = 0) -> tuple[str, str]:
    """② 照单执行：勾选进度 x/N（持久化，刷新不丢；拒绝单不计入应执行笔数）。"""
    suffix = f"（已拒绝 {n_rej} 笔）" if n_rej else ""
    if not has_orders:
        return "disabled", "等待订单"
    if n == 0:
        if n_rej:
            return "ok", f"应执行单已全部拒绝（{n_rej} 笔）"
        return "disabled", "等待订单"
    if x >= n:
        return "ok", f"已全部勾选 {n}/{n}{suffix}"
    return "warn", (f"已勾选 {x}/{n}" if x else f"待执行 0/{n}") + suffix


def _step3_state(orders_ready: bool, holdings_mtime, orders_mtime) -> tuple[str, str]:
    """③ 成交确认：holdings.csv mtime 晚于最新订单 → 绿；无当前周期订单 → 灰。"""
    if not orders_ready:
        return "disabled", "需先有当前周期订单"
    if holdings_mtime is not None and orders_mtime is not None and holdings_mtime >= orders_mtime:
        return "ok", f"已确认 {_fmt_hhmm(holdings_mtime)}"
    return "warn", "待确认（拦截/拒绝单跳过，其余假设成交）"


def _step4_state(acct: pd.DataFrame, today_iso: str) -> tuple[str, str]:
    """④ 现金维护：account.csv 最后一行日期 = 今日 → 绿；未建账 → 灰；否则黄（仅提示）。"""
    if acct is None or acct.empty or "date" not in acct.columns:
        return "disabled", "未建账（输入现金保存即创建）"
    dates = acct["date"].dropna()
    last = str(dates.iloc[-1])[:10] if len(dates) else ""
    if last == today_iso:
        return "ok", "今日已维护"
    return "warn", f"待更新（上次 {last or '—'}，非每日必需）"


def _latest_cash(acct: pd.DataFrame) -> float | None:
    """account.csv 最后一行非空 cash；无 → None。"""
    if acct is None or acct.empty or "cash" not in acct.columns:
        return None
    v = pd.to_numeric(acct["cash"], errors="coerce").dropna()
    return float(v.iloc[-1]) if len(v) else None


def _compute_flow(acct: pd.DataFrame) -> dict:
    """汇总四步状态 + 成交确认可用性（layout 初渲染与 td-poll 轮询共用）。"""
    today = _today_str()
    orders, meta = dataio.latest_orders_with_meta()
    is_trading = dataio.is_trading_day(_today_iso())
    ready = not orders.empty and _orders_fresh(meta.get("date"))
    state = dataio.load_execution_state(today)
    x, n = _exec_progress(orders, state)
    rej = _rejected_rids(state)
    n_rej = 0
    if not orders.empty and rej:
        blk = _blocked_mask(orders)
        n_rej = sum(
            1 for r in orders[~blk].itertuples(index=False) if f"{r.side}-{r.symbol}" in rej
        )
    return {
        "s1": _step1_state(
            meta.get("date"), meta.get("mtime"), meta.get("source", ""), today, is_trading, ready
        ),
        "s2": _step2_state(x, n, not orders.empty, n_rej),
        "s3": _step3_state(ready, _holdings_mtime(), meta.get("mtime")),
        "s4": _step4_state(acct, _today_iso()),
        "cash": _latest_cash(acct),
        "can_commit": ready,
    }


# ---------------------------------------------------------------------------
# 出单入口：参数组装 / 预填 / 进度渲染（纯函数，单元测试锚点）
# ---------------------------------------------------------------------------
def projected_holdings(orders: pd.DataFrame, holdings_df: pd.DataFrame) -> list[dict]:
    """预填持仓 = 当前持仓 + 订单假设全部成交（blocked 跳过；减到 0 删除行）。

    返回 [{"symbol":..., "shares": int}]，顺序：原持仓顺序 + 新买入出现顺序。
    """
    pos: dict[str, int] = {}
    seq: list[str] = []

    def _touch(sym: str) -> None:
        if sym not in pos:
            seq.append(sym)
            pos[sym] = 0

    if holdings_df is not None and not holdings_df.empty:
        for r in holdings_df.itertuples(index=False):
            sh = int(float(getattr(r, "shares", 0) or 0))
            if sh > 0:
                _touch(str(r.symbol))
                pos[str(r.symbol)] += sh
    if orders is not None and not orders.empty:
        blk = _blocked_mask(orders)
        for o in orders[~blk].itertuples(index=False):
            sym = str(o.symbol)
            sh = int(float(getattr(o, "shares", 0) or 0))
            _touch(sym)
            if o.side == "buy":
                pos[sym] += sh
            elif o.side == "sell":
                pos[sym] = max(0, pos[sym] - sh)
    return [{"symbol": s, "shares": pos[s]} for s in seq if pos.get(s, 0) > 0]


def _prefill_holding_rows() -> tuple[list[dict], str]:
    """弹窗持仓表格预填：当前持仓 + 最新订单假设全部成交。

    holdings.csv 已晚于最新订单写回（已成交确认）时，订单不重复应用。
    Returns: (rows, 预填口径说明)
    """
    holds = dataio.holdings()
    orders, meta = dataio.latest_orders_with_meta()
    o_mt = meta.get("mtime")
    h_mt = _holdings_mtime()
    if orders.empty:
        rows = projected_holdings(pd.DataFrame(), holds)
        note = "预填 = 当前持仓（可修改）。" if rows else "无持仓且无订单，空表起步（可添加行）。"
    elif h_mt is not None and o_mt is not None and h_mt >= o_mt:
        rows = projected_holdings(pd.DataFrame(), holds)
        note = "预填 = 当前持仓（最新订单已成交确认，不重复应用）。"
    else:
        rows = projected_holdings(orders, holds)
        note = "预填 = 当前持仓 + 最新订单假设全部成交（可修改）。"
    return (
        rows,
        note + "股数自动向下取整到 100 股整手；重复代码将合并；其余参数取 registry 默认值。",
    )


_SYMBOL_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


def rows_to_holdings_text(rows) -> tuple[str, str | None]:
    """持仓表格行 → `代码:股数` 多行文本（走 parse_initial_holdings 链路）。

    校验：代码需形如 000001.SZ（6 位数字.交易所后缀，大小写不敏感）；
    股数为正数；空白行跳过；重复代码合并（UI 已说明）。
    整手（100 股）规则由下游 parse_initial_holdings 统一向下取整。

    Returns: (text, error)；error 非 None 时 text 为 ""。
    """
    merged: dict[str, float] = {}
    seq: list[str] = []
    for i, r in enumerate(rows or [], 1):
        r = r or {}
        sym = str(r.get("symbol") or "").strip().upper()
        sh_raw = r.get("shares")
        if not sym and (sh_raw is None or str(sh_raw).strip() == ""):
            continue
        if not _SYMBOL_RE.match(sym):
            return "", f"第{i}行股票代码格式错误（应形如 000001.SZ）：{sym or '（空）'!r}"
        try:
            sh = float(sh_raw)
        except (TypeError, ValueError):
            return "", f"第{i}行股数不是数字：{sh_raw!r}"
        if not np.isfinite(sh) or sh <= 0:
            return "", f"第{i}行股数必须为正数：{sh_raw!r}"
        if sym not in merged:
            seq.append(sym)
            merged[sym] = 0.0
        merged[sym] += sh
    return "\n".join(f"{s}:{int(merged[s])}" for s in seq), None


def _default_capital(snap: dict | None) -> float:
    """资金预填口径：账户快照卡净值（现金 + 持仓市值）；无快照回落 registry 默认。"""
    if snap and snap.get("nav"):
        return float(round(float(snap["nav"]), 2))
    return float(registry.default_params("nextday")["capital"])


def _build_nextday_params(capital, holdings_text, label) -> tuple[dict | None, str, str | None]:
    """组装 nextday 会话参数：registry 默认值 + 资金/持仓/label 三个覆盖项。

    持仓文本逐行校验（复用 trading.parse_initial_holdings 的解析口径）：
    非空、非注释(#)行必须能解析出持仓（缺分隔符 / 股数非数字 / 不足一手
    都会被该解析器丢弃，这里视为格式错误并指出行号）。

    Returns:
        (params, label, error)：error 非 None 时 params 为 None；
        label 为去空白后的字符串（"" 由 scheduler.create 落默认名）。
    """
    text = (holdings_text or "").strip()
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not parse_initial_holdings(s):
            bad.append(f"第{i}行 {s!r}")
    if bad:
        shown = "；".join(bad[:3]) + (" …" if len(bad) > 3 else "")
        return None, "", f"持仓格式错误（每行应为 代码:股数，股数为整百正数）：{shown}"
    try:
        cap = float(capital)
    except (TypeError, ValueError):
        cap = float("nan")
    if not np.isfinite(cap) or cap < 10000:
        return None, "", f"组合资金必须是不小于 10000 的数字（当前 {capital!r}）"
    params = registry.default_params("nextday")
    params["capital"] = cap
    params["initial_holdings"] = text
    return params, (label or "").strip(), None


def _cli_orders_today() -> bool:
    """今日已有 CLI 出单产物（outputs/orders_<今天>.csv）。会话产物落在会话目录，不覆盖。"""
    return os.path.exists(os.path.join(dataio.OUTPUTS, f"orders_{_today_str()}.csv"))


def _active_nextday(sessions: list[dict]) -> dict | None:
    """最新一个 queued/running 的 nextday 会话（list_sessions 已按创建时间倒序）。"""
    for s in sessions:
        if s.get("mode") == "nextday" and s.get("status") in ("queued", "running"):
            return s
    return None


def _latest_finished_nextday(sessions: list[dict]) -> dict | None:
    """最新一个已终态（done/failed/killed）的 nextday 会话。"""
    for s in sessions:
        if s.get("mode") == "nextday" and s.get("status") in ("done", "failed", "killed"):
            return s
    return None


def _session_overall_pct(meta: dict) -> float:
    """整体百分比 = (done 步数 + 当前步子进度 pct) / 总步数 ×100；字段缺省一律容错为 0。"""
    steps = meta.get("steps") or []
    n = len(steps)
    if not n:
        return 0.0
    done = sum(1 for s in steps if s.get("status") == "done")
    sub = 0.0
    cur = meta.get("current_step", -1)
    if isinstance(cur, int) and 0 <= cur < n:
        prog = meta.get("current_progress") or {}
        try:
            sub = float(prog.get("pct", 0.0))
        except (TypeError, ValueError):
            sub = 0.0
        sub = max(0.0, min(1.0, sub))
    return max(0.0, min(100.0, (done + sub) / n * 100))


def _fmt_eta(sec) -> str:
    """ETA 秒 → '预计余 45s' / '预计余 3m'；非法输入返回空串。"""
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(sec) or sec <= 0:
        return ""
    return f"预计余 {sec:.0f}s" if sec < 120 else f"预计余 {int(round(sec / 60))}m"


def _fmt_ts(ts) -> str:
    """epoch 秒 → 'YYYY-MM-DD HH:MM'，非法输入返回 ''。"""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


def _detail_link(sid: str) -> dcc.Link:
    return dcc.Link(
        "查看详情 →",
        href=f"/runs/{sid}",
        style={"fontSize": "12px", "marginLeft": "8px"},
    )


def _render_session_status(meta: dict) -> html.Div:
    """活跃 nextday 会话的状态块：queued → 排队中文案；running → Progress + phase/ETA 小字。"""
    sid = str(meta.get("id", ""))
    label = meta.get("label") or sid
    if meta.get("status") == "queued":
        return html.Div(
            [
                dbc.Spinner(size="sm", spinner_style={"width": "1rem", "height": "1rem"}),
                html.Span(
                    f"  会话 {sid}（{label}）排队中，等待调度…",
                    style={"fontSize": "12.5px", "marginLeft": "6px"},
                ),
                _detail_link(sid),
            ],
            style={"marginTop": "8px"},
        )
    pct = _session_overall_pct(meta)
    prog = meta.get("current_progress") or {}
    steps = meta.get("steps") or []
    cur = meta.get("current_step", -1)
    step_txt = ""
    if isinstance(cur, int) and 0 <= cur < len(steps):
        step_txt = f"步骤 {cur + 1}/{len(steps)} {steps[cur].get('title', '')}"
    small = " · ".join(
        x
        for x in (
            step_txt,
            str(prog.get("phase") or ""),
            str(prog.get("message") or ""),
            _fmt_eta(prog.get("eta_sec")),
        )
        if x
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"会话 {sid}（{label}）运行中", style={"fontSize": "12.5px"}),
                    _detail_link(sid),
                ],
                style={"marginBottom": "4px"},
            ),
            dbc.Progress(
                value=round(pct, 1),
                label=f"{pct:.0f}%",
                striped=True,
                animated=True,
                color="warning",
                style={"height": "16px", "fontSize": "11px"},
            ),
            html.Div(
                small or "等待子阶段上报…",
                style={"fontSize": "11.5px", "color": PALETTE["muted"], "marginTop": "4px"},
            ),
        ],
        style={"marginTop": "8px"},
    )


def _render_idle_status(meta: dict | None, cli_today: bool) -> html.Div:
    """无活跃会话时的状态区：今日已有订单文件防双写提示（常显）+ 最近完成态摘要。"""
    parts = []
    if cli_today:
        parts.append(
            dbc.Alert(
                "今日已有订单文件（outputs/orders_"
                + _today_str()
                + ".csv），新会话产物落在会话目录，不覆盖该文件。",
                color="warning",
                className="py-1 px-2 mb-1",
                style={"fontSize": "12px"},
            )
        )
    if meta is not None:
        sid = str(meta.get("id", ""))
        zh = {
            "done": "完成",
            "failed": "失败",
            "killed": "已终止",
        }.get(meta.get("status"), str(meta.get("status", "")))
        when = _fmt_ts(meta.get("finished") or meta.get("created"))
        parts.append(
            html.Div(
                [
                    html.Span(
                        f"最近出单会话 {sid}：{zh}" + (f"（{when}）" if when else ""),
                        style={"fontSize": "12px", "color": PALETTE["muted"]},
                    ),
                    _detail_link(sid),
                ]
            )
        )
    return html.Div(parts, style={"marginTop": "8px"}) if parts else html.Div()


def _gen_modal(snap: dict | None) -> dbc.Modal:
    """「生成次日订单」弹窗：label + 资金 + 可编辑持仓表格（打开时按预填口径填充）。"""
    cap_hint = (
        "预填 = 账户快照卡净值（现金 + 持仓市值）。"
        if snap
        else "无账户快照，预填为 registry 默认 1,000,000 元。"
    )
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("生成次日订单（nextday 会话）")),
            dbc.ModalBody(
                [
                    html.Div("会话名称（可选）", className="control-label"),
                    dbc.Input(id="td-gen-label", placeholder="留空自动命名", size="sm"),
                    html.Div(
                        "组合资金（元）", className="control-label", style={"marginTop": "10px"}
                    ),
                    dbc.Input(
                        id="td-gen-capital",
                        type="number",
                        value=_default_capital(snap),
                        min=10000,
                        step=10000,
                        size="sm",
                    ),
                    html.Div(cap_hint, style={"fontSize": "11px", "color": PALETTE["muted"]}),
                    html.Div(
                        "当前持仓（可编辑：改数量 / 删行 / 添加行）",
                        className="control-label",
                        style={"marginTop": "10px"},
                    ),
                    dash_table.DataTable(
                        id="td-gen-holdings-table",
                        columns=[
                            {"name": "股票代码", "id": "symbol"},
                            {"name": "股数", "id": "shares", "type": "numeric"},
                        ],
                        data=[],
                        editable=True,
                        row_deletable=True,
                        page_size=8,
                        style_table={"maxHeight": "260px", "overflowY": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "fontSize": 12,
                            "padding": "4px",
                            "fontFamily": "monospace",
                        },
                        style_header={"fontWeight": 700},
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "添加行",
                                id="td-gen-addrow",
                                size="sm",
                                outline=True,
                                color="secondary",
                                className="mt-1",
                            ),
                        ]
                    ),
                    html.Div(
                        id="td-gen-prefill-note",
                        style={"fontSize": "11px", "color": PALETTE["muted"], "marginTop": "6px"},
                    ),
                    html.Div(id="td-gen-alert", style={"marginTop": "8px"}),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "取消",
                        id="td-gen-cancel",
                        size="sm",
                        outline=True,
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button("创建并开始", id="td-gen-submit", size="sm", color="success"),
                ]
            ),
        ],
        id="td-gen-modal",
        is_open=False,
    )


def _commit_modal() -> dbc.Modal:
    """「确认成交」弹窗：列出将成交的买/卖笔数，二次确认后写回 holdings.csv。"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("成交确认（写回 holdings.csv）")),
            dbc.ModalBody(html.Div(id="td-commit-body")),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "取消",
                        id="td-commit-cancel",
                        size="sm",
                        outline=True,
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button("确认成交并写回", id="td-commit-run", size="sm", color="danger"),
                ]
            ),
        ],
        id="td-commit-modal",
        is_open=False,
    )


def _flow_status_block(step: str, state: tuple[str, str]) -> html.Div:
    """流程单步状态行（状态灯 + 文案）；id 供 td-poll 轮询刷新。"""
    lamp, text = state
    return html.Div([_lamp(lamp), text], id=f"td-flow-{step}-status", className="td-flow-status")


def _daily_flow_card(flow: dict) -> dbc.Card:
    """今日流程引导卡：生成订单 → 照单执行 → 成交确认 → 现金维护（状态灯 + 动作）。"""
    steps = [
        html.Div(
            [
                html.Div("① 生成订单", className="td-flow-title"),
                _flow_status_block("s1", flow["s1"]),
                dbc.Button("立即生成", id="td-flow-gen", color="primary", size="sm", outline=True),
            ],
            className="td-flow-step",
        ),
        html.Div(
            [
                html.Div("② 照单执行", className="td-flow-title"),
                _flow_status_block("s2", flow["s2"]),
                html.Div("在下方执行单点「执行」/「拒绝」", className="td-flow-hint"),
            ],
            className="td-flow-step",
        ),
        html.Div(
            [
                html.Div("③ 成交确认", className="td-flow-title"),
                _flow_status_block("s3", flow["s3"]),
                dbc.Button(
                    "确认成交",
                    id="td-commit-open",
                    color="danger",
                    size="sm",
                    outline=True,
                    disabled=not flow["can_commit"],
                ),
            ],
            className="td-flow-step",
        ),
        html.Div(
            [
                html.Div("④ 现金维护", className="td-flow-title"),
                _flow_status_block("s4", flow["s4"]),
                html.Div(
                    [
                        dbc.Input(
                            id="td-cash-input",
                            type="number",
                            value=flow["cash"],
                            placeholder="现金（元）",
                            min=0,
                            step=100,
                            size="sm",
                            className="td-flow-cash",
                        ),
                        dbc.Button("保存", id="td-cash-save", color="primary", size="sm"),
                    ],
                    className="td-flow-cashrow",
                ),
                html.Span(id="td-cash-msg", className="td-flow-hint"),
            ],
            className="td-flow-step",
        ),
    ]
    row = []
    for i, st in enumerate(steps):
        if i:
            row.append(html.Span("→", className="td-flow-arrow"))
        row.append(st)
    return _card("今日流程", html.Div(row, className="td-flow-row"))


def _gen_entry_block() -> html.Div:
    """出单入口：小标题 + 按钮 + 状态区（进度/提示由回调填充），与上方订单区分隔。"""
    return html.Div(
        [
            html.Div(
                "生成次日订单",
                style={"fontWeight": 600, "fontSize": "12.5px", "marginBottom": "6px"},
            ),
            html.Div(
                [
                    dbc.Button("生成次日订单", id="td-gen-open", color="primary", size="sm"),
                    html.Span(
                        " 创建 nextday 会话（分钟级），进度在下方实时展示",
                        style={
                            "fontSize": "11px",
                            "color": PALETTE["muted"],
                            "marginLeft": "8px",
                        },
                    ),
                ]
            ),
            html.Div(id="td-gen-status"),
        ],
        style={
            "marginTop": "10px",
            "borderTop": f"1px solid {PALETTE['grid']}",
            "paddingTop": "8px",
        },
    )


# ---------------------------------------------------------------------------
# 页面卡片组件（今日流程 / 今日操作 / 风控五层 / 账户快照 / 执行单 / 目标仓位 / 净值偏差 / 持仓 / 闭环）
# ---------------------------------------------------------------------------
def _lamp(status: str) -> html.Span:
    color = {
        "ok": PALETTE["up"],
        "warn": "#d97706",
        "danger": PALETTE["down"],
        "disabled": PALETTE["faint"],
        "unknown": PALETTE["faint"],
    }.get(status, PALETTE["faint"])
    return html.Span(
        style={
            "display": "inline-block",
            "width": "10px",
            "height": "10px",
            "borderRadius": "50%",
            "backgroundColor": color,
            "marginRight": "6px",
        }
    )


def _card(title: str, body, **kwargs) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(title, style={"fontWeight": 600, "fontSize": "13.5px"}),
            dbc.CardBody(body, style={"fontSize": "13px"}),
        ],
        className="chart-card",
        style={"marginBottom": "14px"},
        **kwargs,
    )


def _empty_hint(lines) -> html.Div:
    """空态引导：说明 + 生成命令。"""
    return html.Div(
        [html.Div(line, style={"marginBottom": "6px"}) for line in lines],
        style={"color": PALETTE["muted"]},
    )


def _card_a(orders: pd.DataFrame, risk: dict | None, orders_date: str | None) -> dbc.Card:
    """今日操作卡：上半「今日订单」KPI，下半「生成次日订单」出单入口与进度状态区。"""
    if orders.empty:
        body = _empty_hint(["今晚订单未生成。点下方「生成次日订单」，持仓自动预填，一键出单。"])
        return _card("今日操作", [body, _gen_entry_block()])
    stats = order_amounts(orders)
    stale = orders_date is not None and not _orders_fresh(orders_date)
    total_txt = f"×{risk['total']:.2f}" if risk else "未初始化"
    total_color = (
        PALETTE["up"]
        if risk and risk["total"] >= 0.999
        else ("#d97706" if risk else PALETTE["faint"])
    )
    rows = [
        html.Div(
            "今日订单",
            style={"fontWeight": 600, "fontSize": "12.5px", "marginBottom": "6px"},
        ),
        dbc.Row(
            [
                dbc.Col(html.Div("订单日期", className="kpi-label"), width=4),
                dbc.Col(html.Div("买单", className="kpi-label"), width=2),
                dbc.Col(html.Div("卖单", className="kpi-label"), width=2),
                dbc.Col(html.Div("风控总乘数", className="kpi-label"), width=4),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        f"{orders_date or stats['asof'] or '—'}" + (" ⚠ 非今日" if stale else ""),
                        className="kpi-value",
                        style={"fontSize": "18px"},
                    ),
                    width=4,
                ),
                dbc.Col(
                    html.Div(
                        str(stats["n_buy"]),
                        className="kpi-value",
                        style={"fontSize": "18px", "color": PALETTE["up"]},
                    ),
                    width=2,
                ),
                dbc.Col(
                    html.Div(
                        str(stats["n_sell"]),
                        className="kpi-value",
                        style={"fontSize": "18px", "color": PALETTE["down"]},
                    ),
                    width=2,
                ),
                dbc.Col(
                    html.Div(
                        total_txt,
                        className="kpi-value",
                        style={"fontSize": "18px", "color": total_color},
                    ),
                    width=4,
                ),
            ],
            className="mb-2",
        ),
        html.Div(
            f"预计成交（剔除 {stats['n_blocked']} 笔拦截单）："
            f"买入 {money(stats['buy_amt'])} / 卖出 {money(stats['sell_amt'])}"
            f"，净投入 {money(stats['buy_amt'] - stats['sell_amt'])}",
            style={"color": PALETTE["muted"], "fontSize": "12.5px"},
        ),
    ]
    if stale:
        rows.append(
            dbc.Alert(
                "订单已过期（不在当前执行周期）：请点下方「生成次日订单」重新生成。",
                color="warning",
                className="py-1 px-2 mt-2 mb-0",
                style={"fontSize": "12px"},
            )
        )
    rows.append(_gen_entry_block())
    return _card("今日操作", rows)


def _card_b(risk: dict | None) -> dbc.Card:
    """风控五层状态灯。"""
    if risk is None:
        return _card(
            "风控五层状态",
            _empty_hint(
                [
                    "风控未初始化：live_risk_state.json 无净值历史。",
                    html.Div(
                        "在「今日流程」卡④维护现金、①生成订单后，vol/回撤层即开始积累状态。",
                        style={"fontSize": "12px"},
                    ),
                ]
            ),
        )
    rows = [
        html.Div(
            [
                "五层乘数逐层叠乘缩放总仓位；术语悬停看释义，如 ",
                term("IC 定权", id_prefix="td-risk"),
                "。",
            ],
            style={"fontSize": "11.5px", "color": PALETTE["muted"], "marginBottom": "8px"},
        )
    ]
    for lay in risk["layers"]:
        scale_txt = "—" if lay["scale"] is None else f"×{lay['scale']:.2f}"
        rows.append(
            html.Div(
                [
                    _lamp(lay["status"]),
                    html.Span(
                        lay["name"],
                        style={"fontWeight": 600, "minWidth": "72px", "display": "inline-block"},
                    ),
                    html.Span(
                        scale_txt,
                        style={
                            "fontFamily": "monospace",
                            "marginRight": "8px",
                            "minWidth": "52px",
                            "display": "inline-block",
                        },
                    ),
                    html.Span(lay["reason"], style={"color": PALETTE["muted"], "fontSize": "12px"}),
                ],
                style={"marginBottom": "7px"},
            )
        )
    rows.append(
        html.Div(
            f"五层叠乘总乘数：×{risk['total']:.2f}（gross_leverage 将按此缩放）",
            style={
                "borderTop": f"1px solid {PALETTE['grid']}",
                "paddingTop": "6px",
                "fontSize": "12px",
                "color": PALETTE["muted"],
            },
        )
    )
    return _card("风控五层状态", rows)


def _card_c(snap: dict | None) -> dbc.Card:
    """账户快照。"""
    if snap is None:
        return _card(
            "账户快照",
            _empty_hint(
                [
                    "缺少 outputs/account.csv（账户现金）。模板：",
                    html.Pre(_ACCOUNT_TEMPLATE, style={"fontSize": "12px", "margin": "4px 0"}),
                    html.Div(
                        "维护 cash 列后启用净值口径与 vol/回撤风控层。", style={"fontSize": "12px"}
                    ),
                ]
            ),
        )
    pnl = snap["pnl"]
    pnl_txt = "—" if pnl is None else money(pnl)
    pnl_color = (
        PALETTE["muted"] if pnl is None else (PALETTE["up"] if pnl >= 0 else PALETTE["down"])
    )
    return _card(
        "账户快照",
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("现金", className="kpi-label"),
                            html.Div(
                                money(snap["cash"]),
                                className="kpi-value",
                                style={"fontSize": "18px"},
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("持仓市值", className="kpi-label"),
                            html.Div(
                                money(snap["mv"]), className="kpi-value", style={"fontSize": "18px"}
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("净值", className="kpi-label"),
                            html.Div(
                                money(snap["nav"]),
                                className="kpi-value",
                                style={"fontSize": "18px"},
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("当日盈亏(估)", className="kpi-label"),
                            html.Div(
                                pnl_txt,
                                className="kpi-value",
                                style={"fontSize": "18px", "color": pnl_color},
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 订单「分析▾」解释面板：把 decide_orders 的决策链翻译成白话（纯函数为单测锚点）
# ---------------------------------------------------------------------------
def _as_float(v) -> float | None:
    """松散数值解析：None/NaN/非法 → None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _truthy(v) -> bool:
    """csv 读回的 blocked 可能是 True/False 字符串，统一成 bool。"""
    return str(v).strip().lower() in ("true", "1")


def _session_explain_info(meta: dict) -> dict:
    """订单来源会话的辅助信息：决策参数（session.json params）+ 步报告里的模型权重 /
    预测只数。权重与只数优先读 steps/01_train_predict.json 的结构化字段
    （weights / n_predicted，新版管线落盘）；历史会话只有文字摘要，退回正则解析。
    CLI 来源或文件缺失 → 各键空值。"""
    out = {"params": {}, "weights": None, "n_stocks": None}
    path = str((meta or {}).get("path") or "")
    sdir = os.path.dirname(os.path.dirname(path)) if path else ""
    if not sdir or not os.path.isdir(sdir):
        return out
    sess = dataio._read_json(os.path.join(sdir, "session.json")) or {}
    out["params"] = sess.get("params") or {}
    rep = dataio._read_json(os.path.join(sdir, "steps", "01_train_predict.json")) or {}
    w_struct = rep.get("weights")
    if isinstance(w_struct, dict) and w_struct:
        try:
            out["weights"] = {str(k): float(v) for k, v in w_struct.items()}
        except (TypeError, ValueError):
            pass
    n_struct = rep.get("n_predicted")
    if isinstance(n_struct, (int, float)) and n_struct:
        out["n_stocks"] = int(n_struct)
    summary = str(rep.get("summary") or "")
    if out["weights"] is None:
        m = re.search(r"权重\s+(.+)$", summary)
        if m:
            w = {k: float(v) for k, v in re.findall(r"([A-Za-z0-9_]+)=([0-9.]+)", m.group(1))}
            out["weights"] = w or None
    if out["n_stocks"] is None:
        m = re.search(r"已预测\s*([\d,]+)\s*只", summary)
        if m:
            out["n_stocks"] = int(m.group(1).replace(",", ""))
    return out


def _explain_context(orders: pd.DataFrame, meta: dict) -> dict:
    """「分析▾」面板的全部外部数据：pred_hist 最新两日得分、模型权重、当日股票数、持仓。

    任一数据源缺失都降级为 None/空，对应小节在渲染时省略（不报错的空态设计）。
    当日股票数优先取 pred_hist 当日行数，会话步报告只作兜底。
    """
    info = _session_explain_info(meta)
    ctx = {
        "pred": {},
        "n_stocks": info["n_stocks"],
        "weights": info["weights"],
        "holdings": {},
        "params": info["params"],
    }
    if orders is None or orders.empty:
        return ctx
    ph = dataio.pred_hist()
    if not ph.empty and {"date", "symbol", "score"} <= set(ph.columns):
        ph = ph.assign(date=ph["date"].astype(str), symbol=ph["symbol"].astype(str))
        dates = sorted(ph["date"].unique())
        last = ph[ph["date"] == dates[-1]]
        ctx["n_stocks"] = int(last["symbol"].nunique())
        leg_cols = [c for c in ph.columns if c.startswith("pred_")]
        prev_score = {}
        if len(dates) > 1:
            prev = ph[ph["date"] == dates[-2]]
            prev_score = dict(zip(prev["symbol"], prev["score"]))
        want = set(orders["symbol"].astype(str))
        for r in last[last["symbol"].isin(want)].itertuples(index=False):
            legs = {}
            for c in leg_cols:
                v = _as_float(getattr(r, c, None))
                if v is not None:
                    legs[c[5:]] = v
            ctx["pred"][str(r.symbol)] = {
                "score": float(r.score),
                "prev_score": prev_score.get(str(r.symbol)),
                "legs": legs,
                "date": dates[-1],
            }
    holds = dataio.holdings()
    if not holds.empty and "symbol" in holds.columns:
        for r in holds.itertuples(index=False):
            ctx["holdings"][str(r.symbol)] = {
                "shares": int(_as_float(getattr(r, "shares", None)) or 0),
                "entry_date": str(getattr(r, "entry_date", "") or "")[:10],
                "ref_close": _as_float(getattr(r, "ref_close", None)),
            }
    return ctx


def _order_explain(row, ctx: dict) -> list[dict]:
    """单笔订单的「为什么」：按决策链组织成白话小节，返回 [{"title", "lines"}]。

    小节顺序：拦截说明（仅 blocked）→ ① 触发规则 → ② 择时检查 → ③ 模型明细 →
    ④ 订单参数 → ⑤ 持仓信息。数据缺失的小节自动省略；①④ 始终存在。
    reason 口径见 src/trading.py::decide_orders（持有带/RSI超买/布林上轨/回踩/
    候选池/再平衡/清退 + 涨停/跌停拦截前缀）。
    """
    ctx = ctx or {}
    sym = str(getattr(row, "symbol", "") or "")
    side = str(getattr(row, "side", "") or "")
    reason = str(getattr(row, "reason", "") or "")
    blocked = _truthy(getattr(row, "blocked", False))
    rank_pct = _as_float(getattr(row, "rank_pct", None))
    bb_pos = _as_float(getattr(row, "bb_pos", None))
    rsi = _as_float(getattr(row, "rsi", None))
    limit = _as_float(getattr(row, "limit_price", None))
    ref = _as_float(getattr(row, "ref_close", None))
    shares = int(_as_float(getattr(row, "shares", None)) or 0)

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

    base = reason
    blocked_kind = ""
    for pre, kind in (("涨停拦截:", "up"), ("跌停拦截:", "down")):
        if reason.startswith(pre):
            base = reason[len(pre) :]
            blocked_kind = kind

    sections = []
    if blocked:
        if blocked_kind == "up":
            txt = (
                "当日涨停：涨停板上的买单几乎无法成交，且追高风险大。该单被决策层拦截，"
                "标记为【不执行】——仅记录意图，勾选执行与成交确认都会跳过它。"
            )
        else:
            txt = (
                "当日跌停：跌停板上的卖单几乎无法成交（排队排不到）。该单被决策层拦截，"
                "标记为【不执行】；若明日仍持有，会按最新排名重新评估。"
            )
        sections.append({"title": "⛔ 拦截说明", "lines": [txt]})

    # ① 触发规则
    if base.startswith("RSI超买"):
        t_title = "① 触发规则：RSI 超买（择时卖出）"
        rsi_txt = f"{rsi:.0f}" if rsi is not None else "?"
        trig = (
            f"RSI（相对强弱指标，越高越过热）升至 {rsi_txt}，"
            f"≥ 超买线 {C.RSI_SELL_MIN:.0f} —— 短线过热，获利了结。"
        )
    elif base == "触布林上轨":
        t_title = "① 触发规则：触布林上轨（择时卖出）"
        bb_txt = f"布林位置 {bb_pos:.2f}" if bb_pos is not None else "收盘价"
        trig = f"{bb_txt}，逼近/突破布林上轨（20 日波动通道上沿）—— 短线超涨，获利了结。"
    elif base == "跌出ML持有带":
        t_title = "① 触发规则：跌出ML持有带（排名衰减卖出）"
        trig = (
            f"模型排名跌至{rank_txt}，" if rank_txt else "模型排名"
        ) + f"跌出持有带（持有要求：排名前 {hold_ratio:.0%} 以内）—— 排名衰减，触发卖出。"
    elif base == "再平衡:清仓出池":
        t_title = "① 触发规则：再平衡清仓（跌出目标池）"
        trig = "满仓再平衡口径下，该股已不在当日目标池 —— 清仓。"
    elif base == "再平衡:减仓":
        t_title = "① 触发规则：再平衡减仓（超配回落）"
        trig = "仍在目标池，但当前持仓超过目标权重 —— 减仓至目标。"
    elif base == "再平衡:建仓":
        t_title = "① 触发规则：再平衡建仓（进入目标池）"
        trig = f"进入当日目标池{('（' + rank_txt + '）') if rank_txt else ''}，按目标权重新建仓。"
    elif base == "再平衡:加仓":
        t_title = "① 触发规则：再平衡加仓（低配补足）"
        trig = "在目标池内且当前低配 —— 加仓至目标权重。"
    elif base == "进入ML候选池":
        t_title = "① 触发规则：进入ML候选池（排名驱动）"
        trig = (
            f"模型排名{rank_txt}，" if rank_txt else "模型排名"
        ) + f"进入候选池（买入资格线：前 {topk:.0%}）。"
    elif base == "布林下轨回踩":
        t_title = "① 触发规则：布林下轨回踩（排名+择时）"
        bb_txt = f"布林位置 {bb_pos:.2f}" if bb_pos is not None else "股价"
        trig = (
            f"模型排名{rank_txt}进入候选池，且{bb_txt}、回踩至布林下轨附近"
            if rank_txt
            else f"{bb_txt}，回踩至布林下轨附近"
        ) + "（20 日波动通道下沿）—— 择时买点。"
    elif base == "跌出横截面:清退":
        t_title = "① 触发规则：跌出横截面（清退）"
        trig = (
            "当日决策样本中查不到该股（停牌或被剔除出样本）—— 按最近可得价发清退单；"
            "明日仍无法成交则继续持有等待。"
        )
    else:
        t_title = "① 触发规则"
        trig = base or "—"
    sections.append({"title": t_title, "lines": [trig]})

    # ② 择时检查（指标在本次决策中的角色）
    timing = []
    if rsi is not None:
        if rsi >= C.RSI_SELL_MIN:
            rsi_note = "超买区"
        elif rsi > C.RSI_BUY_MAX:
            rsi_note = "偏热"
        elif rsi <= 30:
            rsi_note = "超卖区"
        else:
            rsi_note = "中性，未超买"
        timing.append(f"RSI（相对强弱，>70 偏热、>80 超买）{rsi:.1f} —— {rsi_note}")
    if bb_pos is not None:
        if bb_pos >= 0.8:
            bb_note = "贴近上轨"
        elif bb_pos <= 0.2:
            bb_note = "贴近下轨"
        else:
            bb_note = "带内运行"
        timing.append(f"布林位置（0=下轨，1=上轨）{bb_pos:.2f} —— {bb_note}")
    if timing:
        if side == "sell" and base in ("跌出ML持有带", "再平衡:清仓出池", "再平衡:减仓"):
            timing.append("指标未触发卖出 —— 本单是排名/再平衡驱动，非择时卖出。")
        elif side == "buy" and base == "布林下轨回踩":
            timing.append("择时门槛通过（RSI 未过热、价格回踩至下轨可买区）。")
        elif side == "buy" and (base.startswith("再平衡") or base == "进入ML候选池"):
            timing.append("本次为满仓再平衡口径（未启用指标择时门控）。")
        sections.append({"title": "② 择时检查", "lines": timing})

    # ③ 模型明细（各子模型分别打分 → 融合排序分）
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
        prev = _as_float(pred.get("prev_score"))
        if prev is not None:
            diff = pred["score"] - prev
            trend = "略升" if diff > 1e-4 else ("略降" if diff < -1e-4 else "持平")
            score_txt += f"（前一日 {prev:.3f}，{trend}）"
        lines.append((f"权重 {w_txt} → " if w_txt else "") + score_txt)
        sections.append({"title": "③ 模型明细（各子模型分别打分）", "lines": lines})

    # ④ 订单参数
    hold = (ctx.get("holdings") or {}).get(sym)
    side_zh = "卖出" if side == "sell" else "买入"
    whole = "（全部持仓）" if side == "sell" and hold and hold.get("shares") == shares else ""
    price_mode = str(params.get("order_price_mode", C.ORDER_PRICE_MODE))
    if limit is not None and ref is not None and abs(limit - ref) < 1e-6:
        px_txt = f"市价单（次日开盘价撮合，参考收盘 {ref:.2f}）"
    elif limit is not None:
        if price_mode == "limit_bb":
            rail = "布林上轨" if side == "sell" else "布林下轨"
            cmp_txt = "成交价不会低于限价" if side == "sell" else "成交价不会高于限价"
            px_txt = f"限价 {limit:.3f}（挂在{rail}，{cmp_txt}"
            px_txt += f"；参考收盘 {ref:.2f}）" if ref is not None else "）"
        else:
            px_txt = f"限价 {limit:.3f}"
            px_txt += f"（参考收盘 {ref:.2f}）" if ref is not None else ""
    else:
        px_txt = ""
    amt = shares * (limit or ref) if shares and (limit or ref) else 0
    amt_txt = f"，约 ¥{amt:,.0f}" if amt else ""
    line = f"{side_zh} {shares:,} 股{whole}{(' @ ' + px_txt) if px_txt else ''}{amt_txt}"
    sections.append({"title": "④ 订单参数", "lines": [line]})

    # ⑤ 持仓信息（建仓日/成本/浮盈）
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
            sections.append({"title": "⑤ 持仓信息", "lines": [" · ".join(parts)]})
    return sections


def _order_explain_panel(row, ctx: dict) -> html.Details:
    """订单行末尾的「分析▾」展开面板。

    用纯 html.Details（无回调）：执行单卡片不在 td-poll 心跳的输出范围内
    （心跳只刷新出单状态区与流程卡四步状态灯），展开态不会被服务端重渲染重置。
    """
    body = []
    for sec in _order_explain(row, ctx):
        body.append(html.Div(sec["title"], className="td-why-sec"))
        body.extend(html.Div(ln, className="td-why-line") for ln in sec["lines"])
    return html.Details(
        [
            html.Summary("分析 ▸", className="td-why-summary"),
            html.Div(body, className="td-why-panel"),
        ],
        className="td-why",
    )


def _row_style(blocked: bool, done: bool, rejected: bool) -> dict:
    """执行单行样式：初渲染与交互回调共用，保证两处一致。"""
    style = {
        "padding": "4px 6px",
        "borderBottom": f"1px solid {PALETTE['grid']}",
        "fontSize": "12.5px",
    }
    if blocked:
        style.update({"opacity": "0.45", "backgroundColor": "#f3f4f6"})
    elif rejected:
        style.update({"opacity": "0.65", "backgroundColor": "#fef9c3"})
    elif done:
        style.update({"opacity": "0.45", "textDecoration": "line-through"})
    return style


def _order_row(
    row, blocked: bool, done: bool = False, explain_ctx: dict | None = None, rejected: bool = False
) -> html.Div:
    """执行单一行：执行/拒绝两个大按钮（非拦截，回读持久化状态）+ 代码/名称/股数/价/金额/原因。"""
    px = _order_price(row._asdict() if hasattr(row, "_asdict") else row)
    amt = float(getattr(row, "shares", 0) or 0) * px
    amt_txt = money(amt) if np.isfinite(amt) else "—"
    px_txt = f"{px:.2f}" if np.isfinite(px) else "—"
    reason = str(getattr(row, "reason", "") or "")
    name = str(getattr(row, "name", "") or "")
    rid = f"{row.side}-{row.symbol}"
    style = _row_style(blocked, done, rejected)
    if blocked:
        head = html.Span("🚫 ", title="决策层拦截")
    else:
        head = html.Span(
            [
                dbc.Button(
                    "✓ 已执行" if done else "执行",
                    id={"type": "td-done", "index": rid},
                    size="sm",
                    color="success",
                    outline=not done,
                    className="me-1 fw-bold",
                    style={"minWidth": "76px", "fontSize": "12.5px"},
                    title="已按此单下单则点「执行」；再点一次取消标记",
                ),
                dbc.Button(
                    "✗ 已拒绝" if rejected else "拒绝",
                    id={"type": "td-reject", "index": rid},
                    size="sm",
                    color="danger" if rejected else "secondary",
                    outline=not rejected,
                    className="me-2 fw-bold",
                    style={"minWidth": "76px", "fontSize": "12.5px"},
                    title="本单不执行：成交确认与目标仓位均跳过；可随时撤销",
                ),
            ],
            style={"display": "inline-flex", "verticalAlign": "middle"},
        )
    return html.Div(
        [
            head,
            html.Span(f"{row.symbol}", style={"fontFamily": "monospace", "fontWeight": 600}),
            html.Span(f" {name} ", style={"color": PALETTE["muted"]}),
            html.Span(f"{int(getattr(row, 'shares', 0))}股 @ {px_txt} ≈ {amt_txt}"),
            html.Span(
                f"（{reason}）" if reason else "",
                style={"color": PALETTE["muted"], "fontSize": "11.5px"},
            ),
            _order_explain_panel(row, explain_ctx),
        ],
        id={"type": "td-row", "index": rid},
        style=style,
    )


def _card_d(
    orders: pd.DataFrame, exec_state: dict | None = None, explain_ctx: dict | None = None
) -> dbc.Card:
    """订单执行单：卖单区（红）在上，买单区（绿）在下；勾选/拒绝状态从 exec_state 回读；
    每行末尾「分析▾」可展开该单的决策链白话解释（explain_ctx 提供外部数据）。"""
    exec_state = exec_state or {}
    if orders.empty:
        return _card(
            "订单执行单",
            _empty_hint(["无订单。点「今日流程」卡①「立即生成」一键创建出单会话。"]),
        )
    blk = _blocked_mask(orders)
    df = orders.copy()
    df["blkflag"] = blk.to_numpy()
    sells = df[df["side"] == "sell"]
    buys = df[df["side"] == "buy"]

    def _section(title, color, sub):
        rows = [
            html.Div(
                title,
                style={
                    "fontWeight": 700,
                    "color": color,
                    "borderLeft": f"3px solid {color}",
                    "paddingLeft": "8px",
                    "marginBottom": "4px",
                    "marginTop": "8px",
                },
            )
        ]
        if sub.empty:
            rows.append(html.Div("（无）", style={"color": PALETTE["muted"], "fontSize": "12px"}))
        else:
            for r in sub.itertuples(index=False):
                rid = f"{r.side}-{r.symbol}"
                rows.append(
                    _order_row(
                        r,
                        bool(r.blkflag),
                        exec_state.get(rid) is True,
                        explain_ctx,
                        rejected=exec_state.get(rid) == "rejected",
                    )
                )
        return rows

    body = _section(f"卖单区（先卖腾现金，{len(sells)} 笔）", PALETTE["down"], sells)
    body += _section(f"买单区（后买，{len(buys)} 笔）", PALETTE["up"], buys)
    body.append(
        html.Div(
            "点「执行」或「拒绝」均自动保存（outputs/execution_state_<日期>.json），"
            "刷新不丢；拦截单（涨跌停等）与已拒绝单不执行，成交确认时跳过。",
            style={"color": PALETTE["muted"], "fontSize": "11.5px", "marginTop": "8px"},
        )
    )
    return _card("订单执行单", body)


def _card_e(tvc: pd.DataFrame) -> dbc.Card:
    """目标仓位 vs 当前持仓。"""
    if tvc.empty or tvc["cur_w"].isna().all():
        return _card("目标仓位 vs 当前", _empty_hint(["无持仓且无订单，或净值未知。"]))
    fig = base_fig("持仓权重：当前 vs 订单推导目标", height=300)
    fig.add_trace(
        go.Bar(x=tvc["symbol"], y=tvc["cur_w"], name="当前", marker_color=PALETTE["faint"])
    )
    fig.add_trace(
        go.Bar(x=tvc["symbol"], y=tvc["tgt_w"], name="目标", marker_color=PALETTE["strat"])
    )
    fig.update_layout(barmode="group", margin=dict(l=50, r=16, t=40, b=40))
    fig.update_yaxes(title="权重", tickformat=".0%")
    return _card("目标仓位 vs 当前", dcc.Graph(figure=fig))


def _card_f(daily: pd.DataFrame, report: dict | None) -> dbc.Card:
    """净值与偏差。"""
    if daily.empty and not report:
        return _card(
            "净值与偏差",
            _empty_hint(["跟踪在首次「确认成交」（今日流程卡③）后自动开始积累。"]),
        )
    body = []
    if not daily.empty and "live_total_value" in daily.columns:
        d = daily.dropna(subset=["live_total_value"])
        fig = base_fig("实盘净值 vs 全市场等权基准（归一）", height=300)
        if not d.empty:
            x = pd.to_datetime(d["date"])
            nav = d["live_total_value"] / d["live_total_value"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=x, y=nav, name="实盘净值", line=dict(color=PALETTE["strat"], width=1.6)
                )
            )
            if "bench_ret_1d" in d.columns:
                b = (1 + d["bench_ret_1d"].fillna(0.0)).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=b,
                        name="等权基准",
                        line=dict(color=PALETTE["bench"], width=1.2, dash="dash"),
                    )
                )
        else:
            fig = empty_fig("净值列全为空（account.csv 未维护？）", height=300)
        fig.update_layout(margin=dict(l=50, r=16, t=40, b=40))
        body.append(dcc.Graph(figure=fig))
    if report:
        live = report.get("live") or {}
        div = report.get("divergence") or {}
        stats = []
        if live.get("ic_mean_w") is not None:
            stats.append(f"近{live.get('ic_window', 20)}日活体 IC 均值 {live['ic_mean_w']:+.4f}")
        if div.get("sharpe_gap") is not None:
            stats.append(f"Sharpe 偏差 {div['sharpe_gap']:+.2f}")
        if div.get("ret_gap_vs_bench") is not None:
            stats.append(f"近{live.get('window', 20)}日收益差 {div['ret_gap_vs_bench']:+.1%}")
        if stats:
            body.append(
                html.Div(" · ".join(stats), style={"fontSize": "12px", "color": PALETTE["muted"]})
            )
        alerts = report.get("alerts") or []
        if alerts:
            body.append(
                html.Div(
                    [
                        dbc.Alert(
                            a,
                            color="warning",
                            className="py-1 px-2 my-1",
                            style={"fontSize": "12px"},
                        )
                        for a in alerts
                    ],
                    className="mt-2",
                )
            )
        else:
            body.append(
                html.Div(
                    "无活跃告警。",
                    style={"fontSize": "12px", "color": PALETTE["up"], "marginTop": "6px"},
                )
            )
    return _card("净值与偏差", body)


def _card_g(holdings_df: pd.DataFrame, last_px: dict, last_date: str | None) -> dbc.Card:
    """持仓明细表（代码可点击跳 /market?symbol=<代码>）。"""
    if holdings_df.empty:
        return _card("持仓明细", _empty_hint(["当前空仓。"]))
    today = date.today()
    rows = []
    for r in holdings_df.itertuples(index=False):
        sh = float(getattr(r, "shares", 0) or 0)
        if sh <= 0:
            continue
        ref = getattr(r, "ref_close", None)
        ref = float(ref) if ref is not None and ref == ref else None
        px = last_px.get(r.symbol)
        pnl = (px - ref) * sh if (px is not None and ref) else None
        pnl_pct = (px / ref - 1.0) if (px is not None and ref) else None
        ed = getattr(r, "entry_date", None)
        try:
            hold_days = (
                (today - pd.Timestamp(str(ed)).date()).days if ed and str(ed) != "nan" else None
            )
        except Exception:
            hold_days = None
        pnl_color = None if pnl is None else (PALETTE["up"] if pnl >= 0 else PALETTE["down"])
        rows.append(
            html.Tr(
                [
                    html.Td(
                        dcc.Link(
                            r.symbol,
                            href=f"/market?symbol={r.symbol}",
                            style={"fontFamily": "monospace"},
                        )
                    ),
                    html.Td(f"{int(sh)}"),
                    html.Td(str(ed)[:10] if ed and str(ed) != "nan" else "—"),
                    html.Td(f"{ref:.2f}" if ref else "—"),
                    html.Td(f"{px:.2f}" if px else "—"),
                    html.Td(
                        f"{money(pnl)} ({pnl_pct:+.1%})" if pnl is not None else "—",
                        style={"color": pnl_color} if pnl_color else {},
                    ),
                    html.Td(f"{hold_days}天" if hold_days is not None else "—"),
                ]
            )
        )
    if not rows:
        return _card("持仓明细", _empty_hint(["当前空仓。"]))
    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [html.Th(h) for h in ["代码", "股数", "建仓日", "成本", "现价", "浮盈", "持有"]]
                )
            ),
            html.Tbody(rows),
        ],
        bordered=False,
        hover=True,
        size="sm",
        style={"fontSize": "12px"},
    )
    sub = f"现价取自 {last_date} 收盘" if last_date else "现价不可得"
    return _card(
        "持仓明细",
        [table, html.Div(sub, style={"fontSize": "11px", "color": PALETTE["muted"]})],
    )


def _card_h() -> dbc.Card:
    """执行闭环说明（默认折叠，按钮展开 5 步操作说明）。"""
    steps = [
        ("① 今晚收盘后", "打开本页看风控灯 + 偏差告警；异常先查因"),
        ("② 生成订单", "「今日流程」卡①点「立即生成」，持仓自动预填，确认创建后分钟级出单"),
        (
            "③ 次日 9:25",
            "按订单执行单下单：先卖后买，逐条点「执行」，不想做的单点「拒绝」（均自动持久化）",
        ),
        ("④ 成交后晚间", "「今日流程」卡③点「确认成交」，按非拦截单全部成交写回 holdings.csv"),
        ("⑤ 维护现金", "出入金/分红后在「今日流程」卡④输入现金保存，净值曲线自动延长"),
    ]
    return _card(
        "执行闭环",
        [
            dbc.Button(
                "第一次用？展开 5 步操作说明",
                id="td-loop-toggle",
                color="link",
                size="sm",
                className="p-0",
                style={"fontSize": "12.5px", "textDecoration": "none"},
            ),
            dbc.Collapse(
                html.Div(
                    [
                        html.Ol(
                            [
                                html.Li(
                                    [html.Strong(t + "："), s],
                                    style={"marginBottom": "4px", "fontSize": "12.5px"},
                                )
                                for t, s in steps
                            ],
                            className="mb-0 mt-2",
                            style={"paddingLeft": "18px"},
                        ),
                        html.Div(
                            [
                                "等价 CLI：",
                                html.Code("./daily_run.sh"),
                                "（出单）/ ",
                                html.Code("./daily_run.sh commit"),
                                "（成交确认），可 cron 全自动。",
                            ],
                            style={
                                "fontSize": "11px",
                                "color": PALETTE["muted"],
                                "marginTop": "6px",
                            },
                        ),
                    ]
                ),
                id="td-loop-collapse",
                is_open=False,
            ),
        ],
    )


def _onboarding_card() -> dbc.Card:
    """三步开始引导（从未跑过实盘时置顶，设计文档第 8 节）。"""
    return dbc.Card(
        [
            dbc.CardHeader("三步开始实盘跟踪", style={"fontWeight": 600}),
            dbc.CardBody(
                html.Ol(
                    [
                        html.Li(
                            "在下方「今日流程 · ④ 现金维护」输入当前可用现金并保存（自动建账）"
                        ),
                        html.Li(
                            "点「今日流程 · ① 生成订单」的「立即生成」按钮，等待会话完成（分钟级）"
                        ),
                        html.Li(
                            "明天开盘按执行单下单，逐条点「执行」/「拒绝」后点「③ 成交确认」写回持仓"
                        ),
                    ],
                    className="mb-0",
                ),
                style={"fontSize": "13px"},
            ),
        ],
        color="info",
        outline=True,
        style={"marginBottom": "14px"},
    )


# ---------------------------------------------------------------------------
# layout / register
# ---------------------------------------------------------------------------
def layout():
    orders, o_meta = dataio.latest_orders_with_meta()
    holds = dataio.holdings()
    acct = dataio.account()
    state = dataio.live_risk_state()
    report = dataio.tracking_report()
    daily = dataio.live_daily()

    prev_px, last_px, last_date = _close_maps()
    iv_status, iv_scale = _iv_scale_now()
    risk = compute_risk_layers(state, iv_scale=iv_scale, iv_status=iv_status)
    snap = account_snapshot(acct, holds, last_px, prev_px)
    exec_state = dataio.load_execution_state(_today_str())
    rej = _rejected_rids(exec_state)
    tvc_orders = orders
    if rej and not orders.empty:
        tvc_orders = orders.copy()
        tvc_orders["blocked"] = (
            _blocked_mask(tvc_orders) | _reject_mask(tvc_orders, rej)
        ).to_numpy()
    tvc = target_vs_current(tvc_orders, holds, last_px, snap["nav"] if snap else None)
    orders_date = o_meta.get("date")
    explain_ctx = _explain_context(orders, o_meta)
    flow = _compute_flow(acct)

    children = [
        page_header(
            "交易台 Trading Desk",
            "明天开盘我做什么？仓位多重？风控现在让我开多少仓？",
            flow_step=3,
        ),
    ]
    if orders.empty and holds.empty and acct.empty:
        children.append(_onboarding_card())
    children += [
        _daily_flow_card(flow),
        dbc.Row(
            [
                dbc.Col(_card_a(orders, risk, orders_date), md=4),
                dbc.Col(_card_b(risk), md=4),
                dbc.Col(html.Div(_card_c(snap), id="td-card-c-wrap"), md=4),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(_card_d(orders, exec_state, explain_ctx), md=8),
                dbc.Col(_card_e(tvc), md=4),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(_card_f(daily, report), md=8),
                dbc.Col(_card_g(holds, last_px, last_date), md=4),
            ],
            className="g-3",
        ),
        _card_h(),
        _gen_modal(snap),
        _commit_modal(),
        dcc.Store(id="td-exec-store", data=sorted(k for k, v in exec_state.items() if v is True)),
        dcc.Store(id="td-reject-store", data=sorted(rej)),
        dcc.Interval(id="td-poll", interval=_POLL_MS),
    ]
    return dbc.Container(children, fluid=True)


def _account_card_now() -> dbc.Card:
    """重读 account/holdings 重建账户快照卡（现金保存/成交确认后即时刷新）。"""
    prev_px, last_px, _ = _close_maps()
    snap = account_snapshot(dataio.account(), dataio.holdings(), last_px, prev_px)
    return _card_c(snap)


def _reject_mask(orders: pd.DataFrame, rej: set[str]) -> pd.Series:
    """订单 → 是否被拒绝（按 side-symbol 匹配执行状态中的 "rejected" 键）。"""
    if not rej or orders.empty:
        return pd.Series(False, index=orders.index)
    return pd.Series(
        [f"{r.side}-{r.symbol}" in rej for r in orders.itertuples(index=False)],
        index=orders.index,
    )


def _commit_preview() -> html.Div:
    """成交确认弹窗正文：订单来源/日期 + 将成交的买/卖笔数（二次确认防呆）。"""
    orders, meta = dataio.latest_orders_with_meta()
    if orders.empty or not _orders_fresh(meta.get("date")):
        return dbc.Alert(
            "无当前周期的订单，请先在「今日流程」卡①生成。",
            color="warning",
            className="py-1 px-2 mb-0",
            style={"fontSize": "12.5px"},
        )
    stats = order_amounts(orders)
    rej = _rejected_rids(dataio.load_execution_state(_today_str()))
    blk = _blocked_mask(orders)
    rej_live = _reject_mask(orders, rej) & ~blk
    n_rej = int(rej_live.sum())
    sell_rej = int((rej_live & (orders["side"] == "sell")).sum())
    n_sell = stats["n_sell"] - sell_rej
    n_buy = stats["n_buy"] - (n_rej - sell_rej)
    if n_sell + n_buy == 0:
        return dbc.Alert(
            "没有可确认成交的订单（非拦截单已全部拒绝）。如需成交请先撤销拒绝。",
            color="warning",
            className="py-1 px-2 mb-0",
            style={"fontSize": "12.5px"},
        )
    src = meta.get("source", "")
    src_txt = "CLI 产物" if src == "cli" else src
    rej_txt = f"拒绝单 {n_rej} 笔不执行。" if n_rej else ""
    return html.Div(
        [
            html.P(
                f"订单日期 {meta.get('date')}（{src_txt}）。"
                f"将假设 {n_sell + n_buy} 笔非拦截订单全部成交"
                f"（卖 {n_sell} 笔 / 买 {n_buy} 笔），写回 holdings.csv。",
                style={"fontSize": "13px"},
            ),
            html.P(
                f"拦截单 {stats['n_blocked']} 笔不执行。{rej_txt}"
                "写回后请按券商实际成交核对持仓，并维护下方④现金。",
                style={"fontSize": "12px", "color": PALETTE["muted"]},
            ),
        ]
    )


def _do_commit() -> int:
    """网页成交确认：最新订单（剔除拦截单与拒绝单）+ 当前持仓 → 共用 save_holdings 写回。"""
    orders, meta = dataio.latest_orders_with_meta()
    if orders.empty or not _orders_fresh(meta.get("date")):
        raise RuntimeError("无当前周期的订单，无法确认成交")
    orders = orders.copy()
    blk = _blocked_mask(orders)
    rej = _rejected_rids(dataio.load_execution_state(_today_str()))
    blk = blk | _reject_mask(orders, rej)
    if int((~blk).sum()) == 0:
        raise RuntimeError("没有可确认成交的订单（非拦截单已全部拒绝）")
    orders["blocked"] = blk.to_numpy()
    path = os.path.join(dataio.OUTPUTS, "holdings.csv")
    tbl = load_holdings_table(path)
    current = {str(s): int(sh) for s, sh in zip(tbl["symbol"], tbl["shares"])} if len(tbl) else {}
    save_holdings(path, orders, current, meta["date"])
    after = load_holdings_table(path)
    return int((pd.to_numeric(after["shares"], errors="coerce").fillna(0) > 0).sum())


def register(app):
    """回调：勾选划线+持久化、执行闭环折叠、出单弹窗、成交确认、现金保存、进度心跳。"""

    @app.callback(
        Output("td-loop-collapse", "is_open"),
        Input("td-loop-toggle", "n_clicks"),
        State("td-loop-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_loop(_n, is_open):
        return not is_open

    @app.callback(
        Output({"type": "td-row", "index": MATCH}, "style"),
        Output({"type": "td-done", "index": MATCH}, "children"),
        Output({"type": "td-done", "index": MATCH}, "outline"),
        Output({"type": "td-reject", "index": MATCH}, "children"),
        Output({"type": "td-reject", "index": MATCH}, "color"),
        Output({"type": "td-reject", "index": MATCH}, "outline"),
        Input("td-exec-store", "data"),
        Input("td-reject-store", "data"),
        State({"type": "td-done", "index": MATCH}, "id"),
        prevent_initial_call=True,
    )
    def _row_visual(done_store, rej_store, cid):
        """行外观：执行绿实底划线 / 拒绝红实底黄底；两个 store 任一变化即重刷所有行。"""
        done = cid["index"] in (done_store or [])
        rejected = cid["index"] in (rej_store or [])
        return (
            _row_style(False, done, rejected),
            "✓ 已执行" if done else "执行",
            not done,
            "✗ 已拒绝" if rejected else "拒绝",
            "danger" if rejected else "secondary",
            not rejected,
        )

    @app.callback(
        Output("td-exec-store", "data"),
        Output("td-reject-store", "data"),
        Input({"type": "td-done", "index": ALL}, "n_clicks"),
        Input({"type": "td-reject", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle_mark(done_clicks, rej_clicks):
        """执行/拒绝切换：互斥（执行覆盖拒绝、拒绝覆盖执行），写盘持久化后回写两个 store。"""
        trig = ctx.triggered_id
        clicks = (done_clicks or []) + (rej_clicks or [])
        if not isinstance(trig, dict) or not any(c for c in clicks if c):
            raise PreventUpdate
        rid = trig["index"]
        state = dataio.load_execution_state(_today_str())
        if trig["type"] == "td-done":
            if state.get(rid) is True:
                state.pop(rid, None)
            else:
                state[rid] = True
        elif state.get(rid) == "rejected":
            state.pop(rid, None)
        else:
            state[rid] = "rejected"
        dataio.save_execution_state(_today_str(), state)
        done = sorted(k for k, v in state.items() if v is True)
        return done, sorted(_rejected_rids(state))

    @app.callback(
        Output("td-commit-modal", "is_open"),
        Output("td-commit-body", "children"),
        Output("td-card-c-wrap", "children", allow_duplicate=True),
        Input("td-commit-open", "n_clicks"),
        Input("td-commit-cancel", "n_clicks"),
        Input("td-commit-run", "n_clicks"),
        prevent_initial_call=True,
    )
    def _commit_cb(_open, _cancel, _run):
        trig = ctx.triggered_id
        if trig == "td-commit-open":
            return True, _commit_preview(), no_update
        if trig == "td-commit-cancel":
            return False, no_update, no_update
        try:
            n_hold = _do_commit()
        except Exception as e:
            return (
                True,
                dbc.Alert(
                    f"写回失败：{type(e).__name__}: {e}",
                    color="danger",
                    className="py-1 px-2 mb-0",
                    style={"fontSize": "12px"},
                ),
                no_update,
            )
        return (
            True,
            dbc.Alert(
                f"已写回 holdings.csv：当前持仓 {n_hold} 只（拦截单与拒绝单已跳过，其余假设全部成交）。"
                "请按券商实际成交核对；持仓明细/目标仓位卡刷新页面后更新。",
                color="success",
                className="py-1 px-2 mb-0",
                style={"fontSize": "12.5px"},
            ),
            _account_card_now(),
        )

    @app.callback(
        Output("td-cash-msg", "children"),
        Output("td-card-c-wrap", "children", allow_duplicate=True),
        Input("td-cash-save", "n_clicks"),
        State("td-cash-input", "value"),
        prevent_initial_call=True,
    )
    def _save_cash(_n, value):
        try:
            cash = float(value)
        except (TypeError, ValueError):
            cash = float("nan")
        if not np.isfinite(cash) or cash < 0:
            return (
                html.Span("请输入非负数字", style={"color": PALETTE["down"]}),
                no_update,
            )
        try:
            dataio.append_account_cash(_today_iso(), cash)
        except Exception as e:
            return (
                html.Span(f"保存失败：{e}", style={"color": PALETTE["down"]}),
                no_update,
            )
        return (
            html.Span(f"已保存 {money(cash)}", style={"color": PALETTE["up"]}),
            _account_card_now(),
        )

    @app.callback(
        Output("td-gen-modal", "is_open"),
        Output("td-gen-alert", "children"),
        Output("td-gen-status", "children", allow_duplicate=True),
        Output("td-gen-holdings-table", "data"),
        Output("td-gen-prefill-note", "children"),
        Input("td-gen-open", "n_clicks"),
        Input("td-flow-gen", "n_clicks"),
        Input("td-gen-cancel", "n_clicks"),
        Input("td-gen-submit", "n_clicks"),
        Input("td-gen-addrow", "n_clicks"),
        State("td-gen-label", "value"),
        State("td-gen-capital", "value"),
        State("td-gen-holdings-table", "data"),
        prevent_initial_call=True,
    )
    def _gen_modal_cb(_open, _flow_open, _cancel, _submit, _add, label, capital, rows):
        trig = ctx.triggered_id
        if trig in ("td-gen-open", "td-flow-gen"):
            data, note = _prefill_holding_rows()
            return True, no_update, no_update, data, note
        if trig == "td-gen-addrow":
            return (
                True,
                no_update,
                no_update,
                list(rows or []) + [{"symbol": "", "shares": None}],
                no_update,
            )
        if trig == "td-gen-cancel":
            return False, no_update, no_update, no_update, no_update
        # submit：表格行序列化校验 → 创建 nextday 会话 → 尝试立即调度
        text, err = rows_to_holdings_text(rows)
        if err:
            params, lbl = None, ""
        else:
            params, lbl, err = _build_nextday_params(capital, text, label)
        if err:
            return (
                True,
                dbc.Alert(
                    err, color="danger", className="py-1 px-2 mb-0", style={"fontSize": "12px"}
                ),
                no_update,
                no_update,
                no_update,
            )
        try:
            sid = scheduler.create("nextday", lbl, params)
            scheduler.tick()
        except Exception as e:
            return (
                True,
                dbc.Alert(
                    f"创建会话失败：{type(e).__name__}: {e}",
                    color="danger",
                    className="py-1 px-2 mb-0",
                    style={"fontSize": "12px"},
                ),
                no_update,
                no_update,
                no_update,
            )
        notices = []
        if _cli_orders_today():
            notices.append(
                dbc.Alert(
                    "今日已有订单文件，新会话产物落在会话目录，不覆盖该文件。",
                    color="warning",
                    className="py-1 px-2 mb-1",
                    style={"fontSize": "12px"},
                )
            )
        notices.append(
            dbc.Alert(
                ["已创建会话 ", html.Code(sid), "，正在调度。", _detail_link(sid)],
                color="success",
                className="py-1 px-2 mb-0",
                style={"fontSize": "12px"},
            )
        )
        return False, no_update, notices, no_update, no_update

    @app.callback(
        Output("td-gen-status", "children"),
        Output("td-flow-s1-status", "children"),
        Output("td-flow-s2-status", "children"),
        Output("td-flow-s3-status", "children"),
        Output("td-flow-s4-status", "children"),
        Output("td-commit-open", "disabled"),
        Input("td-poll", "n_intervals"),
    )
    def _poll_gen_status(_n):
        # 惯例：轮询回调顺带推进调度（同 session_detail / model_lab）
        scheduler.tick()
        sessions = scheduler.list_sessions()
        active = _active_nextday(sessions)
        if active is not None:
            gen_status = _render_session_status(active)
        else:
            gen_status = _render_idle_status(
                _latest_finished_nextday(sessions), _cli_orders_today()
            )
        flow = _compute_flow(dataio.account())
        lamps = [
            [_lamp(lamp), text] for lamp, text in (flow["s1"], flow["s2"], flow["s3"], flow["s4"])
        ]
        return gen_status, *lamps, not flow["can_commit"]
