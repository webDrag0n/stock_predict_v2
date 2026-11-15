"""成交确认写回：CLI（predict_tomorrow --commit / daily_run.sh commit）与
webapp 交易台「确认成交」共用的唯一实现，保证两条路径行为逐位一致。
CLI 提供 --fills 时按券商实际成交写回；未提供时保留订单全部成交假设。

状态文件：
  outputs/holdings.csv —— 当前持仓（symbol,shares,entry_date,ref_close）
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def load_rejected_rids(outputs_dir: str, day: str) -> set[str]:
    """webapp 交易台「拒绝」的执行单（outputs/execution_state_<day>.json 中值为
    "rejected" 的 side-symbol 键）；缺失/损坏 → 空集。CLI --commit 与网页成交确认共用。"""
    try:
        with open(os.path.join(outputs_dir, f"execution_state_{day}.json"), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    if not isinstance(data, dict):
        return set()
    return {k for k, v in data.items() if v == "rejected"}


def load_holdings_table(path: str) -> pd.DataFrame:
    """读取完整持仓表（symbol/shares/entry_date/ref_close），不存在 → 空表。"""
    cols = ["symbol", "shares", "entry_date", "ref_close"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, dtype={"symbol": str})
        # 兼容旧版 holdings.csv（中文列）
        if "代码" in df.columns and "股数" in df.columns:
            df = df.rename(columns={"代码": "symbol", "股数": "shares"})
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]
    except Exception:
        return pd.DataFrame(columns=cols)


def load_fills_csv(path: str) -> pd.DataFrame:
    required = ["symbol", "side", "shares", "price"]
    frame = pd.read_csv(path, dtype={"symbol": str})
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"成交单缺少必需列：{', '.join(missing)}")
    frame = frame[required].copy()
    frame["side"] = frame["side"].astype(str).str.strip().str.lower()
    invalid_side = ~frame["side"].isin(["buy", "sell"])
    if invalid_side.any():
        raise ValueError("成交单 side 只能为 buy/sell")
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    if frame[["shares", "price"]].isna().any().any():
        raise ValueError("成交单 shares/price 必须为数字")
    if (frame["shares"] <= 0).any() or (frame["price"] <= 0).any():
        raise ValueError("成交单 shares/price 必须大于 0")
    frame["notional"] = frame["shares"] * frame["price"]
    grouped = frame.groupby(["symbol", "side"], sort=False, as_index=False).agg(
        shares=("shares", "sum"), notional=("notional", "sum")
    )
    grouped["price"] = grouped["notional"] / grouped["shares"]
    grouped["shares"] = grouped["shares"].astype(int)
    return grouped[["symbol", "side", "shares", "price"]]


def _save_holdings_with_price(
    path: str,
    transactions: pd.DataFrame,
    current: dict[str, int],
    latest_date,
    price_col: str,
) -> None:
    old = load_holdings_table(path)
    old_entry, old_ref = {}, {}
    for r in old.itertuples(index=False):
        old_entry[r.symbol] = None if pd.isna(r.entry_date) else str(r.entry_date)
        old_ref[r.symbol] = None if pd.isna(r.ref_close) else float(r.ref_close)

    target = dict(current)
    bought: set[str] = set()
    for transaction in transactions.itertuples(index=False):
        if getattr(transaction, "blocked", False):
            continue
        symbol = transaction.symbol
        if transaction.side == "buy":
            target[symbol] = target.get(symbol, 0) + int(transaction.shares)
            bought.add(symbol)
        elif transaction.side == "sell":
            target[symbol] = max(0, target.get(symbol, 0) - int(transaction.shares))

    dstr = str(pd.Timestamp(latest_date).date())
    rows = []
    for symbol, shares in target.items():
        if shares <= 0:
            continue
        ref = old_ref.get(symbol)
        sub = transactions[transactions["symbol"] == symbol]
        if len(sub) and price_col in transactions.columns:
            ref = float(sub.iloc[0][price_col])
        rows.append(
            {
                "symbol": symbol,
                "shares": int(shares),
                "entry_date": dstr if symbol in bought else (old_entry.get(symbol) or dstr),
                "ref_close": ref,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_holdings(path: str, orders: pd.DataFrame, current: dict[str, int], latest_date) -> None:
    """按订单差额把目标持仓写回 holdings.csv（commit 假设非拦截单全部成交）。

    entry_date 语义：只有【买入成交】（新建/加仓）的持仓刷新为当日；卖出减仓
    与未交易的持仓保留原 entry_date（旧版每次 commit 全部重置，会丢失真实
    建仓日，已修复）。
    """
    _save_holdings_with_price(path, orders, current, latest_date, "ref_close")


def save_holdings_from_fills(
    path: str, fills: pd.DataFrame, current: dict[str, int], latest_date
) -> None:
    """只按实际成交写回；fills 中缺失的订单或股票保持原持仓不变。"""
    _save_holdings_with_price(path, fills, current, latest_date, "price")
