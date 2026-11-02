"""成交确认写回：CLI（predict_tomorrow --commit / daily_run.sh commit）与
webapp 交易台「确认成交」共用的唯一实现，保证两条路径行为逐位一致。

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


def save_holdings(path: str, orders: pd.DataFrame, current: dict[str, int], latest_date) -> None:
    """按订单差额把目标持仓写回 holdings.csv（commit 假设非拦截单全部成交）。

    entry_date 语义：只有【买入成交】（新建/加仓）的持仓刷新为当日；卖出减仓
    与未交易的持仓保留原 entry_date（旧版每次 commit 全部重置，会丢失真实
    建仓日，已修复）。
    """
    old = load_holdings_table(path)
    old_entry, old_ref = {}, {}
    for r in old.itertuples(index=False):
        old_entry[r.symbol] = None if pd.isna(r.entry_date) else str(r.entry_date)
        old_ref[r.symbol] = None if pd.isna(r.ref_close) else float(r.ref_close)

    target = dict(current)
    bought: set[str] = set()
    for o in orders.itertuples(index=False):
        if getattr(o, "blocked", False):
            continue
        s = o.symbol
        if o.side == "buy":
            target[s] = target.get(s, 0) + int(o.shares)
            bought.add(s)
        elif o.side == "sell":
            target[s] = max(0, target.get(s, 0) - int(o.shares))

    dstr = str(pd.Timestamp(latest_date).date())
    rows = []
    for s, sh in target.items():
        if sh <= 0:
            continue
        ref = old_ref.get(s)
        sub = orders[orders["symbol"] == s]
        if len(sub) and "ref_close" in orders.columns:
            ref = float(sub.iloc[0]["ref_close"])
        rows.append(
            {
                "symbol": s,
                "shares": int(sh),
                # 未买入的持仓保留原建仓日；查不到（新文件外的来源）才用当日
                "entry_date": dstr if s in bought else (old_entry.get(s) or dstr),
                "ref_close": ref,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
