"""回测页的纯逻辑：窗口指标复算、可信度判定、多次实验横向对比、holdout/DSR 展示数据。

不依赖 dash，便于单测。

为什么要复算窗口 Sharpe：
  增量引擎（engine=incremental）是一次连续重放，窗口只是把净值曲线切段。T17 修复前，
  窗口指标用的基准资金错了，写进 window_metrics.parquet 的 Sharpe 与 meta.json 的
  sharpe_pos_ratio 都偏高（同一份数据：存档 8/9 窗口为正，按曲线复算只有 5/9）。
  修复只对之后的新运行生效，旧产物不会自己更新。所以页面对增量引擎的产物一律按
  oos_curve.parquet 复算，并在与存档不一致时明确标出，而不是照单全收。
  windowed 引擎每个窗口独立重放（各自从空仓起算），曲线拼接不能还原窗口指标，故沿用存档值。

oos_curve 的 equity 列是「累计净 PnL / 初始资金」，总资产 = 1 + equity（以初始资金为 1）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

POS_WINDOW_MIN = 0.6  # Sharpe>0 窗口占比低于它判红（与旧前端同一门槛）
POS_IC_MIN = 0.5
SHARPE_MATCH_TOL = 0.05  # 复算与存档的窗口 Sharpe 相差不超过它视为一致


def number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def local_window_sharpe(curve: pd.DataFrame, test_window: int) -> list[float]:
    """按 test_window 切段，每段内的年化 Sharpe。段首收益以上一段末的总资产为基准。"""
    if curve is None or curve.empty or "equity" not in curve.columns or test_window <= 0:
        return []
    tv = 1.0 + curve["equity"].to_numpy(dtype=float)
    prev = np.concatenate([[1.0], tv[:-1]])
    ret = tv / prev - 1.0
    out: list[float] = []
    for i in range(0, len(ret), test_window):
        r = ret[i : i + test_window]
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        out.append(float(r.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else float("nan"))
    return out


def is_incremental(meta: dict | None) -> bool:
    return ((meta or {}).get("config") or {}).get("engine") == "incremental"


def window_table(meta: dict | None, windows: pd.DataFrame, curve: pd.DataFrame) -> dict:
    """窗口表：存档列 + （增量引擎时）复算列与一致性判定。

    返回 {"df", "recomputed", "stale", "pos_stored", "pos_used", "n"}：
      pos_used = 页面应当采用的「Sharpe 为正的窗口数」（增量引擎用复算值）。
    """
    n = 0 if windows is None else len(windows)
    out = {
        "df": windows,
        "recomputed": False,
        "stale": False,
        "pos_stored": None,
        "pos_used": None,
        "n": n,
    }
    if windows is None or windows.empty:
        return out
    df = windows.copy()
    sharpe_col = "Sharpe" if "Sharpe" in df.columns else None
    if sharpe_col:
        out["pos_stored"] = int((df[sharpe_col] > 0).sum())
        out["pos_used"] = out["pos_stored"]
    if is_incremental(meta) and sharpe_col:
        tw = int(((meta or {}).get("config") or {}).get("test_window") or 0)
        mine = local_window_sharpe(curve, tw)
        if len(mine) == len(df):
            df["Sharpe_复算"] = mine
            out["recomputed"] = True
            diff = (df["Sharpe_复算"] - df[sharpe_col]).abs()
            out["stale"] = bool((diff > SHARPE_MATCH_TOL).any())
            out["pos_used"] = int((df["Sharpe_复算"] > 0).sum())
    out["df"] = df
    return out


def trust_badges(meta: dict | None, pos_used: int | None, n: int) -> list[tuple[str, str]]:
    """[(文字, 级别)]，级别 ok|bad。Sharpe>0 窗口占比用复算值（若有），PredIC 用 meta。"""
    badges: list[tuple[str, str]] = []
    if pos_used is not None and n:
        r = pos_used / n
        badges.append(
            (f"Sharpe>0 窗口 {pos_used}/{n}（{r:.0%}）", "ok" if r >= POS_WINDOW_MIN else "bad")
        )
    ppr = (meta or {}).get("predic_pos_ratio")
    if number(ppr):
        badges.append((f"PredIC>0 窗口 {ppr:.0%}", "ok" if ppr >= POS_IC_MIN else "bad"))
    nw = (meta or {}).get("nw_t")
    if number(nw):
        badges.append((f"NW-t {nw:.2f}", "ok" if abs(nw) >= 2.0 else "bad"))
    return badges


def holdout_rows(meta: dict | None) -> list[dict]:
    """HOLDOUT 前后对照行。meta 没有这两段（旧运行）→ 空列表。"""
    pre, ho = (meta or {}).get("pre_holdout"), (meta or {}).get("holdout")
    if not isinstance(pre, dict) or not isinstance(ho, dict):
        return []
    return [
        {"seg": "选型段（HOLDOUT 之前）", **pre},
        {"seg": "HOLDOUT（只用于最终报告）", **ho},
    ]


def dsr_note(meta: dict | None) -> tuple[str, str] | None:
    """(文字, 级别)。DSR = 考虑多重试验后真实 Sharpe>0 的概率。没有该字段 → None。"""
    m = meta or {}
    d, n = m.get("deflated_sharpe"), m.get("n_trials")
    if not number(d):
        return None
    tail = f"（已累计 {int(n)} 次试验）" if number(n) else ""
    return f"Deflated Sharpe {d:.2f}：真实 Sharpe>0 的概率{tail}", "ok" if d >= 0.95 else "bad"


# ----------------------------------------------------------------------------- 多次实验对比
def compare_rows(runs: list[dict]) -> list[dict]:
    """experiment_runs() → 每行一个实验。正窗口数对增量引擎用复算值。"""
    rows = []
    for r in runs:
        m, d = r["meta"], r["dir"]
        mt, cfg = m.get("metrics") or {}, m.get("config") or {}
        wt = window_table(
            m,
            data.walkforward_windows(f"{d}/window_metrics.parquet"),
            data.walkforward_curve(f"{d}/oos_curve.parquet"),
        )
        rows.append(
            {
                "label": r["label"],
                "engine": cfg.get("engine") or "windowed",
                "capital": cfg.get("initial_capital"),
                "objective": cfg.get("model_objective"),
                "sharpe": mt.get("sharpe"),
                "ann_return": mt.get("ann_return"),
                "max_drawdown": mt.get("max_drawdown"),
                "pred_ic": mt.get("pred_ic"),
                "nw_t": m.get("nw_t"),
                "windows": wt["n"],
                "pos": wt["pos_used"],
                "stale": wt["stale"],
                "generated_at": str(m.get("generated_at", ""))[:10],
                "oos": f"{str((m.get('universe') or {}).get('oos_start', ''))[:10]} ~ {str((m.get('universe') or {}).get('oos_end', ''))[:10]}",
                "dir": d,
            }
        )
    return rows


def snapshot_mismatch(rows: list[dict]) -> bool:
    """参与对比的实验 OOS 区间不同 ⇒ 差异里混着数据快照的影响，不是纯粹的方法差异。"""
    spans = {r["oos"] for r in rows if r["oos"].strip(" ~")}
    return len(spans) > 1


def noise_note() -> str:
    return "同一配置重跑的 Sharpe 随机波动约 ±0.4，差异落在这个带内不构成改进证据。"
