"""Tier 2 敞口实验：TARGET_VOL 对照（每日重放引擎 + score_cache 复用）。

生产归因显示敞口均值仅 50%（vol target 15% + IC 降仓 + 回撤刹车叠乘）。
本脚本量化敞口杠杆的边际收益：一次训练收集 score_cache（约 130 个决策日），
再按多个 TARGET_VOL 值纯重放（无训练，每档分钟级）。

用法
----
  uv run python -u scripts/research/tier2_target_vol.py
环境变量：
  T2_START_DATE  回测交易起点（默认 2026-02-01；训练自动用更早历史）
  T2_END_DATE    回测交易终点（默认到数据最新）
  T2_VOLS        逗号分隔的 TARGET_VOL 档（默认 0.10,0.15,0.20）
  T2_CACHE       score_cache 路径（默认 outputs/tier2_score_cache.pkl）；
                 若已存在且 T2_FORCE_TRAIN!=1 则跳过训练直接复用
输出：outputs/tier2_target_vol.csv + 逐档 daily parquet。

注意：score_cache 与训练时的信号配置（ENSEMBLE_MODELS 等）绑定，
改信号配置后必须删缓存重训（T2_FORCE_TRAIN=1）。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from backtest_daily import run_daily_replay
from measure_alpha import _newey_west_alpha_t

T2_START = os.environ.get("T2_START_DATE", "2026-02-01")
T2_END = os.environ.get("T2_END_DATE", "") or None
T2_VOLS = [float(x) for x in os.environ.get("T2_VOLS", "0.10,0.15,0.20").split(",")]
CACHE_PATH = os.environ.get("T2_CACHE",
                            os.path.join(C.OUTPUT_DIR, "tier2_score_cache.pkl"))
FORCE_TRAIN = os.environ.get("T2_FORCE_TRAIN", "0") == "1"


def _bench_returns(panel: pd.DataFrame, daily_dates) -> np.ndarray:
    ret_col = "return_1d" if "return_1d" in panel.columns else (
        "pct_chg" if "pct_chg" in panel.columns else None)
    if ret_col is None:
        return np.zeros(len(daily_dates))
    bser = panel.groupby("date")[ret_col].mean()
    if ret_col == "pct_chg":
        bser = bser / 100.0
    return bser.reindex(pd.Index(daily_dates)).fillna(0.0).to_numpy()


def _metrics(daily: pd.DataFrame, panel: pd.DataFrame, initial_cash: float) -> dict:
    if len(daily) == 0:
        return {}
    tv = daily["total_value"].to_numpy(dtype=np.float64)
    rets = np.zeros(len(tv))
    rets[1:] = tv[1:] / tv[:-1] - 1.0
    rr = rets[1:]
    sharpe = float(rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 1e-12 else 0.0
    downside = rr[rr < 0]
    sortino = (float(rr.mean() / downside.std() * np.sqrt(252))
               if len(downside) > 1 and downside.std() > 1e-12 else 0.0)
    hit = float((rr > 0).mean()) if len(rr) else float("nan")
    peak = np.maximum.accumulate(tv)
    max_dd = float(np.min((tv - peak) / peak))
    total_ret = float(tv[-1] / initial_cash - 1.0)
    bench = _bench_returns(panel, daily["date"].to_numpy())
    lags = max(1, min(getattr(C, "FORWARD_WINDOW", 20), len(rr) - 2))
    alpha_d, beta, t_nw, _ = _newey_west_alpha_t(rr, bench[1:], lags)
    exposure = (daily["stock_value"] / daily["total_value"].clip(lower=1.0)
                if "stock_value" in daily.columns else pd.Series(dtype=float))
    return {
        "total_ret": total_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "hit_rate": hit,
        "alpha_ann": float(alpha_d * 252),
        "nw_t": t_nw,
        "beta": beta,
        "avg_exposure": float(exposure.mean()) if len(exposure) else float("nan"),
        "turnover": float(daily["turnover"].sum()),
        "cost": float(daily["cost"].sum()),
        "avg_positions": float(daily["n_positions"].mean()),
        "n_days": int(len(daily)),
    }


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print(f"加载面板（keep_trade_cols=True，全历史）... 交易区间 {T2_START} ~ {T2_END or '最新'}")
    panel = load_factor_panel(keep_trade_cols=True, verbose=True)
    factors = list_factor_columns(panel)
    # 逐列原地 float64→float32（整帧 astype 会再复制一份 9GB，直接 OOM）。
    # 回测/撮合对日收益与价格的精度需求远低于 float32 的 7 位有效数字。
    for c in panel.columns:
        if panel[c].dtype == np.float64:
            panel[c] = panel[c].astype(np.float32)
    print(f"因子数={len(factors)} | 股票数={panel['symbol'].nunique()} | "
          f"交易日={panel['date'].nunique()} | 集成 legs={C.ENSEMBLE_MODELS} | "
          f"面板内存={panel.memory_usage(deep=True).sum()/2**30:.1f}GB(float32)")

    initial_cash = float(C.INITIAL_CAPITAL)

    # --- Phase A: 训练并收集 score_cache（或复用已有缓存） ---
    # score_cache_arg 为带指纹的包装（{"scores", "fingerprint"}）或旧格式裸 dict，
    # 交给引擎解包+校验；score_cache 为裸分数 dict 供本脚本查日期。
    if os.path.exists(CACHE_PATH) and not FORCE_TRAIN:
        print(f"复用已有 score_cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            score_cache_arg = pickle.load(f)
        score_cache = (score_cache_arg["scores"]
                       if isinstance(score_cache_arg, dict) and "scores" in score_cache_arg
                       else score_cache_arg)
    else:
        print(f"开始训练收集 score_cache（legs={C.ENSEMBLE_MODELS}）...")
        res = run_daily_replay(
            panel, factors, initial_cash=initial_cash,
            params={"collect_scores": True},
            start_date=T2_START, end_date=T2_END, progress=True)
        score_cache = res.get("score_cache", {})
        score_cache_arg = {"scores": score_cache,
                           "fingerprint": res["meta"]["score_fingerprint"]}
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(score_cache_arg, f)
        print(f"score_cache 已保存: {CACHE_PATH}（{len(score_cache)} 个决策日）")
        # 基线（TARGET_VOL=默认）daily 也存一份，省一次重放
        res["daily"].to_parquet(
            os.path.join(C.OUTPUT_DIR,
                         f"tier2_daily_vol{C.TARGET_VOL:.2f}_train.parquet"),
            index=False)

    if not score_cache:
        raise RuntimeError("score_cache 为空，无法重放")
    # 截断面板：上到缓存最后决策日（避免无缓存的未来决策日）；下到交易起点
    # 前 150 个自然日（缓存模式下不训练，只需覆盖 beta_lookback/regime/vol 等
    # 风控回溯窗）。布尔掩码本身已产生新帧，无需再 .copy()。
    max_score_date = max(score_cache.keys())
    lo = np.datetime64(T2_START) - np.timedelta64(150, "D")
    panel = panel[(panel["date"] <= max_score_date) & (panel["date"] >= lo)]
    print(f"重放面板: {panel['date'].nunique()} 交易日 | "
          f"{panel.memory_usage(deep=True).sum()/2**30:.1f}GB")

    # --- Phase B: 各 TARGET_VOL 档纯重放 ---
    rows = []
    for vol in T2_VOLS:
        print(f"\n===== TARGET_VOL={vol:.2f} 重放 =====")
        res = run_daily_replay(
            panel, factors, initial_cash=initial_cash,
            params={"score_cache": score_cache_arg, "collect_scores": False,
                    "target_vol": vol,
                    # 缓存模式不训练，warmup 只需覆盖风控回溯窗
                    # （beta_lookback=60 最长），默认 TRAIN_WINDOW=500 会
                    # 在截断面板上导致"决策日为空"。
                    "warmup_days": 60},
            start_date=T2_START, end_date=T2_END, progress=True)
        m = _metrics(res["daily"], panel, initial_cash)
        row = {"target_vol": vol, **m}
        rows.append(row)
        res["daily"].to_parquet(
            os.path.join(C.OUTPUT_DIR, f"tier2_daily_vol{vol:.2f}.parquet"),
            index=False)
        print(f"  -> ret={m['total_ret']:+.2%} sharpe={m['sharpe']:+.2f} "
              f"alpha={m['alpha_ann']:+.1%}(NW-t={m['nw_t']:+.2f}) "
              f"beta={m['beta']:+.2f} maxDD={m['max_dd']:+.1%} "
              f"敞口={m['avg_exposure']:.0%} 成本={m['cost']:,.0f}")

    tbl = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "tier2_target_vol.csv")
    tbl.to_csv(out, index=False)
    print(f"\n结果已保存: {out}")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
