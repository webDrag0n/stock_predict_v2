"""Attention model leg ablation using the daily replay engine.

This trains two daily-replay ensembles (baseline vs +attn) on the same data and
compares OOS trading metrics.  Much more realistic than walkforward for the
current production path because it shares trading.py with live/nextday.

Outputs: outputs/ablation_attn_daily.csv
"""
from __future__ import annotations

import os
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
    turnover = float(daily["turnover"].sum())
    cost = float(daily["cost"].sum())
    bench = _bench_returns(panel, daily["date"].to_numpy())
    lags = max(1, min(getattr(C, "FORWARD_WINDOW", 20), len(rr) - 2))
    alpha_d, beta, t_nw, _ = _newey_west_alpha_t(rr, bench[1:], lags)
    alpha_ann = float(alpha_d * 252)
    return {
        "total_ret": total_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "hit_rate": hit,
        "alpha_ann": alpha_ann,
        "nw_t": t_nw,
        "beta": beta,
        "turnover": turnover,
        "cost": cost,
        "avg_positions": float(daily["n_positions"].mean()),
        "avg_risk_scale": float(daily["risk_scale"].mean()) if "risk_scale" in daily.columns else 1.0,
    }


def main():
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    start = os.environ.get("ABL_START_DATE", "2024-01-01")
    end = os.environ.get("ABL_END_DATE", "2024-12-31")
    warmup_days = int(os.environ.get("ABL_WARMUP_DAYS", "120"))

    print(f"加载面板... {start} ~ {end}")
    panel = load_factor_panel(keep_trade_cols=True, start_date=start, end_date=end, verbose=True)
    factors = list_factor_columns(panel)
    print(f"因子数={len(factors)} | 股票数={panel['symbol'].nunique()} | 交易日={panel['date'].nunique()}\n")

    initial_cash = float(C.INITIAL_CAPITAL)

    base_params = {
        "n_seeds": 1,
        "rebuild_days": C.REBUILD_DAYS,
        "train_window": C.TRAIN_WINDOW,
        "ic_window": C.IC_WINDOW,
        "warmup_days": warmup_days,
        "collect_scores": True,
        "progress": True,
        "topk": float(C.TOPK_RATIO),
        "max_positions": int(C.MAX_POSITIONS),
        "min_notional": float(getattr(C, "DAILY_MIN_NOTIONAL", 30000.0)),
        "order_price_mode": str(C.ORDER_PRICE_MODE),
        "use_bb_timing": False,
    }

    variants = [
        ("baseline", ("gbm_l2", "gbm_ret_weighted", "ic_linear")),
        ("with_attn", ("attn", "gbm_l2", "gbm_ret_weighted", "ic_linear")),
    ]

    rows = []
    for name, legs in variants:
        print(f"===== 集成: {name} ({','.join(legs)}) =====")
        p = dict(base_params)
        p["ensemble_models"] = legs
        try:
            res = run_daily_replay(panel, factors, initial_cash=initial_cash,
                                   params=p, progress=True)
            m = _metrics(res["daily"], panel, initial_cash)
            # 保存该变体的 score_cache，方便后续复用
            cache_path = os.path.join(C.OUTPUT_DIR, f"ablation_attn_score_cache_{name}.pkl")
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(res.get("score_cache", {}), f)
            row = {"variant": name, "legs": ",".join(legs), **m}
            print(f"  -> ret={m['total_ret']:+.2%} sharpe={m['sharpe']:+.2f} "
                  f"alpha={m['alpha_ann']:+.1%}(t={m['nw_t']:+.2f}) "
                  f"turnover={m['turnover']:,.0f} cost={m['cost']:,.0f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            row = {"variant": name, "legs": ",".join(legs), "error": str(e)}
        rows.append(row)
        print()

    tbl = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "ablation_attn_daily.csv")
    tbl.to_csv(out, index=False)
    print(f"结果已保存: {out}")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
