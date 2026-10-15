"""Risk-control ablation using the daily replay engine + score_cache reuse.

This is much faster than walkforward: the heavy model training is done once
(baseline score_cache from optimize_daily_params), and we only replay the
trading layer with different risk-control toggles.

Variants: baseline / dynamic_topk / beta_constraint / both.
Outputs: outputs/ablation_risk_controls_daily.csv
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
    cache_path = os.environ.get("ABL_SCORE_CACHE_PATH",
                                os.path.join(C.OUTPUT_DIR, "optimize_daily_params_score_cache.pkl"))

    print(f"加载面板... {start} ~ {end}")
    panel_path = os.environ.get("ABL_PANEL_PATH", "")
    if panel_path and os.path.exists(panel_path):
        print(f"从缓存面板加载: {panel_path}")
        panel = pd.read_parquet(panel_path)
        # 按日期范围过滤
        if start:
            panel = panel[panel["date"] >= np.datetime64(start)]
        if end:
            panel = panel[panel["date"] <= np.datetime64(end)]
    else:
        panel = load_factor_panel(keep_trade_cols=True, start_date=start, end_date=end, verbose=True)
    factors = list_factor_columns(panel)
    print(f"因子数={len(factors)} | 股票数={panel['symbol'].nunique()} | 交易日={panel['date'].nunique()}")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"score_cache 不存在: {cache_path}。请先跑 optimize_daily_params.py 基线。")
    print(f"复用 score_cache: {cache_path}")
    import pickle
    with open(cache_path, "rb") as f:
        score_cache_arg = pickle.load(f)
    # 引擎直接吃包装（解包+校验指纹）；本脚本查日期用裸分数 dict
    score_cache = (score_cache_arg["scores"]
                   if isinstance(score_cache_arg, dict) and "scores" in score_cache_arg
                   else score_cache_arg)
    # 截断到 score_cache 最后一天，避免产生没有缓存分的未来决策日
    max_score_date = max(score_cache.keys())
    panel = panel[panel["date"] <= max_score_date].copy()
    print(f"score_cache 决策日数: {len(score_cache)} | 面板保留交易日: {panel['date'].nunique()}\n")

    initial_cash = float(C.INITIAL_CAPITAL)

    base_params = {
        "warmup_days": warmup_days,
        "score_cache": score_cache_arg,
        "collect_scores": False,
        "progress": False,
        "use_ic_scale": os.environ.get("ABL_USE_IC_SCALE", "0") == "1",
        # 默认交易参数，与 optimize_daily_params 基线保持一致
        "topk": float(C.TOPK_RATIO),
        "max_positions": int(C.MAX_POSITIONS),
        "min_notional": float(getattr(C, "DAILY_MIN_NOTIONAL", 30000.0)),
        "order_price_mode": str(C.ORDER_PRICE_MODE),
        "use_bb_timing": False,
    }

    variants = [
        ("baseline", {"USE_DYNAMIC_TOPK": "0", "USE_BETA_CONSTRAINT": "0"}),
        ("dynamic_topk", {"USE_DYNAMIC_TOPK": "1", "USE_BETA_CONSTRAINT": "0"}),
        ("beta_constraint", {"USE_DYNAMIC_TOPK": "0", "USE_BETA_CONSTRAINT": "1"}),
        ("both", {"USE_DYNAMIC_TOPK": "1", "USE_BETA_CONSTRAINT": "1"}),
    ]

    rows = []
    for name, envs in variants:
        print(f"===== 配置: {name} =====")
        for k, v in envs.items():
            os.environ[k] = v
            setattr(C, k, v == "1")
        p = dict(base_params)
        p["use_dynamic_topk"] = C.USE_DYNAMIC_TOPK
        p["use_beta_constraint"] = C.USE_BETA_CONSTRAINT
        p["dynamic_topk_ic_full"] = float(C.DYNAMIC_TOPK_IC_FULL)
        p["dynamic_topk_ic_zero"] = float(C.DYNAMIC_TOPK_IC_ZERO)
        p["dynamic_topk_low"] = float(C.DYNAMIC_TOPK_LOW)
        p["dynamic_topk_high"] = float(C.DYNAMIC_TOPK_HIGH)
        p["max_portfolio_beta"] = float(C.MAX_PORTFOLIO_BETA)
        p["beta_lookback"] = int(C.BETA_LOOKBACK)
        try:
            res = run_daily_replay(panel, factors, initial_cash=initial_cash,
                                   params=p, progress=True)
            m = _metrics(res["daily"], panel, initial_cash)
            row = {"variant": name, **m}
            print(f"  -> ret={m['total_ret']:+.2%} sharpe={m['sharpe']:+.2f} "
                  f"alpha={m['alpha_ann']:+.1%}(t={m['nw_t']:+.2f}) "
                  f"turnover={m['turnover']:,.0f} cost={m['cost']:,.0f} "
                  f"avg_pos={m['avg_positions']:.1f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            row = {"variant": name, "error": str(e)}
        rows.append(row)
        print()

    tbl = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "ablation_risk_controls_daily.csv")
    tbl.to_csv(out, index=False)
    print(f"结果已保存: {out}")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
