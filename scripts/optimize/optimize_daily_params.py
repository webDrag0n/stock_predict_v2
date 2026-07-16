"""每日重放回测的策略/执行参数扫描。

一次训练生成 score_cache，之后各 (rebalance_every, topk, max_positions,
min_notional, order_price_mode) 组合复用缓存分，只重跑 decide_orders +
simulate_fills，把小时级训练压缩到分钟级决策重放。

产出: outputs/optimize_daily_params.csv
"""
from __future__ import annotations

import os
import sys
import itertools
import json
import pickle
from datetime import date, timedelta

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
    """等权基准日收益，与 daily['date'] 对齐。"""
    ret_col = "return_1d" if "return_1d" in panel.columns else (
        "pct_chg" if "pct_chg" in panel.columns else None)
    if ret_col is None:
        return np.zeros(len(daily_dates))
    bser = panel.groupby("date")[ret_col].mean()
    if ret_col == "pct_chg":
        bser = bser / 100.0
    return bser.reindex(pd.Index(daily_dates)).fillna(0.0).to_numpy()


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame,
             panel: pd.DataFrame, initial_cash: float) -> dict:
    """从 daily/trades 计算关键指标。"""
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
    n_trades = int(len(trades))

    # alpha / beta vs equal-weight benchmark
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
        "n_trades": n_trades,
        "avg_positions": float(daily["n_positions"].mean()),
        "avg_risk_scale": float(daily["risk_scale"].mean()) if "risk_scale" in daily.columns else 1.0,
    }


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print(f"加载面板... FW={C.FORWARD_WINDOW} | 目标={C.MODEL_OBJECTIVE}")

    # 允许通过环境变量覆盖扫描范围，便于测试与不同时间窗口研究
    start = os.environ.get("OPT_START_DATE",
                           (date.today() - timedelta(days=92)).isoformat())
    end = os.environ.get("OPT_END_DATE", None)
    panel_path = os.environ.get("OPT_PANEL_PATH", "")

    if panel_path and os.path.exists(panel_path):
        print(f"从缓存面板加载: {panel_path}")
        panel = pd.read_parquet(panel_path)
    else:
        panel = load_factor_panel(keep_trade_cols=True,
                                  start_date=start or None,
                                  end_date=end or None,
                                  verbose=True)
    factors = list_factor_columns(panel)
    print(f"因子数={len(factors)} | 股票数={panel['symbol'].nunique()} "
          f"| 交易日={panel['date'].nunique()}\n")

    initial_cash = float(C.INITIAL_CAPITAL)

    n_dates = panel["date"].nunique()
    default_warmup = max(60, min(C.TRAIN_WINDOW, n_dates - 50))
    warmup_days = int(os.environ.get("OPT_WARMUP_DAYS", str(default_warmup)))

    base_params = {
        "n_seeds": 1,
        "rebuild_days": C.REBUILD_DAYS,
        "train_window": C.TRAIN_WINDOW,
        "ic_window": C.IC_WINDOW,
        "warmup_days": warmup_days,
        "collect_scores": True,
        "progress": False,
    }

    # 1) 基线训练并收集 score_cache（可通过环境变量复用已保存缓存，跳过小时级训练）
    cache_path = os.environ.get("OPT_SCORE_CACHE_PATH",
                                os.path.join(C.OUTPUT_DIR, "optimize_daily_params_score_cache.pkl"))
    resume_cache = os.environ.get("OPT_RESUME_SCORE_CACHE", "")
    if resume_cache and os.path.exists(resume_cache):
        print(f"===== 复用已有 score_cache: {resume_cache} =====")
        with open(resume_cache, "rb") as f:
            score_cache = pickle.load(f)
        print(f"score_cache: {len(score_cache)} 决策日\n")
    else:
        print("===== 基线训练 + 收集 score_cache =====")
        base_res = run_daily_replay(panel, factors, initial_cash=initial_cash,
                                    params=base_params, progress=True)
        score_cache = base_res["score_cache"]
        with open(cache_path, "wb") as f:
            pickle.dump(score_cache, f)
        print(f"score_cache: {len(score_cache)} 决策日 (已保存 {cache_path})\n")

    # 2) 参数网格（可通过环境变量 JSON 覆盖，便于快速测试）
    default_grids = {
        "rebalance_every": [5, 10, 15, 20],
        "topk": [0.10, 0.15, 0.20, 0.25],
        "max_positions": [20, 30, 50],
        "min_notional": [10000, 30000, 50000],
        "order_price_mode": ["next_open", "limit_bb"],
    }
    grids_json = os.environ.get("OPT_GRIDS", "")
    grids = json.loads(grids_json) if grids_json else default_grids
    keys = list(grids.keys())
    combos = list(itertools.product(*grids.values()))
    print(f"参数组合数: {len(combos)} | keys={keys}")

    rows = []
    for combo in combos:
        p = dict(base_params)
        p["score_cache"] = score_cache          # 复用缓存，跳过训练
        p["collect_scores"] = False
        p.update(dict(zip(keys, combo)))
        # 默认关闭择时，专注策略/执行参数
        p["use_bb_timing"] = False

        label = " | ".join(f"{k}={v}" for k, v in zip(keys, combo))
        print(f"  扫描: {label}", flush=True)
        try:
            res = run_daily_replay(panel, factors, initial_cash=initial_cash,
                                   params=p, progress=False)
            m = _metrics(res["daily"], res["trades"], panel, initial_cash)
            row = dict(zip(keys, combo))
            row.update(m)
            rows.append(row)
            print(f"    -> ret={m['total_ret']:+.2%} sharpe={m['sharpe']:+.2f} "
                  f"alpha={m['alpha_ann']:+.1%}(t={m['nw_t']:+.2f}) "
                  f"cost={m['cost']:,.0f} trades={m['n_trades']}")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            row = dict(zip(keys, combo))
            row["error"] = str(e)
            rows.append(row)

    # 3) 汇总输出
    tbl = pd.DataFrame(rows)
    numeric_cols = [c for c in tbl.columns if c not in keys and c != "error"]
    for c in numeric_cols:
        tbl[c] = pd.to_numeric(tbl[c], errors="coerce")

    out_csv = os.path.join(C.OUTPUT_DIR, "optimize_daily_params.csv")
    tbl.to_csv(out_csv, index=False)

    # 按 sharpe 排序，取 top 10
    ok = tbl[tbl["sharpe"].notna()].copy()
    top = ok.sort_values("sharpe", ascending=False).head(10)
    print("\n===== TOP 10 按 Sharpe =====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(top[keys + ["total_ret", "sharpe", "alpha_ann", "nw_t", "max_dd", "cost"]].to_string(index=False))

    # 按 alpha NW-t 排序
    top_t = ok[ok["nw_t"].notna()].sort_values("nw_t", ascending=False).head(10)
    print("\n===== TOP 10 按 alpha NW-t =====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(top_t[keys + ["total_ret", "sharpe", "alpha_ann", "nw_t", "max_dd", "cost"]].to_string(index=False))

    # 写入 JSON 摘要
    summary = {
        "baseline": _metrics(base_res["daily"], base_res["trades"], panel, initial_cash),
        "n_combos": len(combos),
        "top_sharpe": top.head(1).to_dict("records")[0] if len(top) else None,
        "top_alpha_t": top_t.head(1).to_dict("records")[0] if len(top_t) else None,
    }
    with open(os.path.join(C.OUTPUT_DIR, "optimize_daily_params_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存: {out_csv}")
    print(f"摘要已保存: outputs/optimize_daily_params_summary.json")


if __name__ == "__main__":
    main()
