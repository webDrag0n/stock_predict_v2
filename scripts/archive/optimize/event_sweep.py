"""事件驱动调仓参数扫描: (MAX_HOLD, TURNOVER_THRESHOLD) 网格搜索。

对比固定周期调仓, 测试事件驱动是否能提升 Sharpe/alpha/成本效率。

关键: 每日预测(每个OOS日都有新预测) + 信号衰减 + 偏离触发。训练仍按walk-forward
窗口(训练一次,预测该窗口所有日期), 但回测用事件驱动引擎。

网格:
- MAX_HOLD: 20, 30, 40, 60 (信号衰减到0的天数, 类似旧REBALANCE_DAYS)
- TURNOVER_THRESHOLD: 0.1, 0.2, 0.3, 0.5 (触发调仓的偏离阈值, 越小越频繁)
- DECAY_MODE: linear (固定, 先测线性衰减; exp衰减可后续加)
- REBAL_THRESHOLD: 0.005 (固定, 个股最小调整量0.5%)

用法: TRAIN_WINDOW=500 .venv/bin/python event_sweep.py   # 诚实长OOS
"""
from __future__ import annotations

import os
import sys
import itertools

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import _train_predict, _with_ids
from factor_eval import _windows
from strategy import generate_positions
from backtest_event import run_backtest_event_driven
from measure_alpha import _newey_west_alpha_t


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    max_hold_grid = [int(x) for x in os.environ.get(
        "MAX_HOLD_GRID", "20,30,40,60").split(",")]
    threshold_grid = [float(x) for x in os.environ.get(
        "TURNOVER_GRID", "0.1,0.2,0.3,0.5").split(",")]

    print(f"事件驱动调仓扫描 | 预测器={C.MODEL_OBJECTIVE} topk={C.TOPK_RATIO} "
          f"TRAIN={C.TRAIN_WINDOW}")
    print(f"网格: MAX_HOLD={max_hold_grid}, TURNOVER_THRESHOLD={threshold_grid}")
    panel = load_factor_panel(keep_price=True)
    factors = list_factor_columns(panel)
    dates = np.sort(panel["date"].unique())

    # 一次训练: 逐窗口预测, 但每个窗口**预测所有测试日期**(非仅窗口末)
    # 这样拼接后每个OOS日都有预测
    print("\n训练并生成每日预测...")
    oos_frames = []
    for win_id, tr, te in _windows(dates):
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()
        test["prediction"] = _train_predict(train, test, factors)
        oos_frames.append(test)
    oos = pd.concat(oos_frames, ignore_index=True)
    print(f"OOS 拼接: {len(oos):,} 行, {oos['symbol'].nunique()} 股, "
          f"{oos['date'].nunique()} 日\n")

    # 等权基准(用于alpha)
    ret_col = "return_1d" if "return_1d" in oos.columns else "pct_chg"
    bench = oos.groupby("date")[ret_col].mean().sort_index().to_numpy()
    if ret_col == "pct_chg":
        bench = bench / 100.0

    # 每日生成目标仓位(long_topk, 每天都有)
    oos_ids = _with_ids(oos)
    pred = oos["prediction"].to_numpy()
    pos_daily = generate_positions(oos_ids, pred, strategy="long_topk")

    rows = []
    for max_h, thr in itertools.product(max_hold_grid, threshold_grid):
        res = run_backtest_event_driven(
            oos, pred, pos_daily, max_hold=max_h, decay_mode="linear",
            turnover_threshold=thr, rebal_threshold=0.005,
            pred_in_return_space=True)
        m = res["metrics"]
        strat = res["curve"]["net"].to_numpy()
        n = min(len(strat), len(bench))
        alpha_d, beta, t_nw, _ = _newey_west_alpha_t(
            strat[:n], bench[:n], min(max_h, 30))
        rows.append({
            "max_hold": max_h, "turnover_thr": thr,
            "n_rebal": m["n_rebalances"], "rebal_freq": m["rebal_freq_days"],
            "net_pnl": m["total_pnl"], "sharpe": m["sharpe"],
            "calmar": m["calmar"], "maxdd": m["max_drawdown"],
            "cost": m["total_cost"], "hit": m["hit_rate"],
            "alpha_ann": alpha_d * 252, "alpha_t_nw": t_nw,
        })
        print(f"  MAX_HOLD={max_h:>2} thr={thr:.2f}: "
              f"PnL={m['total_pnl']:+.4f} Sharpe={m['sharpe']:+.2f} "
              f"Calmar={m['calmar']:+.2f} 成本={m['total_cost']:.4f} "
              f"调仓{m['n_rebalances']}次(每{m['rebal_freq_days']:.1f}日) "
              f"| αann={alpha_d*252:+.2%} NWt={t_nw:+.2f}")

    tbl = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\n===== 事件驱动调仓扫描(按 Sharpe 降序)=====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}" if abs(v)>=0.01 else f"{v:.4f}",
                           "display.width", 220, "display.max_columns", 15):
        print(tbl.to_string(index=False))
    tbl.to_csv(os.path.join(C.OUTPUT_DIR, "event_sweep.csv"), index=False)

    best = tbl.iloc[0]
    print(f"\n最优参数: MAX_HOLD={int(best['max_hold'])} "
          f"TURNOVER_THR={best['turnover_thr']:.2f} "
          f"| Sharpe={best['sharpe']:+.2f} PnL={best['net_pnl']:+.4f} "
          f"调仓{int(best['n_rebal'])}次(每{best['rebal_freq']:.1f}日)")

    # 对比: 固定周期20日(当前默认)的结果(从 rebalance_sweep.csv 读, 或重算)
    rebal_csv = os.path.join(C.OUTPUT_DIR, "rebalance_sweep.csv")
    if os.path.exists(rebal_csv):
        periodic = pd.read_csv(rebal_csv)
        p20 = periodic[periodic["rebal_days"] == 20]
        if not p20.empty:
            p20 = p20.iloc[0]
            print(f"\n对比固定周期20日: Sharpe={p20['sharpe']:+.2f} "
                  f"PnL={p20['net_pnl']:+.4f} 成本={p20['cost']:.4f}")
            delta_sharpe = best['sharpe'] - p20['sharpe']
            print(f"事件驱动优势: Δ Sharpe = {delta_sharpe:+.2f} "
                  f"({'改善' if delta_sharpe > 0 else '退化'})")
    print("-> outputs/event_sweep.csv")


if __name__ == "__main__":
    main()
