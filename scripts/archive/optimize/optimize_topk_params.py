"""阶段二-b: 在修正后的周期调仓回测上, 扫描 TOPK_RATIO × HOLD_RATIO。

锁定预测器 ret_weighted, edge 在头部 → 更紧的 top-K 可能更准但更集中。换手缓冲
HOLD_RATIO 控成本。用 20日周期调仓回测(无虚高)逐组合评 净PnL/Sharpe/Calmar/MaxDD。
逐窗口预测先缓存一次, 各 (topk, hold) 组合复用预测, 只重算持仓+回测(快)。
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
from backtest import run_backtest_periodic


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print(f"加载面板... 预测器={C.MODEL_OBJECTIVE} 周期调仓={C.REBALANCE_DAYS}日")
    panel = load_factor_panel(keep_price=True)
    factors = list_factor_columns(panel)
    dates = np.sort(panel["date"].unique())

    # 1) 逐窗口预测一次, 缓存 OOS 测试帧(含预测)
    oos_frames = []
    for win_id, tr, te in _windows(dates):
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()
        test["prediction"] = _train_predict(train, test, factors)
        oos_frames.append(test)
    oos = pd.concat(oos_frames, ignore_index=True)
    oos_ids = _with_ids(oos)
    pred = oos["prediction"].to_numpy()
    print(f"OOS 拼接: {len(oos):,} 行, {oos['symbol'].nunique()} 股\n")

    # 2) 扫描 (topk, hold)。hold>=topk 才有缓冲意义; hold==topk 关闭缓冲
    topk_grid = [0.05, 0.10, 0.15, 0.20, 0.30]
    hold_grid = [0.0, 0.10, 0.15]   # 0 = 关闭(hold=topk); 否则 hold=topk+buffer
    rows = []
    orig_topk, orig_hold = C.TOPK_RATIO, C.HOLD_RATIO
    for topk, buf in itertools.product(topk_grid, hold_grid):
        hold = topk if buf == 0.0 else min(topk + buf, 0.6)
        C.TOPK_RATIO, C.HOLD_RATIO = topk, hold
        pos = generate_positions(oos_ids, pred, strategy="long_topk")
        res = run_backtest_periodic(oos_ids, pred, pos,
                                    rebalance_days=C.REBALANCE_DAYS,
                                    pred_in_return_space=True)
        m = res["metrics"]
        rows.append({
            "topk": topk, "hold": round(hold, 3),
            "net_pnl": m["total_pnl"], "sharpe": m["sharpe"],
            "calmar": m["calmar"], "maxdd": m["max_drawdown"],
            "cost": m["total_cost"], "hit": m["hit_rate"],
        })
        print(f"  topk={topk:.2f} hold={hold:.2f}: PnL={m['total_pnl']:+.4f} "
              f"Sharpe={m['sharpe']:+.2f} Calmar={m['calmar']:+.2f} "
              f"MaxDD={m['max_drawdown']:+.3f} cost={m['total_cost']:.4f}")
    C.TOPK_RATIO, C.HOLD_RATIO = orig_topk, orig_hold

    tbl = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\n===== TOPK × HOLD 扫描(周期调仓, 按 Sharpe 降序)=====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(tbl.to_string(index=False))
    tbl.to_csv(os.path.join(C.OUTPUT_DIR, "optimize_topk_params.csv"), index=False)
    best = tbl.iloc[0]
    print(f"\n最优: topk={best['topk']:.2f} hold={best['hold']:.2f} "
          f"| Sharpe={best['sharpe']:+.2f} PnL={best['net_pnl']:+.4f} "
          f"Calmar={best['calmar']:+.2f}")
    print(f"(当前默认: topk={orig_topk} hold={orig_hold})")
    print("-> outputs/optimize_topk_params.csv")


if __name__ == "__main__":
    main()
