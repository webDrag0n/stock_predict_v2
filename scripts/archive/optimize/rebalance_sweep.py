"""调仓周期扫描: 在锁定预测器+诚实长OOS上, 找最优 REBALANCE_DAYS。

调仓周期与预测目标horizon(FORWARD_WINDOW=20)是**独立**参数: 可以用20日预测
但每 N 日调一次仓。短周期=排序更新更勤(信号新鲜)但成本高; 长周期=成本低但
信号变陈、套牢更久。存在真实最优。

关键: 调仓周期只影响**回测网格**, 不影响预测/每日目标仓位。故只训练一次(逐窗口
预测缓存), 生成一次 long_topk 目标仓位, 再对各 rebalance_days 只重算 run_backtest_
periodic(便宜)。全部周期调仓回测(无FW=20每日重叠虚高), Sharpe 可跨周期比较
(curve 都是日级, ann=252 恒定)。

用法: TRAIN_WINDOW=500 .venv/bin/python rebalance_sweep.py   # 诚实长OOS
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
from walkforward import _train_predict, _with_ids
from factor_eval import _windows
from strategy import generate_positions
from backtest import run_backtest_periodic
from measure_alpha import _newey_west_alpha_t


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    grid = [int(x) for x in os.environ.get(
        "REBAL_GRID", "1,3,5,10,15,20,25,30,40,60").split(",")]
    print(f"调仓周期扫描 | 预测器={C.MODEL_OBJECTIVE} topk={C.TOPK_RATIO} "
          f"TRAIN={C.TRAIN_WINDOW} | 网格={grid}日")
    panel = load_factor_panel(keep_price=True)
    factors = list_factor_columns(panel)
    dates = np.sort(panel["date"].unique())

    # 一次训练: 逐窗口预测, 拼接连续 OOS
    oos_frames = []
    for win_id, tr, te in _windows(dates):
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()
        test["prediction"] = _train_predict(train, test, factors)
        oos_frames.append(test)
    oos = pd.concat(oos_frames, ignore_index=True)
    oos_ids = _with_ids(oos)
    pred = oos["prediction"].to_numpy()
    pos = generate_positions(oos_ids, pred, strategy="long_topk")
    print(f"OOS 拼接: {len(oos):,} 行, {oos['symbol'].nunique()} 股, "
          f"{oos['date'].nunique()} 日\n")

    # 等权基准日收益(用于 alpha)
    ret_col = "return_1d" if "return_1d" in oos.columns else "pct_chg"
    bench = oos.groupby("date")[ret_col].mean().sort_index().to_numpy()
    if ret_col == "pct_chg":
        bench = bench / 100.0

    rows = []
    for rd in grid:
        res = run_backtest_periodic(oos_ids, pred, pos, rebalance_days=rd,
                                    pred_in_return_space=True)
        m = res["metrics"]
        strat = res["oos_result"]["curve"]["net"].to_numpy() if "oos_result" in res \
            else res["curve"]["net"].to_numpy()
        n = min(len(strat), len(bench))
        alpha_d, beta, t_nw, _ = _newey_west_alpha_t(strat[:n], bench[:n],
                                                     min(rd, 30))
        rows.append({
            "rebal_days": rd, "net_pnl": m["total_pnl"], "sharpe": m["sharpe"],
            "calmar": m["calmar"], "maxdd": m["max_drawdown"],
            "cost": m["total_cost"], "hit": m["hit_rate"],
            "alpha_ann": alpha_d * 252, "alpha_t_nw": t_nw,
        })
        print(f"  调仓{rd:>3}日: PnL={m['total_pnl']:+.4f} Sharpe={m['sharpe']:+.2f} "
              f"Calmar={m['calmar']:+.2f} MaxDD={m['max_drawdown']:+.3f} "
              f"成本={m['total_cost']:.4f} | αann={alpha_d*252:+.2%} NWt={t_nw:+.2f}")

    tbl = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\n===== 调仓周期扫描(周期回测, 按 Sharpe 降序)=====")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}",
                           "display.width", 200):
        print(tbl.to_string(index=False))
    tbl.to_csv(os.path.join(C.OUTPUT_DIR, "rebalance_sweep.csv"), index=False)
    best = tbl.iloc[0]
    print(f"\n最优调仓周期: {int(best['rebal_days'])}日 "
          f"| Sharpe={best['sharpe']:+.2f} PnL={best['net_pnl']:+.4f} "
          f"Calmar={best['calmar']:+.2f} 成本={best['cost']:.4f}")
    print(f"(当前默认 REBALANCE_DAYS={C.REBALANCE_DAYS})")
    print("-> outputs/rebalance_sweep.csv")


if __name__ == "__main__":
    main()
