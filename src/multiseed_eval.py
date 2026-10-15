"""多 seed 回测评估（纯计算，无画图）。

原 scripts/plot_multiseed_dashboard.py 的逐 seed OOS 回测逻辑上移至此，
供前端/报表复用：输入单个 seed 的 OOS 帧与 amount 映射，输出逐日
net/bench/cost 序列与指标 dict（含 Newey-West alpha）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def seed_backtest(oos: pd.DataFrame, amount_map: pd.DataFrame) -> dict:
    """单seed: 合并amount → 周期调仓回测(含冲击成本) → 返回curve序列+metrics+alpha序列。"""
    # 延迟导入：walkforward/strategy/backtest/measure_alpha 是 src 内平铺模块，
    # 调用方（脚本/前端）需先把 src 加入 sys.path；保持本模块可独立 import。
    from walkforward import _with_ids
    from strategy import generate_positions
    from backtest import run_backtest_periodic
    from measure_alpha import _newey_west_alpha_t

    oos = oos.merge(amount_map, on=["date", "symbol"], how="left")
    oos_ids = _with_ids(oos)
    pred = oos["prediction"].to_numpy()
    pos = generate_positions(oos_ids, pred, strategy="long_topk")
    # carry amount into the ids frame (run_backtest_periodic reads df['amount'])
    oos_ids["amount"] = oos["amount"].to_numpy()
    res = run_backtest_periodic(oos_ids, pred, pos,
                                rebalance_days=C.REBALANCE_DAYS,
                                pred_in_return_space=True)
    curve = res["curve"]
    m = res["metrics"]
    net = curve["net"].to_numpy()
    bench = oos.groupby("date")["return_1d"].mean().sort_index().to_numpy()
    n = min(len(net), len(bench))
    a_d, beta, t_nw, r2 = _newey_west_alpha_t(net[:n], bench[:n], C.REBALANCE_DAYS)
    strat_cum = np.cumprod(1 + net) - 1.0
    bench_cum = np.cumprod(1 + bench[:n]) - 1.0
    return {
        "net": net, "bench": bench[:n],
        "strat_cum_pct": strat_cum * 100, "bench_cum_pct": bench_cum * 100,
        "alpha_pct": (strat_cum[:n] - bench_cum) * 100,
        "tc_comm": curve["tc_commission"].to_numpy(),
        "tc_impact": curve["tc_impact"].to_numpy(),
        "sharpe": m["sharpe"], "calmar": m["calmar"], "maxdd": m["max_drawdown"],
        "alpha_ann": a_d * 252, "nw_t": t_nw, "beta": beta,
        "total_pnl": m["total_pnl"], "cost_comm": m["total_cost_commission"],
        "cost_impact": m["total_cost_impact"], "hit": m["hit_rate"],
    }
