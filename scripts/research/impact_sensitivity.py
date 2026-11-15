"""一次训练，多组市场冲击系数 K 的每日重放敏感性扫描。"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from experiment_log import log_trial
from perf_metrics import daily_replay_metrics
from walkforward import _replay_with_scores, _scores_to_cache, run_walkforward

DEFAULT_KS = (0.0, 0.05, 0.1, 0.2, 0.3)


def sweep_impact(replay_fn, ks) -> pd.DataFrame:
    rows = []
    for k in sorted(float(value) for value in ks):
        replay = replay_fn(k)
        metrics = daily_replay_metrics(
            replay["daily"], replay.get("meta", {}).get("initial_cash", C.INITIAL_CAPITAL)
        )
        rows.append(
            {
                "k": k,
                "sharpe": metrics["sharpe"],
                "ann_return": metrics["ann_return"],
                "max_drawdown": metrics["max_drawdown"],
                "total_cost": metrics["total_cost"],
            }
        )
    return pd.DataFrame(rows, columns=["k", "sharpe", "ann_return", "max_drawdown", "total_cost"])


def _clip_for_windows(panel: pd.DataFrame, max_windows: int | None) -> pd.DataFrame:
    if max_windows is None:
        return panel
    if max_windows <= 0:
        raise ValueError("--max-windows 必须为正整数")
    dates = sorted(panel["date"].unique())
    required = C.TRAIN_WINDOW + C.EMBARGO_DAYS + C.TEST_WINDOW + (max_windows - 1) * C.WF_STEP
    if required > len(dates):
        raise ValueError(
            f"面板只有 {len(dates)} 日，无法构造 {max_windows} 个窗口（需要 {required} 日）"
        )
    return panel[panel["date"].isin(dates[:required])].copy()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    panel = load_factor_panel(keep_trade_cols=True)
    panel = _clip_for_windows(panel, args.max_windows)
    factors = list_factor_columns(panel)
    result = run_walkforward(panel, factors, save=False)
    oos = result["oos"]
    cache = _scores_to_cache(oos)
    start, end = oos["date"].min(), oos["date"].max()

    def _replay(k):
        return _replay_with_scores(panel, factors, cache, start, end, {"impact_k": k})

    table = sweep_impact(_replay, DEFAULT_KS)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    outdir = os.path.join(C.OUTPUT_DIR, "experiments")
    os.makedirs(outdir, exist_ok=True)
    destination = os.path.join(outdir, "impact_sensitivity.csv")
    table.to_csv(destination, index=False)
    log_trial(
        "impact_sensitivity",
        {"ks": list(DEFAULT_KS), "max_windows": args.max_windows},
        {"results": table.to_dict(orient="records")},
    )
    print(f"冲击敏感性结果 -> {destination}")


if __name__ == "__main__":
    main()
