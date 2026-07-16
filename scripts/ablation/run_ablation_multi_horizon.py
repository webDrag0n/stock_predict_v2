"""Multi-horizon target ablation: ret_weighted vs multi_ret_weighted vs multi_fusion.

Loads the panel once and runs walk-forward for each objective, keeping all other
config identical. Outputs a comparison CSV to outputs/ablation_multi_horizon.csv.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import run_walkforward


def _extract(result: dict) -> dict:
    m = result["oos_result"]["metrics"]
    return {
        "n_windows": int(result["n_windows"]),
        "total_pnl": float(m.get("total_pnl", np.nan)),
        "sharpe": float(m.get("sharpe", np.nan)),
        "sortino": float(m.get("sortino", np.nan)),
        "calmar": float(m.get("calmar", np.nan)),
        "max_drawdown": float(m.get("max_drawdown", np.nan)),
        "hit_rate": float(m.get("hit_rate", np.nan)),
        "pred_ic": float(m.get("pred_ic", np.nan)),
        "total_cost": float(m.get("total_cost", np.nan)),
    }


def main():
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    start = os.environ.get("ABL_START_DATE")
    end = os.environ.get("ABL_END_DATE")
    print(f"加载面板... {start or 'auto'} ~ {end or 'auto'}")
    panel = load_factor_panel(keep_price=True, start_date=start, end_date=end, verbose=True)
    factors = list_factor_columns(panel)
    print(f"因子数={len(factors)} | 股票数={panel['symbol'].nunique()}\n")

    objectives = ["ret_weighted", "multi_ret_weighted", "multi_fusion"]
    rows = []
    for obj in objectives:
        print(f"===== 目标: {obj} =====")
        os.environ["MODEL_OBJECTIVE"] = obj
        # Reload config to pick up the new objective (config reads env at import,
        # but we already imported it; set the module attribute directly).
        C.MODEL_OBJECTIVE = obj
        try:
            res = run_walkforward(panel, factors, select_features=False,
                                  strategy="long_topk", verbose=True)
            row = {"objective": obj, **_extract(res)}
        except Exception as e:
            print(f"FAILED: {e}")
            row = {"objective": obj, "error": str(e)}
        rows.append(row)
        print()

    tbl = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "ablation_multi_horizon.csv")
    tbl.to_csv(out, index=False)
    print(f"结果已保存: {out}")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
