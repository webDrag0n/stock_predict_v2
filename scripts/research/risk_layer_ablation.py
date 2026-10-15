"""风控层松绑消融：哪一层在绑敞口？（复用 tier2 的 score_cache，纯重放）

Tier 2 已证明 TARGET_VOL≥0.15 不生效（0.15 与 0.20 结果逐位相同），敞口被
下游层压在 ~43%。本脚本逐层松绑（ic 降仓 / 回撤刹车 / vol target / IV 降仓 /
全关），用同一批缓存分重放，定位敞口的真实约束层。

用法：uv run python -u scripts/research/risk_layer_ablation.py
环境变量：T2_START_DATE / T2_END_DATE / T2_CACHE 与 tier2_target_vol.py 相同。
输出：outputs/risk_layer_ablation.csv
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
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts", "research"))

import config as C
from data import load_factor_panel, list_factor_columns
from backtest_daily import run_daily_replay
from tier2_target_vol import _metrics, T2_START, T2_END, CACHE_PATH

VARIANTS = [
    ("base（生产默认）", {}),
    ("no_ic_scale", {"use_ic_scale": False}),
    ("no_dd_brake", {"use_dd_brake": False}),
    ("no_vol_target", {"use_vol_target": False}),
    ("no_iv_scale", {"use_iv_scale": False}),
    ("all_off", {"use_ic_scale": False, "use_dd_brake": False,
                 "use_vol_target": False, "use_iv_scale": False,
                 "use_regime": False}),
]


def main() -> None:
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"score_cache 不存在: {CACHE_PATH}。请先跑 tier2_target_vol.py。")
    with open(CACHE_PATH, "rb") as f:
        score_cache_arg = pickle.load(f)
    # 引擎直接吃包装（解包+校验指纹）；本脚本查日期用裸分数 dict
    score_cache = (score_cache_arg["scores"]
                   if isinstance(score_cache_arg, dict) and "scores" in score_cache_arg
                   else score_cache_arg)

    print("加载面板（float32 + 截断，与 tier2 相同口径）...")
    panel = load_factor_panel(keep_trade_cols=True, verbose=True)
    factors = list_factor_columns(panel)
    for c in panel.columns:
        if panel[c].dtype == np.float64:
            panel[c] = panel[c].astype(np.float32)
    max_score_date = max(score_cache.keys())
    lo = np.datetime64(T2_START) - np.timedelta64(150, "D")
    panel = panel[(panel["date"] <= max_score_date) & (panel["date"] >= lo)]
    print(f"重放面板: {panel['date'].nunique()} 交易日 | "
          f"{panel.memory_usage(deep=True).sum()/2**30:.1f}GB | "
          f"缓存决策日 {len(score_cache)}")

    initial_cash = float(C.INITIAL_CAPITAL)
    rows = []
    for name, over in VARIANTS:
        print(f"\n===== {name} =====")
        params = {"score_cache": score_cache_arg, "collect_scores": False,
                  "warmup_days": 60}
        params.update(over)
        res = run_daily_replay(
            panel, factors, initial_cash=initial_cash, params=params,
            start_date=T2_START, end_date=T2_END, progress=False)
        m = _metrics(res["daily"], panel, initial_cash)
        rows.append({"variant": name, **m})
        print(f"  -> ret={m['total_ret']:+.2%} sharpe={m['sharpe']:+.2f} "
              f"alpha={m['alpha_ann']:+.1%}(NW-t={m['nw_t']:+.2f}) "
              f"beta={m['beta']:+.2f} maxDD={m['max_dd']:+.1%} "
              f"敞口={m['avg_exposure']:.0%}")

    tbl = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "risk_layer_ablation.csv")
    tbl.to_csv(out, index=False)
    print(f"\n结果已保存: {out}")
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
