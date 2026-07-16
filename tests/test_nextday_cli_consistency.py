"""一致性验证：predict_tomorrow.py 与 pipeline/steps_nextday.generate_orders
在相同输入下产出相同订单。

通过 monkeypatch 替换 load_factor_panel / ensemble_predict_latest，避免真实数据加载。
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import config as C
from pipeline import context as ctxmod
from pipeline.steps_nextday import generate_orders


def _make_panel():
    dates = pd.date_range("2026-06-01", periods=40, freq="B")
    rows = []
    np.random.seed(7)
    for d in dates:
        for sym in ["600000.SH", "300001.SZ", "688001.SH", "000001.SZ"]:
            close = 20.0 + np.random.randn() * 2.0
            rows.append({
                "date": d, "symbol": sym, "close": close, "adj_close": close,
                "open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
                "amount": 5000.0, "pre_close": close * 0.99,
                "pct_chg": 1.0, "limit_flag": 0, "return_1d": 0.01,
                "bb_upper_20": close * 1.05, "bb_lower_20": close * 0.95,
                "rsi_14": 50.0,
                C.TARGET: np.random.randn() * 0.02,
                C.WEIGHT: 1.0,
                "feature_00": np.random.randn(),
                "feature_01": np.random.randn(),
            })
    return pd.DataFrame(rows)


def _fake_ensemble(panel, factors, params=None):
    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    today["score"] = np.linspace(1.0, 0.1, len(today))
    today["prediction"] = today["score"]
    today["rank_pct"] = today["score"].rank(ascending=False, pct=True)
    info = {"weights": {"gbm_l2": 1.0}, "ic_means": {"gbm_l2": 0.02},
            "models": ["gbm_l2"], "latest_date": latest}
    return today, info


def main():
    panel = _make_panel()
    params = {
        "capital": 1000000, "initial_holdings": "",
        "topk": 0.25, "max_positions": 2, "max_weight": 0.5,
        "use_bb_timing": False, "order_price_mode": "next_open",
        "min_notional": 1000,
    }

    # 1) pipeline/steps_nextday.generate_orders
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ctxmod.StepContext(session_dir=tmp, params=params, shared={"panel": panel})
        ctx.shared["today"], ctx.shared["pred_info"] = _fake_ensemble(panel, ["feature_00"])
        ctx.shared["factors"] = ["feature_00"]
        report = generate_orders(ctx)
        print("nextday report:", report)
        nextday_csv = ctx.artifact_path("orders.csv")
        print("nextday orders csv path:", nextday_csv, "exists:", os.path.exists(nextday_csv))
        if os.path.exists(nextday_csv):
            df_nextday = pd.read_csv(nextday_csv)
        else:
            df_nextday = pd.DataFrame()

    # 2) predict_tomorrow.py main (monkeypatch data/ensemble)
    import predict_tomorrow as pt
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        with patch.object(pt, "load_factor_panel", return_value=panel), \
             patch.object(pt, "ensemble_predict_latest", side_effect=_fake_ensemble):
            old_out = C.OUTPUT_DIR
            try:
                C.OUTPUT_DIR = out_dir
                sys.argv = ["predict_tomorrow.py", "--capital", "1000000",
                            "--topk", "0.25", "--max-positions", "2",
                            "--max-weight", "0.5", "--order-price-mode", "next_open",
                            "--min-notional", "1000", "--price-col", "adj_close"]
                pt.main()
            finally:
                C.OUTPUT_DIR = old_out
        latest = panel["date"].max().strftime("%Y%m%d")
        df_cli = pd.read_csv(os.path.join(out_dir, f"orders_{latest}.csv"))

    # 3) 比较
    print("Nextday session orders:")
    print(df_nextday[["symbol", "side", "shares", "limit_price"]].to_string(index=False) if len(df_nextday) else "(empty)")
    print("\nCLI predict_tomorrow orders:")
    print(df_cli[["symbol", "side", "shares", "limit_price"]].to_string(index=False) if len(df_cli) else "(empty)")

    if len(df_nextday) != len(df_cli):
        print(f"\n行数不一致: nextday={len(df_nextday)}, cli={len(df_cli)}")
        return 1
    if len(df_nextday) == 0:
        print("\n两者都为空，一致。")
        return 0
    cols = ["symbol", "side", "shares", "limit_price"]
    merged = df_nextday[cols].merge(df_cli[cols], on=cols, how="outer", indicator=True)
    diff = merged[merged["_merge"] != "both"]
    if len(diff):
        print(f"\n不一致行：\n{diff}")
        return 1
    print("\n订单完全一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
