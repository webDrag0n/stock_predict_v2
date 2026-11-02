"""一致性验证：predict_tomorrow.py 与 pipeline/steps_nextday.generate_orders
在相同输入下产出相同订单。

通过 monkeypatch 替换 load_factor_panel / ensemble_predict_latest，避免真实数据加载。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_nextday_cli_consistency.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

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
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "close": close,
                    "adj_close": close,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "amount": 5000.0,
                    "pre_close": close * 0.99,
                    "pct_chg": 1.0,
                    "limit_flag": 0,
                    "return_1d": 0.01,
                    "bb_upper_20": close * 1.05,
                    "bb_lower_20": close * 0.95,
                    "rsi_14": 50.0,
                    C.TARGET: np.random.randn() * 0.02,
                    C.WEIGHT: 1.0,
                    "feature_00": np.random.randn(),
                    "feature_01": np.random.randn(),
                }
            )
    return pd.DataFrame(rows)


def _fake_ensemble(panel, factors, params=None):
    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    today["score"] = np.linspace(1.0, 0.1, len(today))
    today["prediction"] = today["score"]
    today["rank_pct"] = today["score"].rank(ascending=False, pct=True)
    info = {
        "weights": {"gbm_l2": 1.0},
        "ic_means": {"gbm_l2": 0.02},
        "models": ["gbm_l2"],
        "latest_date": latest,
    }
    return today, info


def test_nextday_session_matches_cli():
    """同输入下，pipeline nextday 步骤与 predict_tomorrow CLI 产出完全一致的订单。"""
    panel = _make_panel()
    params = {
        "capital": 1000000,
        "initial_holdings": "",
        "topk": 0.25,
        "max_positions": 2,
        "max_weight": 0.5,
        "use_bb_timing": False,
        "order_price_mode": "next_open",
        "min_notional": 1000,
    }

    # 1) pipeline/steps_nextday.generate_orders
    with tempfile.TemporaryDirectory() as tmp:
        # risk_state/pred_hist 走 outputs/ 共享路径（与 CLI 同一文件），
        # 测试里把 OUTPUT_DIR 指到临时目录，避免污染真实 outputs/。
        nx_out = os.path.join(tmp, "outputs")
        os.makedirs(nx_out, exist_ok=True)
        old_out = C.OUTPUT_DIR
        C.OUTPUT_DIR = nx_out
        try:
            ctx = ctxmod.StepContext(session_dir=tmp, params=params, shared={"panel": panel})
            ctx.shared["today"], ctx.shared["pred_info"] = _fake_ensemble(panel, ["feature_00"])
            ctx.shared["factors"] = ["feature_00"]
            generate_orders(ctx)
        finally:
            C.OUTPUT_DIR = old_out
        nextday_csv = ctx.artifact_path("orders.csv")
        df_nextday = pd.read_csv(nextday_csv) if os.path.exists(nextday_csv) else pd.DataFrame()

        # 会话结束后当日预测已写入共享 pred_hist（与 CLI 行为对齐）
        from live_state import load_pred_hist

        ph = load_pred_hist(os.path.join(nx_out, "live_risk_state_pred_hist.parquet"))
        latest_dstr = str(pd.Timestamp(panel["date"].max()).date())
        assert len(ph) and latest_dstr in set(ph["date"]), "webapp 会话未积累 pred_hist"

    # 2) predict_tomorrow.py main (monkeypatch data/ensemble)
    import predict_tomorrow as pt
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        with (
            patch.object(pt, "load_factor_panel", return_value=panel),
            patch.object(pt, "ensemble_predict_latest", side_effect=_fake_ensemble),
        ):
            old_out = C.OUTPUT_DIR
            old_track = C.TRACKING_DIR
            old_models = C.MODEL_DIR
            try:
                C.OUTPUT_DIR = out_dir
                C.TRACKING_DIR = os.path.join(tmp, "tracking")
                # 隔离真实 models/：本机已有 champion 时 adaptive 判定会走
                # 冻结 champion 路径（其特征集与合成面板不符），训练路径由
                # _fake_ensemble 顶替的前提是无 champion。
                C.MODEL_DIR = os.path.join(tmp, "models")
                sys.argv = [
                    "predict_tomorrow.py",
                    "--capital",
                    "1000000",
                    "--topk",
                    "0.25",
                    "--max-positions",
                    "2",
                    "--max-weight",
                    "0.5",
                    "--order-price-mode",
                    "next_open",
                    "--min-notional",
                    "1000",
                    "--price-col",
                    "adj_close",
                ]
                pt.main()
            finally:
                C.OUTPUT_DIR = old_out
                C.TRACKING_DIR = old_track
                C.MODEL_DIR = old_models
        latest = panel["date"].max().strftime("%Y%m%d")
        df_cli = pd.read_csv(os.path.join(out_dir, f"orders_{latest}.csv"))

    # 3) 比较：行数一致且逐行相同
    assert len(df_nextday) == len(df_cli), (
        f"行数不一致: nextday={len(df_nextday)}, cli={len(df_cli)}\n"
        f"nextday:\n{df_nextday}\ncli:\n{df_cli}"
    )
    if len(df_nextday) == 0:
        return  # 两者都为空也算一致
    cols = ["symbol", "side", "shares", "limit_price"]
    merged = df_nextday[cols].merge(df_cli[cols], on=cols, how="outer", indicator=True)
    diff = merged[merged["_merge"] != "both"]
    assert len(diff) == 0, f"订单不一致行：\n{diff}"


if __name__ == "__main__":
    test_nextday_session_matches_cli()
    print("nextday vs CLI consistency test passed")
