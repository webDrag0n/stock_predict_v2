"""Synthetic integration test for backtest_daily.py.

Creates a tiny multi-day panel with two stocks and runs run_daily_replay.
Verifies: no exceptions, cash conservation (strict_check), finite equity curve.

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_backtest_synthetic.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from backtest_daily import run_daily_replay


def _make_panel(n_dates: int = 40):
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rows = []
    np.random.seed(42)
    for d in dates:
        for sym in ["600000.SH", "300001.SZ"]:
            close = 10.0 + np.random.randn() * 0.5
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "close": close,
                    "adj_close": close,
                    "open": close * (1 + np.random.randn() * 0.01),
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "amount": 10000.0,
                    "pre_close": close * (1 + np.random.randn() * 0.01),
                    "pct_chg": np.random.randn() * 2.0,
                    "limit_flag": 0,
                    "return_1d": np.random.randn() * 0.02,
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


def test_backtest_synthetic_runs_with_strict_check():
    """小合成面板端到端跑通：strict_check（现金/估值守恒）不炸，净值序列有限。"""
    panel = _make_panel(80)
    feat_cols = ["feature_00", "feature_01"]

    res = run_daily_replay(
        panel,
        feat_cols,
        params={
            "n_seeds": 1,
            "rebuild_days": 20,
            "train_window": 20,
            "ic_window": 5,
            "warmup_days": 20,
            "strict_check": True,
            "progress": False,
        },
    )
    daily = res["daily"]
    trades = res["trades"]

    # 应有决策日产出（warmup+embargo 之后）
    assert len(daily) > 0, "回测没有产生任何决策日"
    # 净值/现金有限且非负
    assert np.isfinite(daily["total_value"]).all()
    assert (daily["total_value"] > 0).all()
    assert np.isfinite(daily["cash"]).all()
    # 守恒口径：total_value == cash + stock_value（strict_check 内部另有硬断言）
    assert np.allclose(daily["total_value"], daily["cash"] + daily["stock_value"], atol=1.0)
    # 成本与换手非负
    assert (daily["cost"] >= 0).all()
    assert (daily["turnover"] >= 0).all()
    # 成交明细结构完整（有交易时）
    if len(trades):
        for col in ("symbol", "side", "fill_shares", "fill_price"):
            assert col in trades.columns, f"trades 缺列 {col}"


def test_score_cache_fingerprint_validation():
    """score_cache 指纹：带正确指纹的包装可重放；指纹不匹配显式报错。"""
    panel = _make_panel(80)
    feat_cols = ["feature_00", "feature_01"]
    base_params = {
        "n_seeds": 1,
        "rebuild_days": 20,
        "train_window": 20,
        "ic_window": 5,
        "warmup_days": 20,
        "strict_check": True,
        "progress": False,
    }

    res = run_daily_replay(panel, feat_cols, params={**base_params, "collect_scores": True})
    fp = res["meta"]["score_fingerprint"]
    assert fp and isinstance(fp, str)
    cache = res["score_cache"]

    # 正确指纹 → 正常重放（跳过训练）
    res2 = run_daily_replay(
        panel,
        feat_cols,
        params={**base_params, "score_cache": {"scores": cache, "fingerprint": fp}},
    )
    assert len(res2["daily"]) == len(res["daily"])
    assert res2["meta"]["used_cache"] is True

    # 错误指纹 → 显式拒绝（防陈旧缓存静默复用）
    with np.errstate(all="ignore"):
        try:
            run_daily_replay(
                panel,
                feat_cols,
                params={**base_params, "score_cache": {"scores": cache, "fingerprint": "deadbeef"}},
            )
        except ValueError as e:
            assert "指纹不匹配" in str(e)
        else:
            raise AssertionError("指纹不匹配的 score_cache 未被拒绝")


if __name__ == "__main__":
    test_backtest_synthetic_runs_with_strict_check()
    test_score_cache_fingerprint_validation()
    print("synthetic backtest test passed")
