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
            rows.append({
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
            })
    return pd.DataFrame(rows)


def test_backtest_synthetic_runs_with_strict_check():
    """小合成面板端到端跑通：strict_check（现金/估值守恒）不炸，净值序列有限。"""
    panel = _make_panel(80)
    feat_cols = ["feature_00", "feature_01"]

    res = run_daily_replay(
        panel, feat_cols,
        params={"n_seeds": 1, "rebuild_days": 20, "train_window": 20,
                "ic_window": 5, "warmup_days": 20,
                "strict_check": True, "progress": False},
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
    assert np.allclose(daily["total_value"],
                       daily["cash"] + daily["stock_value"], atol=1.0)
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
    base_params = {"n_seeds": 1, "rebuild_days": 20, "train_window": 20,
                   "ic_window": 5, "warmup_days": 20,
                   "strict_check": True, "progress": False}

    res = run_daily_replay(panel, feat_cols,
                           params={**base_params, "collect_scores": True})
    fp = res["meta"]["score_fingerprint"]
    assert fp and isinstance(fp, str)
    cache = res["score_cache"]

    # 正确指纹 → 正常重放（跳过训练）
    res2 = run_daily_replay(
        panel, feat_cols,
        params={**base_params,
                "score_cache": {"scores": cache, "fingerprint": fp}})
    assert len(res2["daily"]) == len(res["daily"])
    assert res2["meta"]["used_cache"] is True

    # 错误指纹 → 显式拒绝（防陈旧缓存静默复用）
    with np.errstate(all="ignore"):
        try:
            run_daily_replay(
                panel, feat_cols,
                params={**base_params,
                        "score_cache": {"scores": cache, "fingerprint": "deadbeef"}})
        except ValueError as e:
            assert "指纹不匹配" in str(e)
        else:
            raise AssertionError("指纹不匹配的 score_cache 未被拒绝")


if __name__ == "__main__":
    test_backtest_synthetic_runs_with_strict_check()
    test_score_cache_fingerprint_validation()
    print("synthetic backtest test passed")


# --- pipeline steps_backtest 报告图（A3：滚动Sharpe / 月收益热力图 / 因子分层） ----
import json  # noqa: E402

from pipeline.context import StepContext  # noqa: E402
from pipeline import steps_backtest as sb  # noqa: E402


def _fake_replay_ctx(tmp_path, panel, n_days: int) -> StepContext:
    """伪造一个最小 replay 结果，直接驱动 performance 步骤（不跑重放）。"""
    dates = np.sort(panel["date"].unique())[-n_days:]
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.01, len(dates))
    tv = 1_000_000.0 * np.cumprod(1.0 + rets)
    daily = pd.DataFrame({
        "date": dates, "total_value": tv,
        "cash": tv * 0.3, "stock_value": tv * 0.7,
        "n_positions": 5, "n_fill": 1, "n_reject": 0,
        "turnover": 1000.0, "commission": 1.0, "stamp": 1.0, "impact": 0.5,
    })
    replay = {"daily": daily, "trades": pd.DataFrame(),
              "weights": pd.DataFrame(),
              "meta": {"initial_cash": 1_000_000.0, "elapsed_sec": 0.1}}
    return StepContext(session_dir=str(tmp_path), params={},
                       shared={"panel": panel, "replay": replay,
                               "factors": ["feature_00", "feature_01"]})


def test_performance_step_rolling_sharpe_and_monthly_heatmap(tmp_path):
    """≥63 交易日时：performance 报告包含滚动 Sharpe 线与月度收益热力图。"""
    panel = _make_panel(100)
    ctx = _fake_replay_ctx(tmp_path, panel, n_days=80)
    rep = sb.performance(ctx)
    titles = [f["title"] for f in rep["figures"]]
    assert any("滚动 63 日年化 Sharpe" in t for t in titles), titles
    hm = [f for f in rep["figures"] if f["type"] == "heatmap"]
    assert hm and "月度收益" in hm[0]["title"]
    assert len(hm[0]["z"]) >= 1 and len(hm[0]["z"][0]) == 12
    json.dumps(rep)  # 整个 report 必须可 JSON 序列化


def test_performance_step_short_window_skips_with_note(tmp_path):
    """<63 交易日时：跳过滚动 Sharpe，notes 里有提示且不炸。"""
    panel = _make_panel(40)
    ctx = _fake_replay_ctx(tmp_path, panel, n_days=30)
    rep = sb.performance(ctx)
    titles = [f["title"] for f in rep["figures"]]
    assert not any("滚动 63 日年化 Sharpe" in t for t in titles)
    assert any("滚动 Sharpe" in n for n in rep["notes"])
    json.dumps(rep)


def _make_layer_panel(n_dates: int = 60, n_syms: int = 10) -> pd.DataFrame:
    """带横截面区分度的多股票面板：因子与 target 相关，分层有意义。"""
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rng = np.random.default_rng(11)
    rows = []
    for d in dates:
        fa = rng.normal(0, 1, n_syms)
        fb = rng.normal(0, 1, n_syms)
        for i in range(n_syms):
            rows.append({
                "date": d, "symbol": f"S{i:03d}",
                "f_a": fa[i], "f_b": fb[i],
                C.TARGET: 0.01 * fa[i] + rng.normal(0, 0.02),
            })
    return pd.DataFrame(rows)


def test_factor_analysis_step_layer_equity_figures(tmp_path):
    """factor_analysis 为 |IC_t| Top 因子各画一张 Q1..Q5 分层累计净值图。"""
    panel = _make_layer_panel()
    ctx = StepContext(session_dir=str(tmp_path), params={},
                      shared={"panel": panel, "factors": ["f_a", "f_b"]})
    rep = sb.factor_analysis(ctx)
    layer_figs = [f for f in rep["figures"] if "因子分层累计净值" in f["title"]]
    assert layer_figs, [f["title"] for f in rep["figures"]]
    f0 = layer_figs[0]
    assert f0["type"] == "line"
    names = [s["name"] for s in f0["series"]]
    assert names == ["Q1", "Q2", "Q3", "Q4", "Q5"], names
    # 最高层强调色
    assert f0["series"][-1].get("color") == "#16a34a"
    json.dumps(rep)
