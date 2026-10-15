"""防泄露专项测试（阶段 1 修复的回归防线）。

覆盖：
  1.2 财报披露滞后：年报（period 尾号 1231）+120 天，其余 +62 天；
  1.4 config._validate 的 embargo 硬断言（含 multi_* 目标的最长 horizon）；
  1.5 list_factor_columns 按 responder_fw_ 前缀排除多 horizon 标签。
"""
from __future__ import annotations

import pandas as pd
import pytest

import config as C
from src.data import list_factor_columns
from src.data.tushare import _announcement_dates


# --------------------------------------------------------------------------
# 1.2 财报披露滞后
# --------------------------------------------------------------------------
def test_annual_report_lag_120_days():
    """年报披露滞后 120 天：2023 年报在 2024-04-29 才可见（4 月前不可见）。"""
    period = pd.Series(["20231231"])
    period_dt = pd.to_datetime(period, format="%Y%m%d")
    avail = _announcement_dates(period, period_dt)
    assert avail.iloc[0] == pd.Timestamp("2023-12-31") + pd.Timedelta(days=120)
    # 4 月 1 日（年报真实披露高峰前）该年报尚不可见
    assert avail.iloc[0] > pd.Timestamp("2024-04-01")


def test_non_annual_report_lag_62_days():
    """一季报/半年报/三季报维持 +62 天。"""
    period = pd.Series(["20230331", "20230630", "20230930"])
    period_dt = pd.to_datetime(period, format="%Y%m%d")
    avail = _announcement_dates(period, period_dt)
    expected = period_dt + pd.Timedelta(days=62)
    pd.testing.assert_series_equal(
        avail.reset_index(drop=True), expected.reset_index(drop=True))


# --------------------------------------------------------------------------
# 1.4 embargo 硬断言
# --------------------------------------------------------------------------
def test_validate_rejects_embargo_below_forward_window(monkeypatch):
    """EMBARGO_DAYS < FORWARD_WINDOW 必须硬失败（前视泄露底线）。"""
    monkeypatch.setattr(C, "EMBARGO_DAYS", C.FORWARD_WINDOW - 1)
    with pytest.raises(AssertionError, match="EMBARGO_DAYS"):
        C._validate()


def test_validate_rejects_multi_objective_with_short_embargo(monkeypatch):
    """multi_* 目标要求 EMBARGO_DAYS >= max(MULTI_HORIZON_WINDOWS)。"""
    monkeypatch.setattr(C, "MODEL_OBJECTIVE", "multi_ret_weighted")
    monkeypatch.setattr(C, "MULTI_HORIZON_WINDOWS", [5, 10, 60])
    monkeypatch.setattr(C, "EMBARGO_DAYS", C.FORWARD_WINDOW)  # 20 < 60
    with pytest.raises(AssertionError, match="multi"):
        C._validate()


def test_validate_passes_with_current_config():
    """当前仓库配置必须通过校验（防回归：默认值本身不能触发断言）。"""
    C._validate()


# --------------------------------------------------------------------------
# 1.5 list_factor_columns 前缀排除
# --------------------------------------------------------------------------
def test_list_factor_columns_excludes_any_responder_fw():
    """任意 horizon 的 responder_fw_* 标签都不得混入特征（含未硬编码的 42）。"""
    panel = pd.DataFrame({
        "date": ["2024-01-02"],
        "symbol": ["000001.SZ"],
        "responder_6": [0.01],
        "responder_fw_5": [0.01],
        "responder_fw_42": [0.01],   # 不在旧硬编码名单里的 horizon
        "weight": [1.0],
        "close": [10.0],
        "real_factor": [0.5],
    })
    cols = list_factor_columns(panel)
    assert cols == ["real_factor"]


# --------------------------------------------------------------------------
# 引擎层防泄露：walkforward embargo 硬断言 + daily replay 成熟标签截断
# --------------------------------------------------------------------------
def test_walkforward_engine_rejects_short_embargo(monkeypatch):
    """run_walkforward 引擎层防御性断言：embargo < FORWARD_WINDOW 必须硬失败
    （即使 config._validate 被绕过/配置被动态篡改）。"""
    from walkforward import run_walkforward

    monkeypatch.setattr(C, "EMBARGO_DAYS", C.FORWARD_WINDOW - 1)
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "symbol": ["000001.SZ", "600000.SH"],
        "f0": [0.1, 0.2],
        C.TARGET: [0.01, -0.01],
    })
    with pytest.raises(AssertionError, match="前视泄露"):
        run_walkforward(panel, ["f0"], verbose=False)


def _make_trade_panel(n_dates: int = 80) -> pd.DataFrame:
    """与 test_backtest_synthetic 同款的两股小面板（含撮合所需交易列）。"""
    import numpy as np
    dates = pd.date_range("2026-05-01", periods=n_dates, freq="B")
    rows = []
    rng = np.random.RandomState(42)
    for d in dates:
        for sym in ["600000.SH", "300001.SZ"]:
            close = 10.0 + rng.randn() * 0.5
            rows.append({
                "date": d, "symbol": sym, "close": close, "adj_close": close,
                "open": close * (1 + rng.randn() * 0.01),
                "high": close * 1.02, "low": close * 0.98,
                "amount": 10000.0,
                "pre_close": close * (1 + rng.randn() * 0.01),
                "pct_chg": rng.randn() * 2.0, "limit_flag": 0,
                "return_1d": rng.randn() * 0.02,
                "bb_upper_20": close * 1.05, "bb_lower_20": close * 0.95,
                "rsi_14": 50.0,
                C.TARGET: rng.randn() * 0.02, C.WEIGHT: 1.0,
                "feature_00": rng.randn(), "feature_01": rng.randn(),
            })
    return pd.DataFrame(rows)


def test_daily_replay_immature_labels_do_not_change_scores():
    """防泄露专项：篡改整个回测期间都不会成熟的未来标签（尾部 FORWARD_WINDOW 天
    的 responder_6 改成极端值），逐日 OOS 分数必须逐位不变。

    若引擎把未成熟标签泄露进训练/定权，被污染日的分数会剧烈漂移。
    """
    import numpy as np
    from backtest_daily import run_daily_replay

    panel = _make_trade_panel(80)
    feats = ["feature_00", "feature_01"]
    params = {"n_seeds": 1, "rebuild_days": 20, "train_window": 20,
              "ic_window": 5, "warmup_days": 20, "strict_check": True,
              "progress": False, "collect_scores": True}

    res1 = run_daily_replay(panel, feats, params=dict(params))

    fw = C.FORWARD_WINDOW
    panel2 = panel.copy()
    future_dates = np.sort(panel2["date"].unique())[-fw:]
    panel2.loc[panel2["date"].isin(future_dates), C.TARGET] = 999.0
    res2 = run_daily_replay(panel2, feats, params=dict(params))

    sc1, sc2 = res1["score_cache"], res2["score_cache"]
    assert len(sc1) > 0 and set(sc1) == set(sc2)
    for d in sc1:
        assert set(sc1[d]) == set(sc2[d])
        for sym, v in sc1[d].items():
            assert v == pytest.approx(sc2[d][sym], rel=1e-9, abs=1e-12), (
                f"未来标签泄露：{pd.Timestamp(d).date()} {sym} 分数 "
                f"{v} → {sc2[d][sym]}")
