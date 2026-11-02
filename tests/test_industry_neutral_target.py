"""P0-2 行业中性化预测目标的合成面板手工核算测试。

覆盖：
  - INDUSTRY_NEUTRAL=1 时标签 = 远期收益 − 当日同行业均值（手工核算）；
  - 行业当日有效样本数 < INDUSTRY_MIN_COUNT 时回退全市场均值；
  - INDUSTRY_NEUTRAL=1 隐含截面中性（market_neutral_target=False 也去均值）；
  - INDUSTRY_NEUTRAL=0 时退回全市场均值 / 原始收益的旧行为。
"""

from __future__ import annotations

import pandas as pd
import pytest

import config as C
from src.data import prepare_target_and_weight


def _panel() -> pd.DataFrame:
    """7 只股票 × 3 日：X 行业 3 只、Y 行业 3 只、Z 行业 1 只（小行业）。

    d0 → d1 的远期收益（fw=1，用 close）手工值：
      A1 0.10, A2 0.05, A3 -0.05   (X 行业均值 = 0.10/3 ≈ 0.033333)
      B1 0.02, B2 0.00, B3 -0.02   (Y 行业均值 = 0.0)
      C1 0.20                      (Z 行业仅 1 只 → 回退全市场均值)
      全市场均值 = (0.10+0.05-0.05+0.02+0.00-0.02+0.20)/7 = 0.30/7 ≈ 0.042857
    """
    closes = {
        "A1": [100.0, 110.0, 110.0],
        "A2": [100.0, 105.0, 105.0],
        "A3": [100.0, 95.0, 95.0],
        "B1": [100.0, 102.0, 102.0],
        "B2": [100.0, 100.0, 100.0],
        "B3": [100.0, 98.0, 98.0],
        "C1": [100.0, 120.0, 120.0],
    }
    industry = {s: ind for s, ind in zip(closes, ["X"] * 3 + ["Y"] * 3 + ["Z"])}
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    rows = []
    for sym, px in closes.items():
        for d, c in zip(dates, px):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "close": c,
                    "industry": industry[sym],
                    "volatility_20": 0.02,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


# 手工核算的 d0 目标值
IND_MEAN_X = (0.10 + 0.05 - 0.05) / 3
MKT_MEAN = 0.30 / 7
EXPECTED_INDUSTRY_D0 = {
    "A1": 0.10 - IND_MEAN_X,
    "A2": 0.05 - IND_MEAN_X,
    "A3": -0.05 - IND_MEAN_X,
    "B1": 0.02 - 0.0,
    "B2": 0.00 - 0.0,
    "B3": -0.02 - 0.0,
    "C1": 0.20 - MKT_MEAN,  # 小行业回退全市场均值
}
EXPECTED_MARKET_D0 = {
    "A1": 0.10 - MKT_MEAN,
    "A2": 0.05 - MKT_MEAN,
    "A3": -0.05 - MKT_MEAN,
    "B1": 0.02 - MKT_MEAN,
    "B2": 0.00 - MKT_MEAN,
    "B3": -0.02 - MKT_MEAN,
    "C1": 0.20 - MKT_MEAN,
}
EXPECTED_RAW_D0 = {
    "A1": 0.10,
    "A2": 0.05,
    "A3": -0.05,
    "B1": 0.02,
    "B2": 0.00,
    "B3": -0.02,
    "C1": 0.20,
}


def _targets_at_d0(df: pd.DataFrame) -> dict:
    d0 = df["date"].min()
    sub = df[df["date"] == d0]
    return dict(zip(sub["symbol"], sub["responder_6"]))


def _run(monkeypatch, industry_neutral: bool, market_neutral: bool) -> pd.DataFrame:
    monkeypatch.setattr(C, "INDUSTRY_NEUTRAL", industry_neutral)
    monkeypatch.setattr(C, "INDUSTRY_MIN_COUNT", 2)
    monkeypatch.setattr(C, "MULTI_HORIZON_WINDOWS", [])
    monkeypatch.setattr(C, "RESIDUAL_TARGET", False)
    return prepare_target_and_weight(
        _panel(), forward_window=1, market_neutral_target=market_neutral
    )


def test_industry_neutral_hand_computed(monkeypatch):
    """两者同开：减同行业均值，小行业（样本<2）回退全市场均值。"""
    got = _targets_at_d0(_run(monkeypatch, industry_neutral=True, market_neutral=True))
    for sym, exp in EXPECTED_INDUSTRY_D0.items():
        assert got[sym] == pytest.approx(exp, abs=1e-9), sym


def test_industry_neutral_implies_cross_sectional(monkeypatch):
    """INDUSTRY_NEUTRAL=1 隐含截面中性：market_neutral_target=False 也去行业均值。"""
    got = _targets_at_d0(_run(monkeypatch, industry_neutral=True, market_neutral=False))
    for sym, exp in EXPECTED_INDUSTRY_D0.items():
        assert got[sym] == pytest.approx(exp, abs=1e-9), sym


def test_market_neutral_only(monkeypatch):
    """INDUSTRY_NEUTRAL=0 且 market_neutral：全部减全市场均值（旧行为不变）。"""
    got = _targets_at_d0(_run(monkeypatch, industry_neutral=False, market_neutral=True))
    for sym, exp in EXPECTED_MARKET_D0.items():
        assert got[sym] == pytest.approx(exp, abs=1e-9), sym


def test_raw_target_when_both_off(monkeypatch):
    """两者均关：原始远期收益。"""
    got = _targets_at_d0(_run(monkeypatch, industry_neutral=False, market_neutral=False))
    for sym, exp in EXPECTED_RAW_D0.items():
        assert got[sym] == pytest.approx(exp, abs=1e-9), sym


def test_industry_neutral_without_labels_falls_back_to_market(monkeypatch):
    """面板无 industry 列时，INDUSTRY_NEUTRAL=1 回退全市场中性而非报错。"""
    monkeypatch.setattr(C, "INDUSTRY_NEUTRAL", True)
    monkeypatch.setattr(C, "MULTI_HORIZON_WINDOWS", [])
    monkeypatch.setattr(C, "RESIDUAL_TARGET", False)
    df = _panel().drop(columns=["industry"])
    out = prepare_target_and_weight(df, forward_window=1, market_neutral_target=False)
    got = _targets_at_d0(out)
    for sym, exp in EXPECTED_MARKET_D0.items():
        assert got[sym] == pytest.approx(exp, abs=1e-9), sym
