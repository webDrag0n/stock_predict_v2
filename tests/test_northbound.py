"""P0-3 北向持股因子的合成面板测试（防泄露回归防线）。

覆盖：
  - 披露滞后一日：T 日行只能用 T-1 日（含）之前披露的 hk_hold 数据；
  - 陆港通休市错开日的 ffill（上限 NORTHBOUND_MAX_FILL_DAYS）；
  - 派生因子 north_ratio_chg5 / north_vol_chg5 手工核算；
  - 非陆股通标的填 0 + north_covered=0；
  - USE_NORTHBOUND=0 时面板原样返回。
"""

from __future__ import annotations

import pandas as pd
import pytest

import config as C
from src.data import merge_northbound_features


class _FakeConnector:
    """runtime_checkable Protocol（SupportsNorthbound）只看方法存在与否。"""

    def __init__(self, hk: pd.DataFrame):
        self._hk = hk

    def fetch_hk_hold(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self._hk.copy()


DATES = pd.to_datetime(pd.date_range("2026-01-05", periods=8, freq="B"))  # 8 个工作日


def _panel(symbol_dtype: str = "object") -> pd.DataFrame:
    rows = []
    for sym in ["S1", "S2"]:
        for d in DATES:
            rows.append({"date": d, "symbol": sym, "close": 10.0})
    df = pd.DataFrame(rows)
    if symbol_dtype == "category":
        df["symbol"] = df["symbol"].astype("category")
    return df


def _hk() -> pd.DataFrame:
    """S1 在 d0..d5 有北向数据（ratio 1..6, vol 100..600），d6/d7 无（休市错开）。"""
    return pd.DataFrame(
        {
            "date": DATES[:6],
            "symbol": ["S1"] * 6,
            "north_vol": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "north_ratio": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _run(monkeypatch, panel: pd.DataFrame, fill_days: int = 2) -> pd.DataFrame:
    monkeypatch.setattr(C, "USE_NORTHBOUND", True)
    monkeypatch.setattr(C, "NORTHBOUND_MAX_FILL_DAYS", fill_days)
    return merge_northbound_features(panel, _FakeConnector(_hk()))


def _s1(out: pd.DataFrame) -> pd.DataFrame:
    return out[out["symbol"] == "S1"].sort_values("date").reset_index(drop=True)


def test_one_day_publication_lag(monkeypatch):
    """核心防泄露：T 日行的 north_ratio 等于 T-1 日披露的值，永远看不到当日。"""
    out = _run(monkeypatch, _panel())
    s1 = _s1(out)
    # shift(1) 后：d0 无可用数据，d1 见 d0 的 1.0，d6 见 d5 的 6.0
    assert s1.loc[0, "north_covered"] == 0
    assert s1.loc[0, "north_ratio"] == 0.0
    assert s1.loc[1, "north_ratio"] == pytest.approx(1.0)
    assert s1.loc[5, "north_ratio"] == pytest.approx(5.0)
    assert s1.loc[6, "north_ratio"] == pytest.approx(6.0)
    # 反例校验：若未滞后，d5 会看到当日的 6.0——必须是 d4 的 5.0
    assert s1.loc[5, "north_ratio"] != pytest.approx(6.0)


def test_ffill_covers_connect_holiday_gap(monkeypatch):
    """d6/d7 无 hk_hold 行（休市错开）：ffill(limit=2) 沿用到 d7，covered 仍=1。"""
    out = _run(monkeypatch, _panel(), fill_days=2)
    s1 = _s1(out)
    assert s1.loc[7, "north_ratio"] == pytest.approx(6.0)
    assert s1.loc[7, "north_covered"] == 1


def test_ffill_limit_expires(monkeypatch):
    """超过填充上限后视为无数据：fill_days=0 时 d7 回落到 0/covered=0。"""
    out = _run(monkeypatch, _panel(), fill_days=0)
    s1 = _s1(out)
    assert s1.loc[7, "north_ratio"] == 0.0
    assert s1.loc[7, "north_covered"] == 0


def test_derived_change_factors(monkeypatch):
    """chg5 手工核算：d6 行 ratio=6（滞后自 d5），5 日前（d1 行）=1 → chg5=5。"""
    out = _run(monkeypatch, _panel())
    s1 = _s1(out)
    assert s1.loc[6, "north_ratio_chg5"] == pytest.approx(6.0 - 1.0)
    # vol：d6 行滞后值 600，d1 行滞后值 100 → 600/100 - 1 = 5.0
    assert s1.loc[6, "north_vol_chg5"] == pytest.approx(600.0 / 100.0 - 1.0)
    # 前 6 行历史不足 5 日 → 填 0
    assert s1.loc[4, "north_ratio_chg5"] == 0.0


def test_uncovered_symbol_filled_zero(monkeypatch):
    """非陆股通标的（S2 无数据行）：数值列全 0，north_covered 恒 0。"""
    out = _run(monkeypatch, _panel())
    s2 = out[out["symbol"] == "S2"]
    assert (s2["north_ratio"] == 0.0).all()
    assert (s2["north_ratio_chg5"] == 0.0).all()
    assert (s2["north_vol_chg5"] == 0.0).all()
    assert (s2["north_covered"] == 0).all()
    # north_vol 原始量纲列不保留为特征
    assert "north_vol" not in out.columns


def test_categorical_symbol_merge(monkeypatch):
    """生产路径中 symbol 是 category dtype（load_data 降内存），合并必须兼容。"""
    out = _run(monkeypatch, _panel(symbol_dtype="category"))
    s1 = _s1(out)
    assert s1.loc[1, "north_ratio"] == pytest.approx(1.0)


def test_switch_off_passthrough(monkeypatch):
    """USE_NORTHBOUND=0：面板原样返回，不新增任何 north_* 列。"""
    monkeypatch.setattr(C, "USE_NORTHBOUND", False)
    df = _panel()
    out = merge_northbound_features(df, _FakeConnector(_hk()))
    assert list(out.columns) == list(df.columns)


def test_empty_hk_hold_passthrough(monkeypatch):
    """连接器返回空表（区间无北向数据）时面板原样返回。"""
    monkeypatch.setattr(C, "USE_NORTHBOUND", True)
    df = _panel()
    empty = pd.DataFrame(columns=["date", "symbol", "north_vol", "north_ratio"])
    out = merge_northbound_features(df, _FakeConnector(empty))
    assert list(out.columns) == list(df.columns)


def test_fetch_failure_passthrough(monkeypatch):
    """连接器抛异常（限速/网络）时降级为跳过该因子族，不中断管线。"""

    class _Boom:
        def fetch_hk_hold(self, start_date: str, end_date: str) -> pd.DataFrame:
            raise RuntimeError("频率超限")

    monkeypatch.setattr(C, "USE_NORTHBOUND", True)
    df = _panel()
    out = merge_northbound_features(df, _Boom())
    assert list(out.columns) == list(df.columns)


def test_fetch_uses_sh_sz_exchanges(tmp_path):
    """回归防线：hk_hold 必须显式按 exchange='SH'/'SZ' 拉北向。

    实测确认：不传 exchange 时 hk_hold 返回南向数据（exchange=HK，
    ts_code 形如 00001.HK），与 A 股面板零交集，会产生静默无效特征。
    """
    from src.data.tushare import TushareConnector

    conn = TushareConnector.__new__(TushareConnector)  # 跳过 __init__ 的 token 检查
    conn._cache_dir = str(tmp_path)
    calls = []

    class _Pro:
        def hk_hold(self, trade_date=None, exchange=None):
            calls.append(exchange)
            return pd.DataFrame(
                {
                    "trade_date": [trade_date],
                    "ts_code": [f"600000.{exchange}"],
                    "vol": [100],
                    "ratio": [1.5],
                }
            )

    conn._pro = _Pro()
    out = conn._fetch_hk_hold_one_date("20261016")
    assert sorted(calls) == ["SH", "SZ"]
    assert set(out["symbol"]) == {"600000.SH", "600000.SZ"}
    # 第二遍应纯走 per-exchange 缓存，不再调 API
    calls.clear()
    out2 = conn._fetch_hk_hold_one_date("20261016")
    assert calls == []
    assert set(out2["symbol"]) == {"600000.SH", "600000.SZ"}
