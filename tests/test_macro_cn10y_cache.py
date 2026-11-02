"""10 年期国债收益率（cn_10y）按日增量缓存测试。

使用 fake akshare（不触网）验证：
  - 首次抓取全量并写缓存；
  - end 前移一天只补抓缺失的尾部区间（日常增量场景，不再每晚全量重抓）；
  - 精确命中 / 已被覆盖的子区间不发起网络请求；
  - 增量抓取失败时回退旧缓存并输出 warning。
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import config as C
from src.data import macro_features as mf


class FakeAkshare:
    """记录调用区间的 bond_china_yield 假实现。"""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.raise_ = False

    def bond_china_yield(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append((start_date, end_date))
        if self.raise_:
            raise RuntimeError("network down")
        days = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="B")
        return pd.DataFrame(
            {
                "曲线名称": ["中债国债收益率曲线"] * len(days),
                "日期": days.strftime("%Y-%m-%d"),
                "10年": [2.5 + 0.001 * i for i in range(len(days))],
            }
        )


@pytest.fixture()
def fake_akshare(monkeypatch, tmp_path):
    fake = FakeAkshare()
    monkeypatch.setattr(C, "CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "akshare", fake)
    monkeypatch.setattr(mf, "time", types.SimpleNamespace(sleep=lambda *_: None))
    return fake


def test_first_fetch_full_range(fake_akshare):
    s = mf._fetch_cn_10y_yield("20240101", "20240630")
    assert fake_akshare.calls == [("20240101", "20240630")]
    assert len(s) > 0


def test_incremental_fetch_only_tail(fake_akshare):
    s1 = mf._fetch_cn_10y_yield("20240101", "20240630")
    fake_akshare.calls.clear()
    s2 = mf._fetch_cn_10y_yield("20240101", "20240703")
    assert len(fake_akshare.calls) == 1
    cs, _ = fake_akshare.calls[0]
    assert pd.to_datetime(cs) > s1.index.max()  # 只补抓缓存之后的尾部
    assert len(s2) > len(s1)
    assert s2.loc[: s1.index[-1]].equals(s1)  # 既有数据不变


def test_exact_hit_and_covered_subrange_no_network(fake_akshare):
    mf._fetch_cn_10y_yield("20240101", "20240630")
    fake_akshare.calls.clear()
    mf._fetch_cn_10y_yield("20240101", "20240630")
    assert not fake_akshare.calls
    s = mf._fetch_cn_10y_yield("20240201", "20240531")  # 已被覆盖的子区间
    assert not fake_akshare.calls
    assert s.index.min() >= pd.Timestamp("2024-02-01")


def test_fetch_failure_falls_back_to_stale_cache(fake_akshare, caplog):
    s1 = mf._fetch_cn_10y_yield("20240101", "20240630")
    fake_akshare.raise_ = True
    fake_akshare.calls.clear()
    with caplog.at_level("WARNING", logger="src.data.macro_features"):
        s2 = mf._fetch_cn_10y_yield("20240101", "20240703")
    assert fake_akshare.calls  # 尝试过增量抓取
    assert s2.equals(s1)  # 回退到旧缓存
    assert any("回退旧缓存" in r.message for r in caplog.records)
