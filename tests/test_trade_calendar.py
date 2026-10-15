"""交易日历防截断回归测试。

2026-08-21 事故：`_trade_dates` 用 daily_*.parquet 缓存派生日历时未校验覆盖
范围，缓存只到 2026-07-01 时静默截断了 7 周数据且 0 次新 API 调用。修复后：
派生日历明显早于请求 end 时必须回退 trade_cal API。
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.tushare import TushareConnector


def _bare_connector(tmp_path):
    """绕过 __init__（需要 token/网络），只挂测试需要的属性。"""
    conn = TushareConnector.__new__(TushareConnector)
    conn._cache_dir = str(tmp_path)
    conn._universe_cache = {}
    return conn


class _FakePro:
    """记录 trade_cal 是否被调用，并返回覆盖到请求末端的日历。"""

    def __init__(self):
        self.calls = []

    def trade_cal(self, exchange, start_date, end_date, is_open, fields):
        self.calls.append((start_date, end_date))
        # 每个自然日都算交易日（测试只需覆盖范围语义）
        days = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame({"cal_date": days.strftime("%Y%m%d")})


def _write_daily_cache(tmp_path, dates):
    for d in dates:
        pd.DataFrame({"date": [pd.Timestamp(d)]}).to_parquet(
            tmp_path / f"daily_{d.replace('-', '')}.parquet")


def test_stale_daily_cache_falls_back_to_api(tmp_path):
    """daily 缓存只到 01-05，请求到 01-20：必须调 trade_cal API 而不是截断。"""
    conn = _bare_connector(tmp_path)
    conn._pro = _FakePro()
    _write_daily_cache(tmp_path, ["2026-01-02", "2026-01-05"])

    dates = conn._trade_dates("2026-01-01", "2026-01-20")

    assert conn._pro.calls, "缓存日历时滞时应回退 trade_cal API"
    assert dates[-1] == "20260120", f"日历被截断到 {dates[-1]}"


def test_fresh_daily_cache_avoids_api(tmp_path):
    """daily 缓存覆盖到请求末端（容差 10 个自然日）时不调 API。"""
    conn = _bare_connector(tmp_path)
    conn._pro = _FakePro()
    _write_daily_cache(tmp_path, ["2026-01-02", "2026-01-05", "2026-01-18"])

    dates = conn._trade_dates("2026-01-01", "2026-01-20")

    assert not conn._pro.calls, "缓存新鲜时不应调 trade_cal API"
    assert dates == ["20260102", "20260105", "20260118"]


def test_poisoned_exact_cache_rebuilt_via_api(tmp_path):
    """exact 文件名命中但内容被截断（历史派生路径污染）→ 弃用并走 API。"""
    conn = _bare_connector(tmp_path)
    conn._pro = _FakePro()
    # 模拟污染：文件名声称覆盖到 2026-01-20，内容只到 2026-01-05
    pd.DataFrame({"cal_date": ["20260102", "20260105"]}).to_parquet(
        tmp_path / "tradecal_20260101_20260120.parquet")

    dates = conn._trade_dates("2026-01-01", "2026-01-20")

    assert conn._pro.calls, "污染的 exact 缓存应被弃用并回退 API"
    assert dates[-1] == "20260120"


def test_tushare_rate_limit_falls_back_to_akshare(tmp_path, monkeypatch):
    """tushare trade_cal 限流（免费档 1 次/小时）→ akshare 日历兜底。"""
    import akshare as ak

    class _RateLimitedPro:
        def trade_cal(self, **kw):
            raise Exception("抱歉，您访问接口(trade_cal)频率超限(1次/小时)")

    days = pd.date_range("2026-01-02", "2026-01-20", freq="B")  # 工作日即可
    monkeypatch.setattr(ak, "tool_trade_date_hist_sina",
                        lambda: pd.DataFrame({"trade_date": days}))

    conn = _bare_connector(tmp_path)
    conn._pro = _RateLimitedPro()

    dates = conn._trade_dates("2026-01-01", "2026-01-20")

    assert dates[0] == "20260102" and dates[-1] == "20260120"
    # 兜底结果应写缓存，下次不再触发任何 API
    dates2 = conn._trade_dates("2026-01-01", "2026-01-20")
    assert dates2 == dates
