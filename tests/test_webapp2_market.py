"""行情页：涨跌停按板块幅度（与交易引擎一致）、单股行情补齐逐日文件尾部、宽度后台计算不阻塞。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from webapp2 import app as appmod
from webapp2 import data
from webapp2 import market_logic as ml
from webapp2.pages import market

from tests.test_webapp2_pages import ids_of, text_of  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    data.refresh()
    ml.reset_caches()
    yield tmp_path
    data.refresh()
    ml.reset_caches()


COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "pct_chg"]


def day_frame(day: str, rows: list[tuple]) -> pd.DataFrame:
    """rows: (symbol, close, pct_chg)。"""
    return pd.DataFrame(
        [[pd.Timestamp(day), s, c, c * 1.01, c * 0.99, c, 1000.0, 5000.0, p] for s, c, p in rows],
        columns=COLS,
    )


def put_days(sandbox, days: dict[str, list[tuple]]):
    d = sandbox / "data_cache" / "tushare"
    d.mkdir(parents=True, exist_ok=True)
    for day, rows in days.items():
        day_frame(day, rows).to_parquet(d / f"daily_{day.replace('-', '')}.parquet")
    data.refresh()
    ml.reset_caches()


# ----------------------------------------------------------------------------- 涨跌停口径
def test_board_limits_match_the_trading_engine_for_every_prefix():
    """行情页向量化/缓存的实现必须与交易引擎 _board_limit_pct 逐前缀一致。"""
    from trading import _board_limit_pct

    codes = [
        "000001.SZ",
        "002001.SZ",
        "600000.SH",
        "601318.SH",
        "603001.SH",
        "300001.SZ",
        "301001.SZ",
        "688001.SH",
        "689009.SH",
        "430047.BJ",
        "830799.BJ",
        "870001.BJ",
        "920001.BJ",
    ]
    got = ml.limit_pcts(pd.Series(codes))
    for code, g in zip(codes, got):
        assert g == pytest.approx(_board_limit_pct(code) * 100), code


def test_day_stats_uses_board_limits_not_a_flat_9_9():
    """创业板 +12% 不是涨停；旧行情浏览器统一按 ±9.9% 会把它算进去（实测 11-13 多算 5 只涨停、32 只跌停）。"""
    df = day_frame(
        "2026-11-13",
        [
            ("600000.SH", 10, 10.0),  # 主板涨停
            ("600001.SH", 10, -10.0),  # 主板跌停
            ("300001.SZ", 10, 12.0),  # 创业板 +12%：不是涨停
            ("300002.SZ", 10, 19.9),  # 创业板涨停
            ("688001.SH", 10, -15.0),  # 科创板 -15%：不是跌停
            ("688002.SH", 10, -19.8),  # 科创板跌停
            ("830001.BJ", 10, 29.9),  # 北交所涨停
            ("000001.SZ", 10, 0.0),
            ("000002.SZ", 10, 2.0),
            ("000003.SZ", 10, np.nan),  # 缺失不计入任何类别
        ],
    )
    s = ml.day_stats(df)
    assert s["limit_up"] == 3 and s["limit_down"] == 2
    # 上涨 5：+10 / +12 / +19.9 / +29.9 / +2；下跌 3：-10 / -15 / -19.8；持平 1；NaN 不计
    assert (s["n"], s["advancers"], s["decliners"], s["unchanged"]) == (9, 5, 3, 1)
    # 旧口径对照：统一 9.9% 会数出更多
    pc = df["pct_chg"]
    assert int((pc >= 9.9).sum()) > s["limit_up"] and int((pc <= -9.9).sum()) > s["limit_down"]


def test_day_stats_tolerance_matches_engine_half_point():
    just_below = day_frame("2026-11-13", [("600000.SH", 10, 9.49)])
    at_tol = day_frame("2026-11-13", [("600000.SH", 10, 9.5)])
    assert ml.day_stats(just_below)["limit_up"] == 0 and ml.day_stats(at_tol)["limit_up"] == 1


# ----------------------------------------------------------------------------- 宽度（后台）
def test_breadth_is_computed_in_background_and_never_blocks(sandbox, monkeypatch):
    put_days(
        sandbox, {"2026-11-12": [("600000.SH", 10, 1.0)], "2026-11-13": [("600000.SH", 10, -1.0)]}
    )
    import threading

    gate = threading.Event()
    real = ml._compute_breadth

    def slow(files):
        gate.wait(5)
        return real(files)

    monkeypatch.setattr(ml, "_compute_breadth", slow)
    t = time.time()
    assert ml.breadth() is None  # 还在算：立即返回 None，不等待
    assert time.time() - t < 0.5
    assert ml.breadth() is None  # 第二次也不会再起第二个线程
    assert len(ml._breadth_jobs) == 1
    gate.set()
    for _ in range(100):
        out = ml.breadth()
        if out is not None:
            break
        time.sleep(0.05)
    assert out is not None and list(out["advancers"]) == [1, 0] and list(out["decliners"]) == [0, 1]
    assert ml._breadth_jobs == {}  # 任务结束后清理


def test_breadth_failure_does_not_leave_the_page_spinning_forever(sandbox, monkeypatch):
    put_days(sandbox, {"2026-11-13": [("600000.SH", 10, 1.0)]})

    def boom(files):
        raise RuntimeError("读文件炸了")

    monkeypatch.setattr(ml, "_compute_breadth", boom)
    assert ml.breadth() is None
    for _ in range(100):
        out = ml.breadth()
        if out is not None:
            break
        time.sleep(0.05)
    assert out is not None and out.empty  # 失败落一个空表，页面显示「没有行情缓存」而不是永远转圈
    assert ml._breadth_jobs == {}


def test_breadth_recomputes_when_a_new_trading_day_arrives(sandbox):
    """新交易日由夜间任务写进 data_cache，页面进程不会收到通知：必须靠文件 mtime 键发现变化。
    注意不能用 put_days 添加第二天——它会顺手清缓存，掩盖「不重算」的缺陷。"""
    put_days(sandbox, {"2026-11-12": [("600000.SH", 10, 1.0)]})
    assert len(ml.breadth(block=True)) == 1
    day_frame("2026-11-13", [("600000.SH", 10, 2.0)]).to_parquet(
        sandbox / "data_cache" / "tushare" / "daily_20261113.parquet"
    )  # 只写文件，不清任何缓存
    assert len(ml.breadth(block=True)) == 2
    assert len(ml._breadth_cache) == 1  # 旧键被清掉，不累积


def test_breadth_recomputes_when_an_existing_day_file_is_rewritten(sandbox):
    put_days(sandbox, {"2026-11-13": [("600000.SH", 10, 1.0)]})
    assert ml.breadth(block=True).iloc[0]["advancers"] == 1
    path = sandbox / "data_cache" / "tushare" / "daily_20261113.parquet"
    day_frame("2026-11-13", [("600000.SH", 10, -1.0)]).to_parquet(path)
    import os

    st = os.stat(path)
    os.utime(path, (st.st_atime + 5, st.st_mtime + 5))  # 同一秒内重写时 mtime 可能不变，显式推进
    assert ml.breadth(block=True).iloc[0]["decliners"] == 1


def test_breadth_without_any_cache_is_empty_not_none(sandbox):
    assert ml.breadth().empty


# ----------------------------------------------------------------------------- 单股行情
def put_bars_long(sandbox, symbol, days):
    d = sandbox / "data_cache"
    d.mkdir(parents=True, exist_ok=True)
    pd.concat([day_frame(day, [(symbol, c, 0.0)]) for day, c in days]).to_parquet(
        d / "wholemkt_bars_long.parquet"
    )
    data.refresh()
    ml.reset_caches()


def test_symbol_bars_are_topped_up_from_daily_files_newer_than_the_consolidated_file(sandbox):
    """真实情形：合并文件止于 11-06，逐日缓存已到 11-13。不补齐，图表会悄悄少一周。"""
    put_bars_long(sandbox, "000001.SZ", [("2026-11-05", 10.0), ("2026-11-06", 10.5)])
    put_days(
        sandbox,
        {
            "2026-11-06": [("000001.SZ", 10.5, 0.0)],
            "2026-11-09": [("000001.SZ", 11.0, 0.0)],
            "2026-11-10": [("000001.SZ", 11.5, 0.0)],
        },
    )
    b = ml.symbol_bars("000001.SZ", 0)
    assert [d.strftime("%m-%d") for d in b["date"]] == ["11-05", "11-06", "11-09", "11-10"]
    assert not b["date"].duplicated().any()  # 11-06 同时在两处，只保留一份
    assert list(b["close"]) == [10.0, 10.5, 11.0, 11.5]


def test_symbol_bars_without_consolidated_file_read_everything_from_daily_files(sandbox):
    put_days(
        sandbox,
        {"2026-11-05": [("000001.SZ", 10.0, 0.0)], "2026-11-06": [("000001.SZ", 10.5, 0.0)]},
    )
    assert len(ml.symbol_bars("000001.SZ", 0)) == 2
    assert ml.symbol_bars("999999.SZ", 0).empty and ml.symbol_bars("000001.SZ", 1).shape[0] == 1


def test_symbol_bars_window_and_stale_note(sandbox):
    days = {
        f"2026-11-{d:02d}": [("000001.SZ", 10.0 + d, 0.0), ("600000.SH", 5.0, 0.0)]
        for d in range(2, 8)
    }
    days["2026-11-09"] = [("600000.SH", 5.0, 0.0)]  # 000001 停牌
    put_days(sandbox, days)
    assert len(ml.symbol_bars("000001.SZ", 3)) == 3
    note = ml.stale_note(ml.bars_last_date("000001.SZ"), ml.latest_day())
    assert note and "2026-11-07" in note and "早 2 天" in note
    assert ml.stale_note(ml.bars_last_date("600000.SH"), ml.latest_day()) is None
    assert ml.stale_note(None, "20261109") is None


def test_indicators():
    c = pd.Series(np.arange(1.0, 61.0))
    assert ml.rsi(c).iloc[-1] == pytest.approx(100.0, abs=0.01)  # 一路上涨 → RSI≈100
    ma, up, dn = ml.bollinger(c)
    assert up.iloc[-1] > ma.iloc[-1] > dn.iloc[-1] and np.isnan(ma.iloc[5])
    dif, dea, hist = ml.macd(c)
    assert hist.iloc[-1] == pytest.approx(dif.iloc[-1] - dea.iloc[-1])


def test_search_symbols_prefers_prefix_matches(sandbox):
    put_days(
        sandbox,
        {
            "2026-11-13": [
                ("600000.SH", 10, 0.0),
                ("000600.SZ", 10, 0.0),
                ("300600.SZ", 10, 0.0),
                ("000001.SZ", 10, 0.0),
            ]
        },
    )
    assert ml.search_symbols("600")[0] == "600000.SH"  # 前缀匹配排在子串匹配之前
    assert set(ml.search_symbols("600")) == {"600000.SH", "000600.SZ", "300600.SZ"}
    assert ml.search_symbols("zzz") == [] and len(ml.search_symbols("", 2)) == 2


# ----------------------------------------------------------------------------- 页面
def test_page_without_cache_shows_guidance(sandbox):
    assert "没有行情缓存" in text_of(market.layout())


def test_page_renders_stock_and_breadth_cards(sandbox):
    put_days(
        sandbox,
        {
            "2026-11-12": [("000001.SZ", 10, 1.0), ("600000.SH", 5, 10.0)],
            "2026-11-13": [("000001.SZ", 10.5, 2.0), ("600000.SH", 5, -10.0)],
        },
    )
    ml.breadth(block=True)
    t = text_of(market.layout("?symbol=600000.SH"))
    for needle in ("行情", "数据到 20261113", "最新交易日", "涨停 / 跌停", "按板块幅度"):
        assert needle in t, needle
    assert "mk-symbol" in ids_of(market.layout())


def test_page_shows_calculating_state_before_breadth_is_ready(sandbox, monkeypatch):
    put_days(sandbox, {"2026-11-13": [("000001.SZ", 10, 1.0)]})
    monkeypatch.setattr(ml, "breadth", lambda block=False: None)
    assert "正在计算" in text_of(market._breadth_cards())


def test_symbol_from_search_validates_against_the_universe(sandbox):
    put_days(sandbox, {"2026-11-13": [("000001.SZ", 10, 0.0)]})
    assert market.symbol_from_search("?symbol=000001.SZ") == "000001.SZ"
    for bad in (None, "", "?symbol=NOPE.SZ", "?symbol=../../etc", "?x=1"):
        assert market.symbol_from_search(bad) is None


def test_candle_figure_has_requested_subplots(sandbox):
    put_days(
        sandbox, {f"2026-11-{d:02d}": [("000001.SZ", 10.0 + d % 3, 0.0)] for d in range(2, 12)}
    )
    base = market.candle_fig("000001.SZ", [], 0)
    full = market.candle_fig("000001.SZ", ["ma", "boll", "rsi", "macd"], 0)
    names = {t.name for t in full.data}
    assert {"MA20", "MA60", "布林上轨", "RSI", "DIF", "DEA"} <= names
    assert len(full.data) > len(base.data) and full.layout.height > base.layout.height
    assert "没有行情数据" in market.candle_fig("999999.SZ", [], 0).layout.annotations[0].text


def test_routing_passes_url_search_to_the_page(sandbox):
    put_days(sandbox, {"2026-11-13": [("000001.SZ", 10, 0.0), ("600000.SH", 5, 0.0)]})
    assert ("/market", "行情") in appmod.NAV
    page = appmod.route("/market", "?symbol=600000.SH")
    assert "mk-symbol" in ids_of(page)
    dd = [c for c in _walk(page) if getattr(c, "id", None) == "mk-symbol"][0]
    assert dd.value == "600000.SH"  # 从交易页「代码」链接跳过来时选中该股


def _walk(n):
    if n is None or isinstance(n, (str, int, float)):
        return
    if isinstance(n, (list, tuple)):
        for c in n:
            yield from _walk(c)
        return
    yield n
    yield from _walk(getattr(n, "children", None))


def test_selected_symbol_outside_the_first_page_of_options_is_still_selectable(sandbox):
    """回归（浏览器里发现）：/market?symbol=300750.SZ 下拉框为空、页面显示「行情缓存为空」。
    全市场 5500+ 只，选项只列前 60 个；选中的代码不在其中时 Dash 会把 value 清成 None。
    测试沙盒里的 universe 太小（全部都在前 60 个里），所以此前的路由测试没能发现。"""
    # 代码按字符串排序：用 688xxx（科创板）排在 150 个 600xxx 之后，确保落在第一页之外
    late = "688981.SH"
    rows = [(f"{600000 + i:06d}.SH", 10.0, 0.0) for i in range(150)] + [(late, 10.0, 0.0)]
    put_days(sandbox, {"2026-11-13": rows})
    assert late not in ml.search_symbols(None, market.OPTION_LIMIT)  # 前置条件：它确实在第一页之外
    opts = market.symbol_options(late)
    assert opts[0] == late and len(opts) == market.OPTION_LIMIT + 1
    page = appmod.route("/market", f"?symbol={late}")
    dd = [c for c in _walk(page) if getattr(c, "id", None) == "mk-symbol"][0]
    assert dd.value == late and late in dd.options
    # 搜索时：选中项保留，且不重复
    assert market.symbol_options(late, "6000").count(late) == 1
    assert market.symbol_options("600010.SH", "6000").count("600010.SH") == 1
    assert market.symbol_options(None, "6000")[0].startswith("6000")


def test_empty_selection_message_does_not_claim_the_cache_is_empty(sandbox):
    assert "请选择" in text_of(market.stock_view(None, [], "250"))
    assert "缓存为空" not in text_of(market.stock_view(None, [], "250"))


def test_one_corrupt_row_cannot_wreck_the_market_mean(sandbox):
    """真实数据：2022-02-25 的 920227.BJ（北交所换码日）pct_chg=200400%，把当天全市场平均拉到 +42.9%。
    均值截断后恢复正常；涨跌家数与中位数不受影响；其余合理的大涨（新股首日 +146%）被截断到 +100% 而非丢弃。"""
    rows = [(f"{600000 + i:06d}.SH", 10.0, 1.0) for i in range(100)]
    clean = ml.day_stats(day_frame("2022-02-25", rows))
    dirty = ml.day_stats(day_frame("2022-02-25", rows + [("920227.BJ", 20.05, 200400.0)]))
    assert clean["mean_ret"] == pytest.approx(1.0)
    assert dirty["mean_ret"] == pytest.approx(
        (100 * 1.0 + ml.MEAN_CLIP_PCT) / 101
    )  # 异常值按 +100% 计入
    assert dirty["mean_ret"] < 3.0  # 未截断时是 ~1982%
    assert dirty["median_ret"] == pytest.approx(1.0) and dirty["advancers"] == 101
    ipo = ml.day_stats(day_frame("2021-11-10", rows + [("301149.SZ", 20.0, 146.28)]))
    assert ipo["mean_ret"] == pytest.approx(
        (100 * 1.0 + 100.0) / 101
    )  # 合理的新股首日涨幅也只算 +100%
    neg = ml.day_stats(day_frame("2021-11-10", [("600000.SH", 10.0, -5.0)]))
    assert neg["mean_ret"] == pytest.approx(-5.0)  # 截断只影响超过 ±100% 的值
