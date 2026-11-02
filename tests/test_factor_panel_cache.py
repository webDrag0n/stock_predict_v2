"""因子面板磁盘缓存测试（load_factor_panel 的签名命中 / 尾部增量重算）。

合成全市场数据源（fake connector）覆盖四条路径：
  (a) 无缓存 → 全量构建并写缓存（parquet + 签名 json）
  (b) 数据/配置不变 → 命中缓存，结果与首次构建逐位相等
  (c) 尾部新增 2 个交易日 → 增量重算尾部窗口并拼接：拼接点之前的旧部分与首
      次缓存逐位相等；拼接段与全量重算对比——非 ewm 列逐位相等，ewm 近似列
      （MACD 族，无限记忆截断）max abs diff < 1e-6（远严于任务容差 1e-4）
  (d) 因子代码版本戳变化 → 签名失效，回退全量重算
另覆盖：staleness 超包络回退全量、缓存损坏回退全量、单一超集缓存服务不同
keep_* 视图。

因子函数用真实的 alpha158 / gtja191（子集）/ technical / orthogonal 小规模调用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
import src.data as D

SYMS = [f"{i:06d}.SZ" for i in range(1, 61)]
DATES = pd.bdate_range("2023-01-02", periods=800)


def _make_bars() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for i, s in enumerate(SYMS):
        n = len(DATES)
        rets = rng.normal(0.0003, 0.02, n)
        close = 10.0 * np.exp(np.cumsum(rets)) * (1 + i % 5)
        open_ = close * (1 + rng.normal(0, 0.005, n))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n)))
        vol = rng.uniform(1e5, 5e5, n)
        prev_close = np.r_[close[0], close[:-1]]
        pct_chg = (close / prev_close - 1.0) * 100.0
        pct_chg[0] = 0.0
        frames.append(
            pd.DataFrame(
                {
                    "date": DATES,
                    "symbol": s,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                    "amount": vol * close / 10.0,
                    "pct_chg": pct_chg,
                    "turnover_rate": rng.uniform(0.5, 5.0, n),
                    "volume_ratio": rng.uniform(0.5, 2.0, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


BARS = _make_bars()
START = str(DATES[0].date())
# 全市场质量过滤（rolling(20, min_periods=10) 成交额中位数）会裁掉头部 9 个
# 暖机交易日，全量构建的因子输入从第 10 个交易日开始。
FIRST_FACTOR_DATE = DATES[9]


class _FakeConnector(D.DataConnector):
    name = "fake"

    def get_universe(self, index_code, start_date=None, end_date=None):
        return SYMS

    def fetch_ohlcv(self, symbols, start_date, end_date):
        return self.fetch_ohlcv_whole_market(start_date, end_date)

    def fetch_ohlcv_whole_market(self, start_date, end_date, symbols=None):
        s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
        return BARS[(BARS["date"] >= s) & (BARS["date"] <= e)].copy().reset_index(drop=True)

    def trade_dates(self, start_date, end_date):
        s, e = pd.Timestamp(start_date), pd.Timestamp(end_date)
        return [d.strftime("%Y%m%d") for d in DATES if s <= d <= e]


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CACHE_DIR", str(tmp_path))
    for k in (
        "USE_FUNDAMENTALS",
        "USE_FINANCIAL_FACTORS",
        "USE_NORTHBOUND",
        "USE_OPTION_FEATURES",
        "USE_GLOBAL_MACRO",
        "REGIME_IC_FEATURES",
        "INDUSTRY_NEUTRAL",
    ):
        monkeypatch.setattr(C, k, False)
    monkeypatch.setattr(C, "GTJA191_ALPHA_SUBSET", "1,2,3,4,5,6,7,8")
    monkeypatch.setitem(D._CONNECTORS, "fake", _FakeConnector)
    return tmp_path


def _spy_compute(monkeypatch):
    """Wrap src.data.compute_all_factors, returning the list of calls."""
    calls = []
    orig = D.compute_all_factors

    def spy(df):
        calls.append(df)
        return orig(df)

    monkeypatch.setattr(D, "compute_all_factors", spy)
    return calls


def _load(end_idx: int, **kw):
    kw.setdefault("verbose", False)
    return D.load_factor_panel(
        source="fake", start_date=START, end_date=str(DATES[end_idx].date()), **kw
    )


def test_cold_build_writes_cache(fake_env):
    parquet_path, meta_path = D._panel_cache_paths()
    panel = _load(-3)
    assert not panel.empty and C.TARGET in panel.columns
    # 视图层丢弃隐藏/中间列
    assert "adj_close" not in panel.columns and "_adj_cp" not in panel.columns
    meta = D._read_panel_cache_meta(meta_path)
    assert meta is not None and meta["sig"] and meta["req_start"] == START
    assert meta["data_end"] == str(DATES[-3].date())
    cached = pd.read_parquet(parquet_path)
    # 缓存是超集面板：隐藏锚点列在缓存里、交易列在缓存里
    assert {"adj_close", "_adj_cp", "open", "bb_upper_20"} <= set(cached.columns)


def test_cache_hit_bit_identical(fake_env, monkeypatch):
    p1 = _load(-3)
    # 命中路径不得触发任何重算
    monkeypatch.setattr(
        D,
        "compute_all_factors",
        lambda df: (_ for _ in ()).throw(RuntimeError("unexpected recompute")),
    )
    p2 = _load(-3)
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)


def test_cache_serves_all_keep_views(fake_env, monkeypatch):
    _load(-3)
    monkeypatch.setattr(
        D,
        "compute_all_factors",
        lambda df: (_ for _ in ()).throw(RuntimeError("unexpected recompute")),
    )
    trade = _load(-3, keep_trade_cols=True)
    assert {"open", "high", "low", "close", "bb_upper_20", "bb_lower_20"} <= set(trade.columns)
    plain = _load(-3)
    assert "open" not in plain.columns and "pct_chg" not in plain.columns
    live = _load(-3, keep_unlabeled=True)
    # keep_unlabeled: 最新 FORWARD_WINDOW 天的无标签行保留；默认视图则丢弃
    assert len(live) > len(plain)
    assert plain[C.TARGET].notna().all()
    last = live.groupby("symbol")["date"].max()
    assert (last == DATES[-3]).all()


def test_tail_incremental_matches_full(fake_env, monkeypatch):
    p1 = _load(-3)  # 缓存写到 DATES[-3]
    calls = _spy_compute(monkeypatch)
    p_inc = _load(-1)  # 尾部新增 2 个交易日
    assert len(calls) == 1
    tail_input = calls[0]
    n_win, keep_span = D._panel_cache_windows()
    # 增量路径只重算尾部窗口（n_win 回看 + 20 天过滤暖机 + 日历换算余量），
    # 而非全部 800 天历史
    assert tail_input["date"].nunique() <= int((n_win + 20) * 1.5)
    assert tail_input["date"].min() > FIRST_FACTOR_DATE
    assert tail_input["date"].max() == DATES[-1]

    # 参考：同区间无缓存全量重算
    monkeypatch.setattr(C, "FACTOR_PANEL_CACHE", False)
    p_ref = _load(-1)
    monkeypatch.setattr(C, "FACTOR_PANEL_CACHE", True)

    assert p_inc.shape == p_ref.shape
    assert list(p_inc.columns) == list(p_ref.columns)
    m = p_inc.merge(p_ref, on=["date", "symbol"], suffixes=("_a", "_b"))
    assert len(m) == len(p_inc)
    ewm_cols = {"macd", "macd_signal", "macd_diff"}
    for c in p_inc.columns:
        if c in ("date", "symbol"):
            continue
        a = pd.to_numeric(m[f"{c}_a"], errors="coerce").to_numpy(np.float64)
        b = pd.to_numeric(m[f"{c}_b"], errors="coerce").to_numpy(np.float64)
        diff = np.abs(a - b)
        diff[np.isnan(a) & np.isnan(b)] = 0.0
        mx = float(np.nanmax(diff))
        if c in ewm_cols:
            assert mx < 1e-6, f"ewm 列 {c} 截断误差 {mx:.2e} 超界"
        else:
            # 非 ewm 列理论逐位相等；alpha158 闭式解基于前缀和（cumsum），浮点
            # 累加与截断位置相关，残留 ~1ulp（1e-9 量级）差异——阈值 1e-7 仍远
            # 严于任务容差 1e-4。
            assert mx < 1e-7, f"非 ewm 列 {c} 差异 {mx:.2e} 超界"

    # 拼接点之前的旧部分与首次缓存逐位相等
    splice = DATES[-keep_span]
    old_inc = p_inc[p_inc["date"] < splice].reset_index(drop=True)
    old_first = p1[p1["date"] < splice].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_inc, old_first, check_exact=True)

    # 缓存已推进到新尾日，再次请求直接命中
    meta = D._read_panel_cache_meta(D._panel_cache_paths()[1])
    assert meta["data_end"] == str(DATES[-1].date())


def test_no_new_trading_day_serves_cache(fake_env, monkeypatch):
    p1 = _load(-1)  # 缓存写到 DATES[-1]
    calls = _spy_compute(monkeypatch)
    # 请求尾部超出缓存 3 个自然日，但无新交易日（当日数据未入库/周末重跑）：
    # 日历预判直接命中，不得触发任何重算
    end = str((DATES[-1] + pd.Timedelta(days=3)).date())
    p2 = D.load_factor_panel(source="fake", start_date=START, end_date=end, verbose=False)
    assert not calls
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)


def test_trading_day_without_bars_serves_cache(fake_env, monkeypatch):
    """日历有新交易日但行情尚未入库（盘中重跑）→ 命中缓存，不构建尾部窗口。"""
    p1 = _load(-1)  # 缓存写到 DATES[-1]
    calls = _spy_compute(monkeypatch)
    extra = DATES[-1] + pd.offsets.BDay(1)
    monkeypatch.setattr(
        _FakeConnector, "trade_dates", lambda self, s, e: [extra.strftime("%Y%m%d")]
    )
    monkeypatch.setattr(_FakeConnector, "bar_available", lambda self, d: False, raising=False)
    p2 = D.load_factor_panel(
        source="fake", start_date=START, end_date=str(extra.date()), verbose=False
    )
    assert not calls
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)


def test_stale_beyond_envelope_falls_back_to_full(fake_env, monkeypatch):
    _load(-50)  # 缓存写到 DATES[-50]
    calls = _spy_compute(monkeypatch)
    p_stale = _load(-1)  # staleness = 49 > _PANEL_CACHE_STALE_SLACK(40) → 全量
    assert len(calls) == 1  # 日历日预判直接回退，未浪费尾部窗口构建
    assert calls[0]["date"].min() == FIRST_FACTOR_DATE
    monkeypatch.setattr(C, "FACTOR_PANEL_CACHE", False)
    p_ref = _load(-1)
    pd.testing.assert_frame_equal(p_stale, p_ref, check_exact=True)


def test_corrupt_cache_falls_back_to_full(fake_env, monkeypatch):
    p1 = _load(-3)
    parquet_path, _ = D._panel_cache_paths()
    with open(parquet_path, "wb") as f:
        f.write(b"garbage")
    calls = _spy_compute(monkeypatch)
    p2 = _load(-3)
    assert calls and calls[0]["date"].min() == FIRST_FACTOR_DATE
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)


def test_factor_code_change_forces_full_rebuild(fake_env, monkeypatch):
    _load(-3)
    monkeypatch.setattr(D, "_factor_source_hash", lambda: "deadbeefdead")
    calls = _spy_compute(monkeypatch)
    _load(-3)
    assert calls and calls[0]["date"].min() == FIRST_FACTOR_DATE
