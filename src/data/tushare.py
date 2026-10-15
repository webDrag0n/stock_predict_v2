"""Tushare data connector for Chinese A-share data.

Fetches index constituents and forward-adjusted (qfq) daily bars, merged with
`daily_basic` fields (turnover rate, volume ratio, pct change) that act as
A-share-specific factors. Each symbol's raw bars are cached to parquet so the
tushare API (which is rate-limited on the free tier) is hit at most once per
(symbol, date-range).

Requires a tushare pro token — set it in config_credentials.toml or the
TUSHARE_TOKEN env var. Register at https://tushare.pro.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C
from .base import DataConnector


def _announcement_dates(period: pd.Series, period_dt: pd.Series) -> pd.Series:
    """按报告期区分财报披露滞后（point-in-time 可用日）。

    A 股披露规则（证监会时限）：年报（period 尾号 1231）最迟 4 月 30 日
    （+120 天），其余定期报告（一季报/半年报/三季报）+62 天近似覆盖。
    此前统一 +62 天会让年报在 1~3 月就“可见”，造成前视泄露。
    """
    lag = np.where(period.astype(str).str.endswith("1231"), 120, 62)
    return period_dt + pd.to_timedelta(lag, unit="D")


class TushareConnector(DataConnector):
    name = "tushare"

    def __init__(self, token: str | None = None):
        token = (token or C.TUSHARE_TOKEN or "").strip()
        if not token:
            raise ValueError(
                "未配置 tushare token。请在 config_credentials.toml 的 [tushare] "
                "下填写 token，或设置环境变量 TUSHARE_TOKEN。注册地址：https://tushare.pro")
        import tushare as ts
        ts.set_token(token)
        self._pro = ts.pro_api()
        self._cache_dir = os.path.join(C.CACHE_DIR, "tushare")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._universe_cache: dict[str, list[str]] = {}

    # -- universe -----------------------------------------------------------
    def get_universe(self, index_code: str,
                     start_date: str | None = None,
                     end_date: str | None = None) -> list[str]:
        """Return the stock universe via whole-market quality screening.

        This token tier cannot access any index-constituent API
        (index_weight / index_member_all / hs_const all require 2000+ points).
        With UNIVERSE_SIZE=0 (default) the function returns the whole tradable
        market after quality filters. A positive UNIVERSE_SIZE is deprecated:
        small fixed pools (e.g. 300 large caps) destroy cross-sectional
        correlation and produce unstable OOS alpha.

        ST / *ST names and recently-listed stocks (< 1y before the anchor) are
        filtered out. ``index_code`` is accepted for API compatibility but the
        selection is purely quality-based.
        """
        anchor = (start_date or C.START_DATE or "").replace("-", "")
        size = getattr(C, "UNIVERSE_SIZE", 0)
        if size == 0:
            return self._screen_by_mktcap(anchor, top_n=0)
        return self._screen_by_mktcap(anchor, size)

    def _screen_by_mktcap(self, anchor_date: str, top_n: int) -> list[str]:
        """Top-N large-cap stocks using stock_basic heuristics.

        Primary method: full-market daily_basic snapshot → sort by total_mv.
        Fallback (rate-limited or insufficient points): sort by list_date
        ascending — longest-listed companies in the main board are typically
        large-cap. Either way, ST names and IPOs < 1y before anchor are excluded.
        When ``top_n`` is 0, return all filtered stocks (whole market).
        """
        cache_key = f"screen_{anchor_date or 'latest'}_{top_n}"
        if cache_key in self._universe_cache:
            return self._universe_cache[cache_key]
        cache = os.path.join(self._cache_dir, f"universe_{cache_key}.parquet")
        if os.path.exists(cache):
            # 旧缓存可能仍含 B 股（900xxx/200xxx），读入即过滤，无需删缓存重建
            members = [m for m in pd.read_parquet(cache)["ts_code"].tolist()
                       if not m.startswith(("900", "200"))]
            self._universe_cache[cache_key] = members
            return members

        cutoff = ""
        if anchor_date:
            cutoff = (pd.to_datetime(anchor_date, format="%Y%m%d")
                      - pd.Timedelta(days=365)).strftime("%Y%m%d")

        # Load all listed stocks from stock_basic (1 call, always works).
        sb = self._pro.stock_basic(
            exchange="", list_status="L",
            fields="ts_code,name,list_date,market")
        sb = sb[sb["market"] == "主板"]                          # main board only
        sb = sb[~sb["name"].str.contains("ST", na=False)]       # exclude ST
        # 剔除 B 股（沪市 900xxx / 深市 200xxx）：它们也挂"主板"标签，但交易币种
        # 不同，且 trading 的板幅前缀表会把 "9" 开头误判为北交所 30% 幅度。
        sb = sb[~sb["ts_code"].str.startswith(("900", "200"))]
        if cutoff:
            sb = sb[sb["list_date"] <= cutoff]                   # exclude recent IPOs

        # Try to get market cap from daily_basic for a cap-based sort.
        mktcap_ok = False
        probe = anchor_date or pd.Timestamp.today().strftime("%Y%m%d")
        for _ in range(10):
            try:
                df = self._pro.daily_basic(
                    trade_date=probe,
                    fields="ts_code,total_mv")
                if df is not None and not df.empty:
                    sb = sb.merge(df[["ts_code", "total_mv"]], on="ts_code", how="left")
                    sb = sb.sort_values("total_mv", ascending=False)
                    mktcap_ok = True
                    print(f"  市值排序 @ {probe}: 使用 daily_basic")
                    break
            except Exception:
                pass
            probe = (pd.to_datetime(probe, format="%Y%m%d")
                     + pd.Timedelta(days=1)).strftime("%Y%m%d")

        if not mktcap_ok:
            # Fallback: oldest-listed companies ≈ large cap (imperfect but no API needed)
            sb = sb.sort_values("list_date")
            print(f"  daily_basic 不可用，改用上市年龄排序（近似大盘股）")

        members = sb["ts_code"].tolist() if top_n == 0 else sb.head(top_n)["ts_code"].tolist()
        if not members:
            raise ValueError("市值筛选后股票池为空，请检查 token 与网络")
        label = "全部" if top_n == 0 else f"前 {len(members)}"
        print(f"  股票池: 主板 {'市值' if mktcap_ok else '上市年龄'} {label} 只"
              f"（剔除ST/次新）")
        pd.DataFrame({"ts_code": members}).to_parquet(cache, index=False)
        self._universe_cache[cache_key] = members
        return members

    # -- bars ---------------------------------------------------------------
    def _fetch_one(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        """Fetch one symbol's qfq daily bars from pro_bar (cached).

        We do NOT call daily_basic per-symbol — it is rate-limited to 1/hour on
        the free tier, making per-symbol fundamental fetches at market scale
        infeasible. pct_chg comes from pro_bar directly.
        """
        key = f"{ts_code}_{start}_{end}".replace("-", "")
        cache = os.path.join(self._cache_dir, f"bar_{key}.parquet")
        if os.path.exists(cache):
            return pd.read_parquet(cache)

        import tushare as ts
        s, e = start.replace("-", ""), end.replace("-", "")
        # adj=None avoids the adj_factor API (rate-limited to 1/min on free tier).
        # Large-cap A-shares rarely split, and all our features are ratio-based
        # (returns, RSI, MACD, BB) so they're scale-invariant over 5y windows.
        bars = ts.pro_bar(ts_code=ts_code, adj=None, start_date=s, end_date=e, freq="D")
        if bars is None or bars.empty:
            return pd.DataFrame()

        out = pd.DataFrame({
            "date":   pd.to_datetime(bars["trade_date"], format="%Y%m%d"),
            "symbol": ts_code,
            "open":   bars["open"],
            "high":   bars["high"],
            "low":    bars["low"],
            "close":  bars["close"],
            "volume": bars["vol"],
        })
        if "pct_chg" in bars.columns:
            out["pct_chg"] = bars["pct_chg"]
        out = out.sort_values("date").reset_index(drop=True)
        out.to_parquet(cache, index=False)
        return out

    def get_industry_map(self) -> dict[str, str]:
        """Return {ts_code: industry} for all listed stocks. Cached."""
        # 行业归属会随时间变化，缓存文件名带 YYYYMM 月份键，跨月自动重拉
        cache = os.path.join(self._cache_dir,
                             f"industry_map_{pd.Timestamp.now():%Y%m}.parquet")
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            return dict(zip(df["ts_code"], df["industry"]))
        sb = self._pro.stock_basic(exchange="", list_status="L",
                                   fields="ts_code,industry")
        sb = sb.dropna(subset=["industry"])
        sb.to_parquet(cache, index=False)
        return dict(zip(sb["ts_code"], sb["industry"]))

    def fetch_ohlcv(self, symbols: list[str], start_date: str,
                    end_date: str) -> pd.DataFrame:
        import time
        frames = []
        n = len(symbols)
        for i, code in enumerate(symbols):
            cached = os.path.exists(os.path.join(
                self._cache_dir,
                f"bar_{code}_{start_date}_{end_date}".replace("-", "") + ".parquet"))
            try:
                df = self._fetch_one(code, start_date, end_date)
                if df.empty:
                    print(f"  [{i+1}/{n}] {code}: 无数据")
                    continue
                frames.append(df)
                if (i + 1) % 10 == 0 or i + 1 == n:
                    print(f"  已获取 {i+1}/{n} 只股票")
                if not cached:
                    time.sleep(0.35)   # tushare free-tier: ~200 req/min
            except Exception as ex:
                print(f"  [{i+1}/{n}] {code}: 获取失败 {ex}")
        if not frames:
            raise ValueError("所有股票均未获取到数据，请检查 token 与网络")
        return pd.concat(frames, ignore_index=True)

    # -- whole-market by-date fetch -----------------------------------------
    def _find_cached_trade_dates(self, start: str, end: str) -> list[str] | None:
        """Return cached trading dates covering [start, end] if one exists."""
        import re
        start_i = int(start.replace("-", ""))
        end_i = int(end.replace("-", ""))
        best = None
        best_score = -1
        for f in os.listdir(self._cache_dir):
            m = re.match(r"tradecal_(\d{8})_(\d{8})\.parquet$", f)
            if not m:
                continue
            cs, ce = int(m.group(1)), int(m.group(2))
            if cs <= start_i and ce >= end_i:
                # Prefer the narrowest calendar that still covers the range.
                score = -(ce - cs)
                if score > best_score:
                    best_score = score
                    best = os.path.join(self._cache_dir, f)
        if best is None:
            return None
        df = pd.read_parquet(best)
        if "cal_date" not in df.columns:
            return None
        dates = sorted(df["cal_date"].astype(str).tolist())
        return [d for d in dates if start_i <= int(d) <= end_i]

    def _trade_dates_from_daily_cache(self, start: str, end: str) -> list[str] | None:
        """Derive trading dates from already cached daily_*.parquet files.

        This avoids the tushare trade_cal API entirely when we have already
        fetched whole-market daily bars. Returns dates in [start, end] (YYYYMMDD).
        """
        import re
        start_i = int(start.replace("-", ""))
        end_i = int(end.replace("-", ""))
        dates = []
        for f in os.listdir(self._cache_dir):
            m = re.match(r"daily_(\d{8})\.parquet$", f)
            if not m:
                continue
            d = int(m.group(1))
            if start_i <= d <= end_i:
                dates.append(str(d))
        return sorted(dates) if dates else None

    @staticmethod
    def _calendar_covers(dates: list[str], e: str) -> bool:
        """日历内容确实延伸到请求的 end 附近（容差 10 个自然日，覆盖春节等
        长假 + end 落在非交易日）。

        历史上派生路径把"只到最后一个 daily 缓存日"的日历按请求区间文件名
        存盘（文件名声称覆盖到 end，内容实际短 7 周），之后所有运行都命中
        这个被污染的 exact 缓存而静默用旧数据。因此每一级缓存都要校验内容
        覆盖度，不能只看文件名。
        """
        if not dates:
            return False
        return (pd.Timestamp(str(dates[-1]))
                >= pd.Timestamp(e) - pd.Timedelta(days=10))

    def _trade_dates(self, start: str, end: str) -> list[str]:
        """Trading-calendar dates in [start, end] (YYYYMMDD). Cached.

        Avoids calling the tushare trade_cal API wherever possible because the
        free tier is limited to 1 call/hour. We first look for an already cached
        calendar that covers the requested range, then derive from existing daily
        bar caches, and only fall back to the API as a last resort.
        每一级缓存都用 _calendar_covers 校验内容末端，防止 stale 缓存截断数据。
        """
        s, e = start.replace("-", ""), end.replace("-", "")
        cache = os.path.join(self._cache_dir, f"tradecal_{s}_{e}.parquet")
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            if "cal_date" in df.columns:
                dates = sorted(df["cal_date"].astype(str).tolist())
                if self._calendar_covers(dates, e):
                    return dates
                # 内容被截断的污染缓存：弃用并落到 API 重建
                print(f"  ⚠️ 日历缓存 {os.path.basename(cache)} 内容仅到 "
                      f"{dates[-1] if dates else '空'}，早于请求的 {e}，弃用重建")

        cached = self._find_cached_trade_dates(s, e)
        if cached is not None and self._calendar_covers(cached, e):
            pd.DataFrame({"cal_date": cached}).to_parquet(cache, index=False)
            return cached

        derived = self._trade_dates_from_daily_cache(s, e)
        # 防 stale：daily 缓存派生的日历只覆盖到最后一个缓存日，早于请求的
        # end 时必须落到 trade_cal API，否则静默截断数据。
        if derived is not None:
            if self._calendar_covers(derived, e):
                pd.DataFrame({"cal_date": derived}).to_parquet(cache, index=False)
                return derived
            print(f"  ⚠️ daily 缓存日历仅到 {derived[-1]}，早于请求的 {e}，"
                  "回退到 trade_cal API 拉取最新交易日历")

        try:
            cal = self._pro.trade_cal(exchange="SSE", start_date=s, end_date=e,
                                      is_open="1", fields="cal_date")
        except Exception as api_err:
            # tushare 免费档 trade_cal 限 1 次/小时；限流时用 akshare 的新浪
            # 财经交易日历兜底（免费、无限速），避免整个数据链路被一次
            # 频率限制卡死。
            print(f"  ⚠️ tushare trade_cal 不可用（{api_err}），"
                  "改用 akshare 交易日历兜底")
            import akshare as ak
            cal_ak = ak.tool_trade_date_hist_sina()
            col = "trade_date" if "trade_date" in cal_ak.columns else cal_ak.columns[0]
            days = pd.to_datetime(cal_ak[col]).dt.strftime("%Y%m%d")
            dates = sorted(d for d in days.tolist() if s <= d <= e)
            if not dates:
                raise RuntimeError(
                    f"akshare 交易日历也无 {s}~{e} 的数据") from api_err
            pd.DataFrame({"cal_date": dates}).to_parquet(cache, index=False)
            return dates
        if cal is None or cal.empty or "cal_date" not in cal.columns:
            raise RuntimeError(
                f"无法获取交易日历 ({s}~{e})。tushare trade_cal 接口受限或返回为空；"
                "请检查 token 权限，或手动提供交易日历缓存。")
        dates = sorted(cal["cal_date"].astype(str).tolist())
        pd.DataFrame({"cal_date": dates}).to_parquet(cache, index=False)
        return dates

    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame:
        """All stocks' daily bars for one trading date (1 API call, cached).

        Uses daily(trade_date=) — returns the whole market in a single call
        (~5500 rows / 0.3s), the only feasible way to fetch whole-market history
        without 5000+ per-symbol calls. adj=None (raw price), consistent with
        _fetch_one; ratio-based factors are scale-invariant.
        """
        cache = os.path.join(self._cache_dir, f"daily_{trade_date}.parquet")
        if os.path.exists(cache):
            return pd.read_parquet(cache)
        bars = self._pro.daily(trade_date=trade_date)
        if bars is None or bars.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "date":   pd.to_datetime(bars["trade_date"], format="%Y%m%d"),
            "symbol": bars["ts_code"],
            "open":   bars["open"],
            "high":   bars["high"],
            "low":    bars["low"],
            "close":  bars["close"],
            "volume": bars["vol"],
            "amount": bars["amount"],     # 成交额(千元) — for liquidity filter
            "pct_chg": bars["pct_chg"],
        })
        out.to_parquet(cache, index=False)
        return out

    def fetch_ohlcv_whole_market(self, start_date: str, end_date: str,
                                 symbols: list[str] | None = None) -> pd.DataFrame:
        """Whole-market daily bars via per-date fetch (1 call/trading-day).

        The efficient path for the entire A-share market: instead of N per-symbol
        calls, iterate the trading calendar and pull every stock for each date in
        one call. Per-date parquet cache → incremental daily updates cost 1 call.
        If `symbols` is given, the result is filtered to those codes (but the
        fetch still pulls the whole market per date, which is cached for reuse).
        """
        import time
        dates = self._trade_dates(start_date, end_date)
        n = len(dates)
        frames = []
        n_calls = 0
        for i, d in enumerate(dates):
            cached = os.path.exists(os.path.join(self._cache_dir, f"daily_{d}.parquet"))
            try:
                df = self._fetch_one_date(d)
                if not df.empty:
                    frames.append(df)
                if not cached:
                    n_calls += 1
                    time.sleep(0.32)        # ~190 req/min, under the 200/min cap
                if (i + 1) % 50 == 0 or i + 1 == n:
                    print(f"  全市场按日拉取 {i+1}/{n} 个交易日"
                          f"（新增API调用 {n_calls}）")
            except Exception as ex:
                print(f"  [{i+1}/{n}] {d}: 获取失败 {str(ex)[:80]}")
                time.sleep(1.0)
        if not frames:
            raise ValueError("全市场按日拉取无数据，请检查 token 与网络")
        out = pd.concat(frames, ignore_index=True)
        if symbols is not None:
            out = out[out["symbol"].isin(set(symbols))].reset_index(drop=True)
        return out

    # -- fundamentals (valuation factors) -----------------------------------
    def fetch_fundamentals(self, start_date: str, end_date: str,
                           freq_days: int = 21) -> pd.DataFrame:
        """逐股票估值因子（PE/PB/PS/市值），数据源 akshare（免费、无限速、日频）。

        tushare 免费 token 的 daily_basic 限速过严（密集调用触发小时级封锁），
        故估值因子改用 akshare 的 stock_value_em（东财个股每日估值），逐股票拉取
        并缓存到 data_cache/akshare_value/。日频原生数据，无需前向填充，也不存在
        快照采样的近似。`freq_days` 参数保留仅为接口兼容，此实现忽略。

        返回长表：date, symbol, pe_ttm, pb, ps_ttm, total_mv, circ_mv
        （symbol 用 tushare ts_code 格式，便于与价格面板对齐）
        """
        import time
        import akshare as ak

        cache_dir = os.path.join(C.CACHE_DIR, "akshare_value")
        os.makedirs(cache_dir, exist_ok=True)
        symbols = C.DEFAULT_SYMBOLS or list(self._universe_cache.get(
            f"screen_{start_date.replace('-', '')}_{C.UNIVERSE_SIZE}", []))
        if not symbols:
            print("  ⚠️ 无股票列表用于基本面拉取，跳过")
            return pd.DataFrame()

        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)
        frames = []
        n = len(symbols)
        for i, ts_code in enumerate(symbols):
            code = ts_code.split(".")[0]   # 600519.SH -> 600519
            # 缓存带 YYYYMM 月份键：估值快照持续更新，跨月自动重拉，
            # 避免旧缓存永不刷新
            cache = os.path.join(
                cache_dir, f"val_{code}_{pd.Timestamp.now():%Y%m}.parquet")
            if os.path.exists(cache):
                df = pd.read_parquet(cache)
            else:
                try:
                    raw = ak.stock_value_em(symbol=code)
                    if raw is None or raw.empty:
                        continue
                    df = pd.DataFrame({
                        "date": pd.to_datetime(raw["数据日期"]),
                        "symbol": ts_code,
                        "pe_ttm": pd.to_numeric(raw["PE(TTM)"], errors="coerce"),
                        "pb": pd.to_numeric(raw["市净率"], errors="coerce"),
                        "ps_ttm": pd.to_numeric(raw["市销率"], errors="coerce"),
                        "total_mv": pd.to_numeric(raw["总市值"], errors="coerce"),
                        "circ_mv": pd.to_numeric(raw["流通市值"], errors="coerce"),
                    })
                    df.to_parquet(cache, index=False)
                    time.sleep(0.3)
                except Exception as ex:
                    print(f"  [{i+1}/{n}] {ts_code} 估值获取失败: {str(ex)[:60]}")
                    continue
            frames.append(df)
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"  基本面(估值): 已获取 {i+1}/{n} 只股票")

        if not frames:
            print("  ⚠️ 未获取到任何估值数据，跳过基本面因子")
            return pd.DataFrame()
        allv = pd.concat(frames, ignore_index=True)
        allv = allv[(allv["date"] >= s) & (allv["date"] <= e)]
        return allv.reset_index(drop=True)


    def fetch_financial_factors(self, symbols: list[str]) -> pd.DataFrame:
        """Quarterly fundamental factors via akshare stock_financial_abstract (free).

        Fetches ROE, revenue, net profit, gross margin per stock, applies a
        point-in-time delay (announcement lag), derives YoY growth rates, and
        forward-fills to daily frequency so it can be merged with the price panel.

        Announcement delay (Chinese A-share rules):
          Q1/Q3 reports + semi-annual: +62 days
          Annual (period ends 1231):   +120 days (released by April 30)
        年报用 +120 天，其余 +62 天（见模块级 _announcement_dates）。

        Returns long DataFrame: date, symbol, roe, gross_margin,
                                 revenue_yoy, profit_yoy
        """
        import time
        import akshare as ak

        cache_dir = os.path.join(C.CACHE_DIR, "akshare_financial")
        os.makedirs(cache_dir, exist_ok=True)

        METRICS = {
            "净资产收益率(ROE)":  "roe",
            "毛利率":            "gross_margin",
            "归母净利润":        "net_profit",
            "营业总收入":        "revenue",
        }

        frames = []
        n = len(symbols)
        for i, ts_code in enumerate(symbols):
            code = ts_code.split(".")[0]
            # 缓存带 YYYYMM 月份键：财报随披露季更新，跨月自动重拉
            cache = os.path.join(
                cache_dir, f"fin_{code}_{pd.Timestamp.now():%Y%m}.parquet")
            if os.path.exists(cache):
                df = pd.read_parquet(cache)
            else:
                try:
                    raw = ak.stock_financial_abstract(symbol=code)
                    if raw is None or raw.empty:
                        continue
                    # keep only needed rows
                    raw = raw[raw["指标"].isin(METRICS)].copy()
                    # melt: (metric × period) → long
                    period_cols = [c for c in raw.columns if c.isdigit() and len(c) == 8]
                    melted = raw.melt(id_vars=["指标"], value_vars=period_cols,
                                      var_name="period", value_name="value")
                    melted["metric"] = melted["指标"].map(METRICS)
                    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
                    melted = melted.dropna(subset=["value"])
                    # pivot: one column per metric
                    piv = (melted.pivot_table(index="period", columns="metric",
                                              values="value", aggfunc="first")
                                 .reset_index())
                    piv["period_dt"] = pd.to_datetime(piv["period"], format="%Y%m%d")
                    # announcement delay: 年报 +120 天，其余 +62 天（防泄露）
                    piv["date"] = _announcement_dates(piv["period"], piv["period_dt"])
                    piv = piv.sort_values("date")
                    # YoY growth: 按 period 年份对齐去年同期（period - 10000），
                    # 而不是 shift(4)——财报缺季时 shift(4) 会错位到错误季度，
                    # 按期间键匹配则缺季得 NaN（诚实缺失而非错值）。
                    piv["period_int"] = piv["period"].astype(int)
                    for col in ("net_profit", "revenue"):
                        if col in piv.columns:
                            lookup = piv.set_index("period_int")[col]
                            prior = piv["period_int"].map(
                                lambda p, lk=lookup: lk.get(p - 10000, np.nan))
                            piv[f"{col}_yoy"] = (piv[col] - prior) / prior.abs().clip(lower=1e-9)
                    piv["symbol"] = ts_code
                    keep = ["date", "symbol", "roe", "gross_margin",
                            "revenue_yoy", "profit_yoy"]
                    if "net_profit_yoy" in piv.columns:
                        piv = piv.rename(columns={"net_profit_yoy": "profit_yoy",
                                                  "revenue_yoy": "revenue_yoy"})
                    df = piv[[c for c in keep if c in piv.columns]].copy()
                    df.to_parquet(cache, index=False)
                    time.sleep(0.5)
                except Exception as ex:
                    print(f"  [{i+1}/{n}] {ts_code} 财务数据失败: {str(ex)[:60]}")
                    continue
            frames.append(df)
            if (i + 1) % 20 == 0 or i + 1 == n:
                print(f"  财务因子: {i+1}/{n}")

        if not frames:
            print("  ⚠️ 未获取到任何财务数据")
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
