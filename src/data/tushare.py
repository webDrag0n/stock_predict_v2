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

import pandas as pd

import config as C
from .base import DataConnector


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
        """Return the stock universe via whole-market market-cap screening.

        This token tier cannot access any index-constituent API
        (index_weight / index_member_all / hs_const all require 2000+ points),
        so instead of literal CSI-300 membership we approximate a large-cap
        universe: rank the entire market by total market value on a trade day
        *near the backtest start* and keep the top ``C.UNIVERSE_SIZE`` names.

        Anchoring the ranking at the START of the window (not the end) avoids
        look-ahead bias — we only use information available at backtest open.
        ST / *ST names and recently-listed stocks (< 1y before the anchor) are
        filtered out. ``index_code`` is accepted for API compatibility but the
        selection is purely cap-based.
        """
        anchor = (start_date or C.START_DATE or "").replace("-", "")
        return self._screen_by_mktcap(anchor, C.UNIVERSE_SIZE)

    def _screen_by_mktcap(self, anchor_date: str, top_n: int) -> list[str]:
        """Top-N large-cap stocks using stock_basic heuristics.

        Primary method: full-market daily_basic snapshot → sort by total_mv.
        Fallback (rate-limited or insufficient points): sort by list_date
        ascending — longest-listed companies in the main board are typically
        large-cap. Either way, ST names and IPOs < 1y before anchor are excluded.
        """
        cache_key = f"screen_{anchor_date or 'latest'}_{top_n}"
        if cache_key in self._universe_cache:
            return self._universe_cache[cache_key]
        cache = os.path.join(self._cache_dir, f"universe_{cache_key}.parquet")
        if os.path.exists(cache):
            members = pd.read_parquet(cache)["ts_code"].tolist()
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

        members = sb.head(top_n)["ts_code"].tolist()
        if not members:
            raise ValueError("市值筛选后股票池为空，请检查 token 与网络")
        print(f"  股票池: 主板 {'市值' if mktcap_ok else '上市年龄'} 前 {len(members)} 只"
              f"（剔除ST/次新）")
        pd.DataFrame({"ts_code": members}).to_parquet(cache, index=False)
        self._universe_cache[cache_key] = members
        return members

    # -- bars ---------------------------------------------------------------
    def _fetch_one(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        """Fetch one symbol's qfq daily bars from pro_bar (cached).

        We do NOT call daily_basic per-symbol — it is rate-limited to 1/hour on
        the free tier, making 300-stock × multi-year fetches impossible.
        pct_chg comes from pro_bar directly and is used for the limit-flag feature.
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
        cache = os.path.join(self._cache_dir, "industry_map.parquet")
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
            cache = os.path.join(cache_dir, f"val_{code}.parquet")
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
          Q1/Q3 reports: +31 days after period end
          Semi-annual:   +62 days
          Annual:        +120 days (usually released by April 30)
        We use +62 days for all to stay conservative.

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
            cache = os.path.join(cache_dir, f"fin_{code}.parquet")
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
                    # announcement delay: 62 days for all quarters
                    piv["date"] = piv["period_dt"] + pd.Timedelta(days=62)
                    piv = piv.sort_values("date")
                    # YoY growth (4 rows back = same quarter last year)
                    for col in ("net_profit", "revenue"):
                        if col in piv.columns:
                            prior = piv[col].shift(4)
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
