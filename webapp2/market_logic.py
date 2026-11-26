"""行情页的纯逻辑：全市场宽度、涨跌停统计（按板块幅度）、单股行情与指标。不依赖 dash。

涨跌停口径：与交易引擎同一套板块幅度（src/trading._board_limit_pct：主板 10%、创业板/科创板 20%、
北交所 30%）加 0.5 个百分点容差。旧行情浏览器对全市场统一用 ±9.9%：创业板/科创板日常涨跌超过 9.9%
不是涨跌停，会被多算（实测 2026-11-13 旧口径数出 81 只跌停，按板块只有 49 只）。
"""

from __future__ import annotations

import functools
import glob
import os
import threading

import numpy as np
import pandas as pd

from . import data

LIMIT_TOL = 0.5  # 百分点，与 trading.detect_board_limit 一致
MEAN_CLIP_PCT = 100.0  # 计算全市场平均涨跌幅时，单只股票单日涨跌幅的截断幅度（%）
_DAILY_RE_GLOB = "daily_????????.parquet"


@functools.lru_cache(maxsize=1 << 15)
def _limit_of(symbol: str) -> float:
    code = symbol.split(".")[0]
    if code.startswith(("8", "4", "9")):
        return 30.0
    if code.startswith(("30", "68")):
        return 20.0
    return 10.0


def limit_pcts(symbols: pd.Series) -> pd.Series:
    """symbol → 涨跌停幅度（%）：规则同 trading._board_limit_pct（行情页没有 ST 信息，按非 ST）。

    板块只取决于代码前缀，所以每个代码只算一次（lru_cache），再按值映射。逐日逐只调用引擎函数
    或用 .str 访问器都会逐元素走 Python：1131 个交易日 × 5500 只，首次打开页面要 10~18 秒。
    与引擎的一致性由 tests/test_webapp2_market.py 对引擎函数逐前缀比对来保证。
    """
    return symbols.astype(str).map(_limit_of).astype(float)


def day_stats(df: pd.DataFrame) -> dict:
    """一个交易日的横截面统计：涨跌家数、均值/中位数、按板块幅度的涨跌停数。

    df 需含 symbol/pct_chg（百分比）。NaN 的 pct_chg 不计入任何类别。
    """
    pc = pd.to_numeric(df["pct_chg"], errors="coerce")
    valid = pc.notna()
    lim = limit_pcts(df["symbol"])
    up = (pc >= lim - LIMIT_TOL) & valid
    dn = (pc <= -(lim - LIMIT_TOL)) & valid
    arr = pc[valid].to_numpy(dtype=float)
    # 均值只用于「全市场平均涨跌幅」曲线：单只股票的异常值（行情缓存里 2022-02-25 的 920227.BJ
    # 北交所换码日 pct_chg=200400%）会把当天均值拉到 +42.9%，曲线失去意义。把每只的单日涨跌幅截断在
    # ±MEAN_CLIP_PCT 再取均值；涨跌家数、涨跌停、中位数不受影响（它们本来就对极端值不敏感）。
    clipped = np.clip(arr, -MEAN_CLIP_PCT, MEAN_CLIP_PCT)
    return {
        "advancers": int((arr > 0).sum()),
        "decliners": int((arr < 0).sum()),
        "unchanged": int((arr == 0).sum()),
        "mean_ret": float(np.mean(clipped)) if len(arr) else float("nan"),
        "median_ret": float(np.median(arr)) if len(arr) else float("nan"),
        "limit_up": int(up.sum()),
        "limit_down": int(dn.sum()),
        "n": int(valid.sum()),
    }


def _daily_files() -> list[str]:
    return sorted(glob.glob(os.path.join(data.CACHE, "tushare", _DAILY_RE_GLOB)))


def _compute_breadth(files: tuple[tuple[str, float | None], ...]) -> pd.DataFrame:
    recs = []
    for path, _ in files:
        try:
            df = pd.read_parquet(path, columns=["date", "symbol", "pct_chg"])
        except Exception:
            continue
        if df.empty:
            continue
        recs.append({"date": pd.Timestamp(df["date"].iloc[0]), **day_stats(df)})
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True) if recs else pd.DataFrame()


_breadth_cache: dict = {}  # files 键（含每个文件的 mtime）→ 结果表；只保留最新一个键
_breadth_jobs: dict = {}
_breadth_lock = threading.Lock()


def _fill_breadth(files) -> pd.DataFrame:
    try:
        out = _compute_breadth(files)
    except Exception:
        # 失败也要结束任务并落一个空表：否则任务永远「计算中」，页面一直转圈
        out = pd.DataFrame()
    with _breadth_lock:
        _breadth_cache.clear()  # 旧键对应旧的文件集合，留着只会占内存
        _breadth_cache[files] = out
        _breadth_jobs.pop(files, None)
    return out


def breadth(block: bool = False) -> pd.DataFrame | None:
    """逐日全市场宽度。算完前返回 None（页面显示「计算中」），算完后返回表。

    读 1131 个逐日文件首次要 6~7 秒，不能卡住页面加载：首次调用在后台线程里算，之后命中缓存。
    键里含每个日线文件的 mtime，新增交易日或补写文件会触发重算。block=True 同步等待（测试/脚本用）。
    """
    files = tuple((f, data.mtime(f)) for f in _daily_files())
    if not files:
        return pd.DataFrame()
    with _breadth_lock:
        if files in _breadth_cache:
            return _breadth_cache[files]
        if not block:
            if files not in _breadth_jobs:
                job = threading.Thread(target=_fill_breadth, args=(files,), daemon=True)
                _breadth_jobs[files] = job
                job.start()
            return None
    return _fill_breadth(files)


def reset_caches() -> None:
    """测试用：清空宽度缓存与单股缓存。"""
    with _breadth_lock:
        _breadth_cache.clear()
        _breadth_jobs.clear()
    _bars.cache_clear()
    _tail_rows.cache_clear()
    _universe.cache_clear()


def latest_day() -> str | None:
    fs = _daily_files()
    return os.path.basename(fs[-1])[6:14] if fs else None


@functools.lru_cache(maxsize=2)
def _universe(path: str, _mt: float | None) -> tuple[str, ...]:
    try:
        return tuple(
            sorted(pd.read_parquet(path, columns=["symbol"])["symbol"].astype(str).unique())
        )
    except Exception:
        return ()


def universe() -> tuple[str, ...]:
    """最新交易日出现过的全部代码（已退市/停牌的不在其中）。"""
    fs = _daily_files()
    return _universe(fs[-1], data.mtime(fs[-1])) if fs else ()


def search_symbols(query: str | None, limit: int = 30) -> list[str]:
    """按代码子串匹配，前缀匹配优先；空查询返回前 limit 个。"""
    syms = universe()
    q = (query or "").strip().upper()
    if not q:
        return list(syms[:limit])
    pref = [s for s in syms if s.startswith(q)]
    rest = [s for s in syms if q in s and not s.startswith(q)]
    return (pref + rest)[:limit]


@functools.lru_cache(maxsize=64)
def _bars(path: str, _mt: float | None, symbol: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    try:
        tb = pq.read_table(path, filters=[("symbol", "==", symbol)])
    except Exception:
        return pd.DataFrame()
    return tb.to_pandas().sort_values("date").reset_index(drop=True)


@functools.lru_cache(maxsize=256)
def _tail_rows(files: tuple[tuple[str, float | None], ...], symbol: str) -> pd.DataFrame:
    rows = []
    for path, _ in files:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        sub = df[df["symbol"].astype(str) == symbol]
        if not sub.empty:
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def symbol_bars(symbol: str, window: int = 250) -> pd.DataFrame:
    """单股 OHLCV 历史。window<=0 取全部。

    历史来自合并文件 wholemkt_bars_long.parquet（只读该股的行，不加载 600 万行）；它不是每天
    更新的（实测止于 11-06，而逐日缓存已到 11-13），所以合并文件之后的交易日从逐日文件补齐，
    否则图表会悄悄少一周。合并文件缺失时，全部从逐日文件读取。
    """
    symbol = str(symbol)
    path = os.path.join(data.CACHE, "wholemkt_bars_long.parquet")
    mt = data.mtime(path)
    base = _bars(path, mt, symbol) if mt is not None else pd.DataFrame()
    last = pd.Timestamp(base["date"].iloc[-1]) if not base.empty else None
    newer = [
        (f, data.mtime(f))
        for f in _daily_files()
        if last is None or pd.Timestamp(os.path.basename(f)[6:14]) > last
    ]
    if newer:
        tail = _tail_rows(tuple(newer), symbol)
        if not tail.empty:
            cols = (
                [c for c in base.columns if c in tail.columns]
                if not base.empty
                else list(tail.columns)
            )
            base = pd.concat(
                [base[cols] if not base.empty else base, tail[cols]], ignore_index=True
            )
            base = (
                base.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
            )
    return base.tail(window) if window and window > 0 else base


def bars_last_date(symbol: str) -> pd.Timestamp | None:
    b = symbol_bars(symbol, 1)
    return pd.Timestamp(b["date"].iloc[-1]) if not b.empty else None


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    dif = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    dea = dif.ewm(span=9).mean()
    return dif, dea, dif - dea


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return ma, ma + k * sd, ma - k * sd


def stale_note(bars_last, market_last: str | None) -> str | None:
    """个股行情落后于全市场最新日 ⇒ 说明（停牌，或 bars 缓存没更新）。"""
    if bars_last is None or not market_last:
        return None
    gap = (pd.Timestamp(market_last) - pd.Timestamp(bars_last)).days
    return (
        None
        if gap <= 0
        else f"该股行情止于 {pd.Timestamp(bars_last).date()}，比全市场最新日 {market_last} 早 {gap} 天"
    )


def signature() -> list:
    return [data.mtime(f) for f in _daily_files()[-3:]] + [len(_daily_files())]
