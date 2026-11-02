"""Cached data-access layer for the web app.

Every loader is memoized so callbacks are fast and the heavy parquet files are
read at most once. Paths resolve against the project root (parent of webapp/).
Nothing here trains or fetches from the network — it only reads on-disk caches
produced by the CLI pipeline.
"""

from __future__ import annotations

import os
import glob
import functools
import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache")
TUSHARE_CACHE = os.path.join(CACHE, "tushare")
OUTPUTS = os.path.join(ROOT, "outputs")
TRACKING = os.path.join(OUTPUTS, "tracking")
MODELS = os.path.join(ROOT, "models")
FACTOR_LIBRARY = os.path.join(ROOT, "factor_library")

SEED_OOS = {42: "wholemkt_oos.parquet", 123: "seed_oos_123.parquet", 7: "seed_oos_7.parquet"}


def _p(*parts) -> str:
    return os.path.join(*parts)


def _mtime(path: str) -> float | None:
    """文件 mtime；不存在返回 None（作为缓存 key 的一部分，None 即空态）。"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


@functools.lru_cache(maxsize=1)
def available_seeds() -> tuple[int, ...]:
    """Seeds whose OOS prediction cache exists on disk."""
    out = []
    for sd, fn in SEED_OOS.items():
        if os.path.exists(_p(CACHE, fn)):
            out.append(sd)
    return tuple(sorted(out))


@functools.lru_cache(maxsize=8)
def seed_oos(seed: int) -> pd.DataFrame:
    """One seed's continuous OOS predictions: date, symbol, prediction, responder_6, return_1d."""
    fn = SEED_OOS.get(seed)
    if fn is None or not os.path.exists(_p(CACHE, fn)):
        return pd.DataFrame()
    return pd.read_parquet(_p(CACHE, fn))


@functools.lru_cache(maxsize=1)
def amount_map() -> pd.DataFrame:
    """(date, symbol) -> amount (千元) from the whole-market panel, for the impact cost."""
    path = _p(CACHE, "wholemkt_panel.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "symbol", "amount"])
    return pd.read_parquet(path, columns=["date", "symbol", "amount"])


@functools.lru_cache(maxsize=1)
def oos_with_amount_keys() -> tuple:
    """Cheap merge helper: returns nothing heavy, just confirms amount availability."""
    return ("amount" in amount_map().columns,)


@functools.lru_cache(maxsize=1)
def daily_files() -> tuple[str, ...]:
    """Sorted list of whole-market per-date parquet files (date-keyed cache)."""
    return tuple(sorted(glob.glob(_p(TUSHARE_CACHE, "daily_*.parquet"))))


@functools.lru_cache(maxsize=1)
def trade_dates() -> tuple[str, ...]:
    """YYYYMMDD strings parsed from the daily cache filenames."""
    return tuple(
        os.path.basename(f).replace("daily_", "").replace(".parquet", "") for f in daily_files()
    )


@functools.lru_cache(maxsize=1)
def universe() -> pd.DataFrame:
    """All symbols seen in the most recent daily file (code list for the explorer)."""
    files = daily_files()
    if not files:
        return pd.DataFrame(columns=["symbol"])
    last = pd.read_parquet(files[-1], columns=["symbol"])
    return pd.DataFrame({"symbol": sorted(last["symbol"].unique())})


@functools.lru_cache(maxsize=256)
def symbol_bars(symbol: str) -> pd.DataFrame:
    """Full OHLCV history for one symbol. Fast path: a consolidated long-format
    parquet (all daily caches concatenated, sorted by symbol) built once; falls
    back to scanning per-date files if the consolidated cache is absent.
    """
    consolidated = consolidated_bars()
    if consolidated is not None and not consolidated.empty:
        out = consolidated[consolidated["symbol"] == symbol]
        return out.sort_values("date").reset_index(drop=True)
    # fallback: scan per-date files (slow, ~5s)
    rows = []
    for f in daily_files():
        df = pd.read_parquet(f)
        sub = df[df["symbol"] == symbol]
        if not sub.empty:
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def consolidated_bars() -> pd.DataFrame | None:
    """All whole-market daily bars in one long-format frame (built once, cached
    to disk). ~6M rows; lets per-symbol / per-date slices be instant. Returns None
    if the daily caches are missing.
    """
    path = _p(CACHE, "wholemkt_bars_long.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    files = daily_files()
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


@functools.lru_cache(maxsize=1)
def market_breadth() -> pd.DataFrame:
    """Per-date whole-market breadth: advancers, decliners, mean/median return, count.

    Computed once across all daily caches. pct_chg is in percent.
    """
    recs = []
    for f in daily_files():
        df = pd.read_parquet(f, columns=["date", "pct_chg"])
        if df.empty:
            continue
        pc = df["pct_chg"].to_numpy(dtype=float)
        d = df["date"].iloc[0]
        recs.append(
            {
                "date": d,
                "advancers": int((pc > 0).sum()),
                "decliners": int((pc < 0).sum()),
                "unchanged": int((pc == 0).sum()),
                "mean_ret": float(np.nanmean(pc)),
                "median_ret": float(np.nanmedian(pc)),
                "limit_up": int((pc >= 9.9).sum()),
                "limit_down": int((pc <= -9.9).sum()),
                "n": len(df),
            }
        )
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)


def output_csv(name: str) -> pd.DataFrame:
    """Load an outputs/*.csv by name (e.g. 'multiseed_robust.csv').

    mtime 感知：文件被重写后自动失效重读，无需手动 refresh_caches()。
    """
    path = _p(OUTPUTS, name)
    return _output_csv_cached(path, _mtime(path))


@functools.lru_cache(maxsize=32)
def _output_csv_cached(path: str, mtime: float | None) -> pd.DataFrame:
    if mtime is None:
        return pd.DataFrame()
    return pd.read_csv(path)


output_csv.cache_clear = _output_csv_cached.cache_clear


@functools.lru_cache(maxsize=1)
def panel_factor_columns() -> tuple[str, ...]:
    """Factor (non-meta) column names of the whole-market panel."""
    import pyarrow.parquet as pq

    path = _p(CACHE, "wholemkt_panel.parquet")
    if not os.path.exists(path):
        return ()
    meta = {
        "date",
        "symbol",
        "responder_6",
        "close",
        "return_1d",
        "pct_chg",
        "amount",
        "weight",
        "date_id",
        "time_id",
        "symbol_id",
    }
    return tuple(c for c in pq.read_schema(path).names if c not in meta)


# --- 交易台 / 模型实验室 / 因子工作台契约 loader ------------------------------
# 全部空态容错：文件缺失/损坏返回 None 或空 DataFrame，不抛异常。


def _read_json(path: str) -> dict | None:
    """容错读 JSON 文件：不存在/损坏返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def latest_orders() -> pd.DataFrame:
    """最新订单 df（统一来源，口径见 latest_orders_with_meta）。

    列契约见 src/trading.py::decide_orders：symbol,name,side,shares,limit_price,
    ref_close,rank_pct,bb_pos,rsi,blocked,asof_date。

    mtime 感知：CLI 或 nextday 会话落盘新订单后，交易台轮询无需
    refresh_caches() 即可看到（最新来源的路径或 mtime 变化都会失效重读）。
    """
    return latest_orders_with_meta()[0]


def latest_orders_with_meta() -> tuple[pd.DataFrame, dict]:
    """最新订单 + 来源信息：(df, {"date","mtime","source","path"})；无订单 → (空表, {})。

    统一来源：CLI 产物 outputs/orders_*.csv 与已完成 nextday 会话的
    artifacts/orders.csv，取【订单日期最新】的一份（同日期取 mtime 最新）。
    订单日期口径：CLI 取文件名日期；会话取 asof_date 列（缺失回落会话完成时间）。
    """
    best = None  # (date, mtime, source, path)
    for f in glob.glob(_p(OUTPUTS, "orders_*.csv")):
        d = os.path.basename(f).replace("orders_", "").replace(".csv", "")
        mt = _mtime(f) or 0.0
        if best is None or (d, mt) > (best[0], best[1]):
            best = (d, mt, "cli", f)
    for mp in glob.glob(_p(OUTPUTS, "sessions", "nextday_*", "session.json")):
        meta = _read_json(mp)
        if not meta or meta.get("status") != "done":
            continue
        opath = os.path.join(os.path.dirname(mp), "artifacts", "orders.csv")
        mt = _mtime(opath)
        if mt is None:
            continue
        d = _orders_asof_date(opath) or _session_finished_date(meta)
        if not d:
            continue
        if best is None or (d, mt) > (best[0], best[1]):
            best = (d, mt, f"会话 {meta.get('id')}", opath)
    if best is None:
        return pd.DataFrame(), {}
    d, mt, source, path = best
    return _latest_orders_cached(path, mt), {
        "date": d,
        "mtime": mt,
        "source": source,
        "path": path,
    }


def _orders_asof_date(path: str) -> str | None:
    """订单 csv 的 asof_date 列（YYYYMMDD）；缺列/读取失败 → None。"""
    try:
        col = pd.read_csv(path, usecols=["asof_date"])["asof_date"].dropna()
    except Exception:
        return None
    if not len(col):
        return None
    return str(col.iloc[-1])[:10].replace("-", "")


def _session_finished_date(meta: dict) -> str | None:
    """会话完成时间 → YYYYMMDD；非法 → None。"""
    try:
        return datetime.fromtimestamp(float(meta.get("finished") or 0)).strftime("%Y%m%d")
    except (TypeError, ValueError, OSError):
        return None


@functools.lru_cache(maxsize=4)
def _latest_orders_cached(path: str, mtime: float | None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


latest_orders.cache_clear = _latest_orders_cached.cache_clear


def pred_hist() -> pd.DataFrame:
    """持久化预测历史 outputs/live_risk_state_pred_hist.parquet。

    列契约见 src/live_state.py::append_pred_hist：date,symbol,score,pred_<模型腿>。
    交易台订单「分析」面板的模型明细数据源。缺失/损坏返回空表。mtime 感知失效。
    """
    path = _p(OUTPUTS, "live_risk_state_pred_hist.parquet")
    return _parquet_cached(path, _mtime(path))


@functools.lru_cache(maxsize=8)
def _parquet_cached(path: str, mtime: float | None) -> pd.DataFrame:
    """parquet 版 _csv_cached（mtime 感知，空态容错）。"""
    if mtime is None:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


pred_hist.cache_clear = _parquet_cached.cache_clear


def holdings() -> pd.DataFrame:
    """当前实盘持仓 outputs/holdings.csv；缺失返回空表。mtime 感知失效。"""
    path = _p(OUTPUTS, "holdings.csv")
    return _csv_cached(path, _mtime(path))


def account() -> pd.DataFrame:
    """实盘账户快照 outputs/account.csv；缺失返回空表。mtime 感知失效。"""
    path = _p(OUTPUTS, "account.csv")
    return _csv_cached(path, _mtime(path))


# --- 交易台「今日流程」写操作（②勾选持久化 / ④现金维护）与交易日判定 ----------


def _execution_state_path(day: str) -> str:
    return _p(OUTPUTS, f"execution_state_{day}.json")


def load_execution_state(day: str) -> dict:
    """执行勾选状态 outputs/execution_state_<day>.json（{side-symbol: true}）；缺失/损坏 → {}。"""
    data = _read_json(_execution_state_path(day))
    return data if isinstance(data, dict) else {}


def save_execution_state(day: str, state: dict) -> None:
    """勾选状态写盘；失败静默（降级为纯前端态）。"""
    try:
        with open(_execution_state_path(day), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception:
        pass


def append_account_cash(day_iso: str, cash: float) -> None:
    """向 outputs/account.csv 追加一行 {date, cash}；文件不存在则创建（表头 date,cash）。

    已有文件的表头缺 date/cash 列时抛 ValueError（列结构对不齐，追加会错位）。
    """
    path = _p(OUTPUTS, "account.csv")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("date,cash\n")
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    cols = [c.strip() for c in header.split(",") if c.strip()]
    if "date" not in cols or "cash" not in cols:
        raise ValueError(f"account.csv 表头缺 date/cash 列（实际：{header!r}），请手工修复")
    row = {"date": day_iso, "cash": f"{float(cash):.2f}"}
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(row.get(c, "") for c in cols) + "\n")


@functools.lru_cache(maxsize=4)
def is_trading_day(day_iso: str) -> bool | None:
    """某日是否 A 股交易日（日历有缓存，廉价）；日历不可用返回 None（未知，UI 不置灰）。"""
    try:
        from data.tushare import TushareConnector

        return bool(TushareConnector().is_trading_day(day_iso))
    except Exception:
        return None


@functools.lru_cache(maxsize=16)
def _csv_cached(path: str, mtime: float | None) -> pd.DataFrame:
    """holdings/account 共用的 mtime 感知 CSV reader（空态容错）。"""
    if mtime is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


holdings.cache_clear = _csv_cached.cache_clear
account.cache_clear = _csv_cached.cache_clear


@functools.lru_cache(maxsize=1)
def live_risk_state() -> dict | None:
    """实盘风控状态 outputs/live_risk_state.json；缺失/损坏返回 None。"""
    return _read_json(_p(OUTPUTS, "live_risk_state.json"))


@functools.lru_cache(maxsize=1)
def tracking_report() -> dict | None:
    """实盘跟踪报告 outputs/tracking/tracking_report.json；缺失返回 None。"""
    return _read_json(_p(TRACKING, "tracking_report.json"))


@functools.lru_cache(maxsize=1)
def live_daily() -> pd.DataFrame:
    """实盘逐日净值 outputs/tracking/live_daily.parquet；缺失返回空表。"""
    path = _p(TRACKING, "live_daily.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@functools.lru_cache(maxsize=1)
def champion_manifest() -> dict | None:
    """models/champion/manifest.json（含 model_id/created_at/config_snapshot）。

    无 champion（models/ 为空或文件缺失/损坏）返回 None。
    """
    return _read_json(_p(MODELS, "champion", "manifest.json"))


@functools.lru_cache(maxsize=1)
def staging_manifests() -> tuple[dict, ...]:
    """models/staging/*/ 的全部候选：每条形如
    {"id": <目录名>, "manifest": dict|None, "evaluation": dict|None}。
    staging 目录不存在返回空 tuple。
    """
    sdir = _p(MODELS, "staging")
    if not os.path.isdir(sdir):
        return ()
    out = []
    for name in sorted(os.listdir(sdir)):
        d = _p(sdir, name)
        if not os.path.isdir(d):
            continue
        out.append(
            {
                "id": name,
                "manifest": _read_json(_p(d, "manifest.json")),
                "evaluation": _read_json(_p(d, "evaluation.json")),
            }
        )
    return tuple(out)


# 因子定义 DataFrame 的列契约（与 src/data/factors/schema.py::FactorDefinition 对齐）
_FACTOR_COLS = [
    "id",
    "name",
    "description",
    "category",
    "source",
    "formula",
    "tags",
    "expected_sign",
    "enabled",
    "version",
    "author",
    "created",
    "explicit_deps",
]


@functools.lru_cache(maxsize=1)
def factor_library() -> pd.DataFrame:
    """factor_library/*.yaml 的全部因子定义，一行一个因子。

    字段与 src/data/factors/schema.py::FactorDefinition 对齐；tags/explicit_deps
    保留为 list。目录缺失/单文件损坏均容错（坏文件跳过），_manifest.yaml 的
    exclude_ids 生效。无任何定义时返回带列名的空表。
    """
    import yaml

    lib = FACTOR_LIBRARY
    if not os.path.isdir(lib):
        return pd.DataFrame(columns=_FACTOR_COLS)
    # _manifest.yaml 的排除名单（与 schema 加载器行为一致）
    exclude: set = set()
    mpath = _p(lib, "_manifest.yaml")
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                mdata = yaml.safe_load(f) or {}
            exclude = {str(x) for x in mdata.get("exclude_ids", [])}
        except Exception:
            exclude = set()

    rows = []
    seen: set = set()
    files = sorted(glob.glob(_p(lib, "*.yaml"))) + sorted(glob.glob(_p(lib, "*.yml")))
    files = [f for f in files if os.path.basename(f) not in ("_manifest.yaml", "_manifest.yml")]
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            raw_list = data.get("factors", []) if isinstance(data, dict) else []
        except Exception:
            continue  # 坏文件跳过，不拖垮整库
        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            fid = raw.get("id")
            if not fid or fid in exclude or fid in seen:
                continue
            seen.add(fid)
            rows.append(
                {
                    "id": fid,
                    "name": raw.get("name") or fid,
                    "description": raw.get("description", ""),
                    "category": raw.get("category", "general"),
                    "source": raw.get("source", ""),
                    "formula": raw.get("formula", ""),
                    "tags": list(raw.get("tags") or []),
                    "expected_sign": raw.get("expected_sign"),
                    "enabled": bool(raw.get("enabled", True)),
                    "version": int(raw.get("version", 1)),
                    "author": raw.get("author", ""),
                    "created": raw.get("created", ""),
                    "explicit_deps": list(raw.get("explicit_deps") or []),
                }
            )
    return pd.DataFrame(rows, columns=_FACTOR_COLS)


def factor_eval_table() -> pd.DataFrame:
    """全量因子评估结果 outputs/factor_eval.csv；缺失返回空表。

    mtime 感知：factoreval 会话落盘新 csv 后自动失效重读。
    """
    path = _p(OUTPUTS, "factor_eval.csv")
    return _csv_cached(path, _mtime(path))


def factor_ic_decay() -> pd.DataFrame:
    """单因子逐窗口 IC 衰减 outputs/diag_factor_ic_decay.csv（原 factor_lab 数据源）。
    mtime 感知失效。"""
    path = _p(OUTPUTS, "diag_factor_ic_decay.csv")
    return _csv_cached(path, _mtime(path))


def factor_corr() -> pd.DataFrame:
    """因子相关矩阵 outputs/diag_factor_corr.csv（原 factor_lab 数据源）。
    mtime 感知失效。"""
    path = _p(OUTPUTS, "diag_factor_corr.csv")
    return _csv_cached(path, _mtime(path))


factor_eval_table.cache_clear = _csv_cached.cache_clear
factor_ic_decay.cache_clear = _csv_cached.cache_clear
factor_corr.cache_clear = _csv_cached.cache_clear


def warm_load() -> dict:
    """Pre-load the light caches at startup; return a small status dict."""
    seeds = available_seeds()
    for sd in seeds:
        seed_oos(sd)
    amount_map()
    daily_files()
    universe()
    return {
        "seeds": seeds,
        "n_daily_files": len(daily_files()),
        "n_symbols": len(universe()),
        "amount_available": oos_with_amount_keys()[0],
        "n_factors": len(panel_factor_columns()),
    }


# --- cache lifecycle ---------------------------------------------------------
def refresh_caches() -> None:
    """回填/新数据落盘后调用：清空全部 memoized loader，下次访问重新读盘。"""
    for fn in (
        available_seeds,
        seed_oos,
        amount_map,
        oos_with_amount_keys,
        daily_files,
        trade_dates,
        universe,
        symbol_bars,
        consolidated_bars,
        market_breadth,
        output_csv,
        panel_factor_columns,
        latest_orders,
        pred_hist,
        holdings,
        account,
        is_trading_day,
        live_risk_state,
        tracking_report,
        live_daily,
        champion_manifest,
        staging_manifests,
        factor_library,
        factor_eval_table,
        factor_ic_decay,
        factor_corr,
    ):
        fn.cache_clear()


def preheat_async() -> threading.Thread:
    """后台线程预热重型首次 IO（consolidated_bars / market_breadth）。

    两者各要读全部 daily_*.parquet（~6M 行），同步做会拖慢启动数秒；
    放 daemon 线程里跑，回调若先触发也只是重复一次 IO（lru_cache 在 GIL 下
    并发安全，最坏结果是算两遍，不会损坏状态）。预热失败不致命。
    """

    def _job():
        try:
            consolidated_bars()
            market_breadth()
        except Exception:
            pass

    t = threading.Thread(target=_job, name="dataio-preheat", daemon=True)
    t.start()
    return t
