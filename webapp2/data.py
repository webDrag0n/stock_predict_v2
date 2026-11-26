"""前端数据层：页面只经这里读文件，不直接碰磁盘；每个读取函数缺失/损坏都返回空态，不抛异常。

读取约定：
  · 一切数据来自 outputs/、models/、factor_library/ 下的文件，或 run_registry 的 outputs/runs/。
    本模块不启动任何子进程、不联网、不训练。
  · 缓存键含文件 mtime：CLI 在另一个进程里写完文件，下一次读取立刻看到新内容。
  · 路径经模块级变量（ROOT/OUTPUTS/MODELS/...）解析，测试 monkeypatch 它们即可隔离真实目录。
"""

from __future__ import annotations

import functools
import glob
import json
import os
import re
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
MODELS = os.path.join(ROOT, "models")
FACTOR_LIBRARY = os.path.join(ROOT, "factor_library")
CACHE = os.path.join(ROOT, "data_cache")

_ORDERS_RE = re.compile(r"^orders_(\d{8})\.csv$")


def _p(*parts: str) -> str:
    return os.path.join(*parts)


def mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def read_json(path: str) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@functools.lru_cache(maxsize=64)
def _csv(path: str, mt: float | None) -> pd.DataFrame:
    if mt is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@functools.lru_cache(maxsize=64)
def _parquet(path: str, mt: float | None) -> pd.DataFrame:
    if mt is None:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@functools.lru_cache(maxsize=64)
def _json_cached(path: str, mt: float | None):
    return read_json(path) if mt is not None else None


def csv(path: str) -> pd.DataFrame:
    return _csv(path, mtime(path))


def parquet(path: str) -> pd.DataFrame:
    return _parquet(path, mtime(path))


def jsonf(path: str):
    return _json_cached(path, mtime(path))


def refresh() -> None:
    """清全部缓存（测试与「刷新」按钮用）。"""
    for fn in (
        _csv,
        _parquet,
        _json_cached,
        _latest_data_day,
        _close_maps,
        _iv_scale,
        _panel_columns,
        _panel_all_columns,
    ):
        fn.cache_clear()


# --------------------------------------------------------------------- 订单 / 账户 / 持仓
def order_files() -> list[tuple[str, str]]:
    """[(YYYYMMDD, 路径)]，新到旧。"""
    out = []
    for p in glob.glob(_p(OUTPUTS, "orders_*.csv")):
        m = _ORDERS_RE.match(os.path.basename(p))
        if m:
            out.append((m.group(1), p))
    return sorted(out, reverse=True)


def latest_orders() -> tuple[pd.DataFrame, str | None]:
    """最新一份订单与其日期（YYYYMMDD）。无订单 → (空表, None)。"""
    files = order_files()
    if not files:
        return pd.DataFrame(), None
    day, path = files[0]
    return csv(path), day


def holdings() -> pd.DataFrame:
    return csv(_p(OUTPUTS, "holdings.csv"))


def holdings_history(n: int = 100) -> pd.DataFrame:
    """最近的持仓快照行，新到旧；每份快照用 snapshot_id 分组。"""
    frame = csv(_p(OUTPUTS, "holdings_history.csv"))
    if frame.empty:
        return frame
    return frame.tail(max(1, int(n))).iloc[::-1].reset_index(drop=True)


_DAILY_RE = re.compile(r"^daily_(\d{8})\.parquet$")


@functools.lru_cache(maxsize=4)
def _latest_data_day(dir_mtime: float | None, path: str) -> str | None:
    if dir_mtime is None:
        return None
    try:
        days = [m.group(1) for n in os.listdir(path) if (m := _DAILY_RE.match(n))]
    except OSError:
        return None
    return max(days) if days else None


def latest_data_day() -> str | None:
    """行情缓存里最新交易日（YYYYMMDD），取自 data_cache/tushare/daily_*.parquet 的文件名。

    订单日期早于它，说明订单是用更旧的数据生成的，已过期。缓存目录 mtime 变了才重扫。
    """
    path = _p(CACHE, "tushare")
    return _latest_data_day(mtime(path), path)


@functools.lru_cache(maxsize=4)
def _close_maps(key: tuple) -> tuple[dict, dict, str | None]:
    def _load(path: str) -> dict:
        try:
            df = pd.read_parquet(path, columns=["symbol", "close"])
        except Exception:
            return {}
        return {str(s): float(c) for s, c in zip(df["symbol"], df["close"]) if c == c and c > 0}

    if not key:
        return {}, {}, None
    last_path = key[-1][0]
    m = _DAILY_RE.match(os.path.basename(last_path))
    prev = _load(key[-2][0]) if len(key) > 1 else {}
    return prev, _load(last_path), (m.group(1) if m else None)


def close_maps() -> tuple[dict, dict, str | None]:
    """最近两个交易日的收盘价 (前一日, 最新日, 最新日期YYYYMMDD)。

    只读尾部两个 daily_*.parquet（各约 5k 行），不加载 600 万行的合并面板。
    """
    files = sorted(glob.glob(_p(CACHE, "tushare", "daily_????????.parquet")))[-2:]
    return _close_maps(tuple((f, mtime(f)) for f in files))


@functools.lru_cache(maxsize=2)
def _iv_scale(mt: float | None, path: str) -> tuple[str, float | None]:
    import config as C

    if not getattr(C, "USE_IV_SCALE", False):
        return "disabled", None
    if mt is None:
        return "unknown", None
    try:
        import pyarrow.parquet as pq

        feat = getattr(C, "IV_FEATURE_FOR_SCALE", "opt_vix_like")
        if feat not in pq.read_schema(path).names:
            return "unknown", None
        panel = pd.read_parquet(path, columns=["date", feat])
        if panel.empty or panel[feat].dropna().empty:
            return "unknown", None
        from live_state import _iv_scale as _calc

        v = _calc(
            panel,
            panel["date"].max(),
            feat,
            int(getattr(C, "IV_SCALE_WINDOW", 60)),
            float(getattr(C, "IV_MIN_SCALE", 0.5)),
        )
        return "ok", float(v)
    except Exception:
        return "unknown", None


def iv_scale_now() -> tuple[str, float | None]:
    """IV 降仓乘数 ('disabled'|'unknown'|'ok', scale)，口径同 live_state._iv_scale。"""
    # 夜间任务实际使用的面板缓存（outputs 里的 IV 降仓就是从它算的）。旧前端读的
    # wholemkt_panel.parquet 在本仓库里并不存在，导致 IV 层永远显示「数据不可得」，
    # 总乘数也漏掉这一层。
    path = _p(CACHE, "factor_panel_cache.parquet")
    return _iv_scale(mtime(path), path)


def account() -> pd.DataFrame:
    return csv(_p(OUTPUTS, "account.csv"))


def account_status(max_age_days: int = 4) -> dict:
    """账户文件新鲜度：供总览与体检提示共用。"""
    path = _p(OUTPUTS, "account.csv")
    mt = mtime(path)
    df = account()
    cash = None
    if not df.empty and "cash" in df.columns:
        s = pd.to_numeric(df["cash"], errors="coerce").dropna()
        cash = float(s.iloc[-1]) if len(s) else None
    age = None if mt is None else (time.time() - mt) / 86400.0
    return {
        "exists": mt is not None,
        "cash": cash,
        "age_days": age,
        "stale": age is None or age > max_age_days,
        "max_age_days": max_age_days,
        "path": path,
    }


def orders_mtime(day: str | None) -> float | None:
    return mtime(_p(OUTPUTS, f"orders_{day}.csv")) if day else None


def holdings_mtime() -> float | None:
    return mtime(_p(OUTPUTS, "holdings.csv"))


def execution_state(day: str) -> dict:
    d = jsonf(_p(OUTPUTS, f"execution_state_{day}.json"))
    return d if isinstance(d, dict) else {}


_EXECUTION_STATE_RE = re.compile(r"^execution_state_(\d{8})\.json$")


def execution_state_files() -> list[tuple[str, str]]:
    """[(执行日, 路径)]，新到旧，用于跨日展示落实状态。"""
    out = []
    for path in glob.glob(_p(OUTPUTS, "execution_state_*.json")):
        match = _EXECUTION_STATE_RE.match(os.path.basename(path))
        if match:
            out.append((match.group(1), path))
    return sorted(out, reverse=True)


def latest_execution_state() -> tuple[str | None, dict]:
    files = execution_state_files()
    if not files:
        return None, {}
    day, path = files[0]
    state = jsonf(path)
    return day, state if isinstance(state, dict) else {}


# --------------------------------------------------------------------- 风控 / 跟踪
def risk_state() -> dict | None:
    d = jsonf(_p(OUTPUTS, "live_risk_state.json"))
    return d if isinstance(d, dict) else None


def rebalance_state() -> dict | None:
    d = jsonf(_p(OUTPUTS, "live_rebalance_state.json"))
    return d if isinstance(d, dict) else None


def tracking_report() -> dict | None:
    d = jsonf(_p(OUTPUTS, "tracking", "tracking_report.json"))
    return d if isinstance(d, dict) else None


def live_daily() -> pd.DataFrame:
    return parquet(_p(OUTPUTS, "tracking", "live_daily.parquet"))


def pred_hist() -> pd.DataFrame:
    return parquet(_p(OUTPUTS, "live_risk_state_pred_hist.parquet"))


def alerts_log(n: int = 50) -> list[dict]:
    path = _p(OUTPUTS, "alerts.log")
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f.readlines()[-n:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    rows.append({"ts": "", "text": line.strip()})
    except OSError:
        return []
    return rows[::-1]


# --------------------------------------------------------------------- 模型
def champion_manifest() -> dict | None:
    d = jsonf(_p(MODELS, "champion", "manifest.json"))
    return d if isinstance(d, dict) else None


def champion_evaluation() -> dict | None:
    d = jsonf(_p(MODELS, "champion", "evaluation.json"))
    return d if isinstance(d, dict) else None


def promotion_log(n: int = 30) -> list[str]:
    """models/promotions.log 末尾 n 行（新到旧）：晋升/删除/现金/确认成交的审计流水。"""
    try:
        with open(_p(MODELS, "promotions.log"), encoding="utf-8", errors="replace") as f:
            return [ln.rstrip("\n") for ln in f.readlines()[-n:]][::-1]
    except OSError:
        return []


def models_signature() -> list:
    """模型目录的变更签名：champion/staging/archive 各目录名与 manifest/evaluation 的 mtime。"""
    sig: list = [mtime(_p(MODELS, "promotions.log"))]
    for sub in ("champion", "staging", "archive"):
        base = _p(MODELS, sub)
        if sub == "champion":
            sig.append([mtime(_p(base, f)) for f in ("manifest.json", "evaluation.json")])
            continue
        try:
            names = sorted(os.listdir(base))
        except OSError:
            names = []
        sig.append(
            [
                (n, mtime(_p(base, n, "manifest.json")), mtime(_p(base, n, "evaluation.json")))
                for n in names
            ]
        )
    return sig


def _list_models(sub: str) -> list[dict]:
    base = _p(MODELS, sub)
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        d = _p(base, name)
        if not os.path.isdir(d):
            continue
        man = jsonf(_p(d, "manifest.json"))
        ev = jsonf(_p(d, "evaluation.json"))
        out.append(
            {
                "id": name,
                "manifest": man if isinstance(man, dict) else None,
                "evaluation": ev if isinstance(ev, dict) else None,
                "mtime": mtime(d),
            }
        )
    return out


def staging_models() -> list[dict]:
    return sorted(_list_models("staging"), key=lambda r: r["mtime"] or 0, reverse=True)


def archived_models() -> list[dict]:
    return sorted(_list_models("archive"), key=lambda r: r["mtime"] or 0, reverse=True)


# --------------------------------------------------------------------- 回测与实验
def walkforward_meta() -> dict | None:
    d = jsonf(_p(OUTPUTS, "walkforward", "meta.json"))
    return d if isinstance(d, dict) else None


def walkforward_windows(path: str | None = None) -> pd.DataFrame:
    return parquet(path or _p(OUTPUTS, "walkforward", "window_metrics.parquet"))


def walkforward_curve(path: str | None = None) -> pd.DataFrame:
    return parquet(path or _p(OUTPUTS, "walkforward", "oos_curve.parquet"))


def experiment_runs() -> list[dict]:
    """outputs/experiments/**/walkforward/meta.json 与 outputs/walkforward：横向对比用。"""
    out = []
    metas = [(_p(OUTPUTS, "walkforward"), "当前 canonical")]
    for p in sorted(
        glob.glob(_p(OUTPUTS, "experiments", "**", "walkforward", "meta.json"), recursive=True)
    ):
        d = os.path.dirname(p)
        metas.append((d, os.path.relpath(os.path.dirname(d), _p(OUTPUTS, "experiments"))))
    for d, label in metas:
        m = jsonf(_p(d, "meta.json"))
        if isinstance(m, dict):
            out.append({"label": label, "dir": d, "meta": m})
    return out


def trials(n: int = 200) -> list[dict]:
    path = _p(OUTPUTS, "experiments", "trials.jsonl")
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f.readlines()[-n:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return []
    return rows


def impact_sensitivity() -> pd.DataFrame:
    return csv(_p(OUTPUTS, "experiments", "impact_sensitivity.csv"))


# --------------------------------------------------------------------- 因子
def factor_eval_table() -> pd.DataFrame:
    return csv(_p(OUTPUTS, "factor_eval.csv"))


def panel_cache_meta() -> dict | None:
    """因子面板缓存的签名/覆盖范围（data_cache/factor_panel_cache.sig.json）。"""
    d = jsonf(_p(CACHE, "factor_panel_cache.sig.json"))
    return d if isinstance(d, dict) else None


def factor_ic_decay() -> pd.DataFrame:
    return csv(_p(OUTPUTS, "diag_factor_ic_decay.csv"))


def factor_corr() -> pd.DataFrame:
    return csv(_p(OUTPUTS, "diag_factor_corr.csv"))


def model_vs_factor() -> pd.DataFrame:
    return csv(_p(OUTPUTS, "diag_model_vs_factor.csv"))


def _safe_fid(fid: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(fid)).strip("_") or "factor"


def factor_layers(fid: str) -> pd.DataFrame:
    """分层净值 outputs/factor_layer_<id>.parquet（文件名规则同 src/factor_layers）。"""
    return parquet(_p(OUTPUTS, f"factor_layer_{_safe_fid(fid)}.parquet"))


def factor_daily_ic(fid: str) -> pd.DataFrame:
    return parquet(_p(OUTPUTS, f"factor_ic_daily_{_safe_fid(fid)}.parquet"))


_PANEL_META = frozenset(
    {
        "date",
        "symbol",
        "responder_6",
        "close",
        "open",
        "high",
        "low",
        "adj_close",
        "return_1d",
        "pct_chg",
        "amount",
        "weight",
        "date_id",
        "time_id",
        "symbol_id",
        "bb_upper_20",
        "bb_lower_20",
    }
)


@functools.lru_cache(maxsize=2)
def _panel_columns(mt: float | None, path: str) -> tuple[str, ...]:
    if mt is None:
        return ()
    try:
        import pyarrow.parquet as pq

        names = pq.read_schema(path).names
    except Exception:
        return ()
    return tuple(c for c in names if c not in _PANEL_META and not c.startswith("responder_fw_"))


def panel_factor_columns() -> tuple[str, ...]:
    """因子面板缓存里的因子列（只读 parquet 元数据，不加载数据）。"""
    path = _p(CACHE, "factor_panel_cache.parquet")
    return _panel_columns(mtime(path), path)


@functools.lru_cache(maxsize=2)
def _panel_all_columns(mt: float | None, path: str) -> tuple[str, ...]:
    if mt is None:
        return ()
    try:
        import pyarrow.parquet as pq

        return tuple(pq.read_schema(path).names)
    except Exception:
        return ()


def base_columns() -> frozenset[str]:
    """公式可以引用的列：行情缓存的原始列 + 面板里已有的全部列。"""
    path = _p(CACHE, "factor_panel_cache.parquet")
    cols = set(_panel_all_columns(mtime(path), path))
    bars = _p(CACHE, "wholemkt_bars_long.parquet")
    cols |= set(_panel_all_columns(mtime(bars), bars))
    return frozenset(cols)


def factor_library() -> pd.DataFrame:
    """factor_library/*.yaml 的全部因子定义（与加载器同样：跳过 manifest 排除名单与重复 id）。"""
    import glob as _glob

    cols = [
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
        "file",
    ]
    lib = FACTOR_LIBRARY
    if not os.path.isdir(lib):
        return pd.DataFrame(columns=cols)
    exclude: set = set()
    man = read_json_yaml(_p(lib, "_manifest.yaml"))
    if isinstance(man, dict):
        exclude = {str(x) for x in man.get("exclude_ids", []) or []}
    rows, seen = [], set()
    files = sorted(_glob.glob(_p(lib, "*.yaml"))) + sorted(_glob.glob(_p(lib, "*.yml")))
    for path in files:
        if os.path.basename(path) in ("_manifest.yaml", "_manifest.yml"):
            continue
        d = read_json_yaml(path)
        raw_list = d.get("factors", []) if isinstance(d, dict) else []
        for raw in raw_list or []:
            if (
                not isinstance(raw, dict)
                or not raw.get("id")
                or raw["id"] in exclude
                or raw["id"] in seen
            ):
                continue
            seen.add(raw["id"])
            rows.append(
                {
                    "id": raw["id"],
                    "name": raw.get("name") or raw["id"],
                    "description": raw.get("description", ""),
                    "category": raw.get("category", "general"),
                    "source": raw.get("source", ""),
                    "formula": raw.get("formula", ""),
                    "tags": list(raw.get("tags") or []),
                    "expected_sign": raw.get("expected_sign"),
                    "enabled": bool(raw.get("enabled", True)),
                    "version": int(raw.get("version", 1) or 1),
                    "author": raw.get("author", ""),
                    "created": raw.get("created", ""),
                    "file": os.path.basename(path),
                }
            )
    return pd.DataFrame(rows, columns=cols)


def read_json_yaml(path: str):
    """读 yaml（缺失/损坏 → None）。"""
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def factor_contrib() -> tuple[pd.DataFrame, pd.Series]:
    """champion 内逐因子贡献长表与综合贡献分。缺失/损坏 → (空表, 空序列)。"""
    try:
        import factor_contrib as fc

        contrib = fc.load_champion_contrib(MODELS)
        if contrib.empty:
            return contrib, pd.Series(dtype=float)
        return contrib, fc.combined_scores(contrib, fc.champion_leg_weights(MODELS))
    except Exception:
        return pd.DataFrame(columns=["factor", "source", "kind", "value", "asof"]), pd.Series(
            dtype=float
        )


def library_signature() -> list:
    """因子库目录的变更签名：各 yaml 的 mtime。"""
    try:
        names = sorted(n for n in os.listdir(FACTOR_LIBRARY) if n.endswith((".yaml", ".yml")))
    except OSError:
        return []
    return [(n, mtime(_p(FACTOR_LIBRARY, n))) for n in names]


# --------------------------------------------------------------------- 一次性总览快照
def overview() -> dict:
    """总览页所需的全部数据，一次读完（各项缺失时对应键为 None/空）。"""
    orders, day = latest_orders()
    return {
        "orders": orders,
        "orders_day": day,
        "data_day": latest_data_day(),
        "account": account_status(),
        "risk_state": risk_state(),
        "rebalance_state": rebalance_state(),
        "tracking": tracking_report(),
        "champion": champion_manifest(),
        "alerts": alerts_log(10),
    }
