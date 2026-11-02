"""champion 模型内逐因子贡献提取（前端因子工作台「模型内贡献」的后端契约）。

设计文档 docs/frontend_redesign.html 5.4 第 ② 项：
champion 内逐因子权重/重要性 → 「因子 → 模型内贡献」长表。

数据源
------
· ``models/champion/ensemble_state.pkl`` —— ``IncrementalEnsemble.state_dict()``
  的快照（结构见 src/ensemble.py）：各腿滚动 IC 权重（``weights``）、ic_linear
  腿的训练段 IC 权重（``linear_state``: feats + weights）、GBM 腿的 booster
  文本（``model_to_string``，可还原 gain 重要性）。
· ``models/champion/manifest.json`` —— ``config_snapshot.feat_cols`` 提供特征列
  顺序（booster 内部列名是 Column_N，按位置映射回因子名）。
· ``outputs/sessions/*/artifacts/model_weights.parquet`` —— backtest 会话落盘
  的逐日各腿 IC 权重时序（``w_<model>`` / ``ic_<model>`` 列），供详情页 P5
  的「腿权重随时间」图。

输出表契约（长表，列固定）::

    factor  因子名（panel 列名）
    source  来源腿：ic_linear / gbm_l2 / gbm_ret_weighted / ...
    kind    取值类型：ic_weight（带符号训练段 IC 权重）/
            gbm_importance（gain，腿内归一，和为 1）
    value   原始取值
    asof    状态截止日（last_trained_date 或工件 mtime）

champion 不存在 / 损坏 / 无任何可用信息时返回带列名的空表（不抛异常）。
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLUMNS = ["factor", "source", "kind", "value", "asof"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


# ---------------------------------------------------------------------------
# state_dict → 贡献长表
# ---------------------------------------------------------------------------
def _map_feat_name(name: str, feat_cols: list[str]) -> str | None:
    """booster 内部列名映射回因子名：原名直接用；Column_N 按位置映射。"""
    if name in feat_cols:
        return name
    m = re.fullmatch(r"Column_(\d+)", str(name))
    if m:
        i = int(m.group(1))
        if 0 <= i < len(feat_cols):
            return feat_cols[i]
    return None


def _gbm_importance_rows(
    model_strs: list[str], feat_cols: list[str], source: str, asof
) -> list[tuple]:
    """一组 booster 文本 → (factor, source, gbm_importance, value, asof) 行。

    多 seed 的 gain 重要性直接求和后归一（同一特征空间，量纲一致）。
    单个 booster 损坏跳过，不拖垮整表。
    """
    import lightgbm as lgb

    acc: dict[str, float] = {}
    for s in model_strs or []:
        try:
            bst = lgb.Booster(model_str=s)
        except Exception:
            continue
        imp = np.asarray(bst.feature_importance(importance_type="gain"), dtype=float)
        for name, v in zip(bst.feature_name(), imp):
            f = _map_feat_name(name, feat_cols)
            if f is None or not np.isfinite(v) or v <= 0:
                continue
            acc[f] = acc.get(f, 0.0) + float(v)
    tot = sum(acc.values())
    if tot <= 0:
        return []
    return [(f, source, "gbm_importance", v / tot, asof) for f, v in sorted(acc.items())]


def contrib_from_state(state: dict, feat_cols: list[str] | None = None, asof=None) -> pd.DataFrame:
    """从 ``IncrementalEnsemble.state_dict()`` 快照提取因子贡献长表。

    不依赖 ensemble 类本身（state_dict 是纯 dict，pickle.load 即可），
    也不重建 booster 对象树（逐条 model_to_string 解析，坏的跳过）。
    """
    if not isinstance(state, dict):
        return _empty()
    feat_cols = [str(c) for c in (feat_cols or [])]
    if asof is None:
        asof = str(state.get("last_trained_date") or "")[:10] or None

    rows: list[tuple] = []
    # ic_linear 腿：训练段 IC 权重（带符号；含 nl_/ix_ 变换列）
    lin = state.get("linear_state") or {}
    feats = list(lin.get("feats") or [])
    w = lin.get("weights")
    if isinstance(w, dict):
        # linear_fit 的 weights 是 feat→weight dict（zip 迭代 dict 会拿到键名，
        # 曾因此 float('amount') 崩溃）；只取 feats 里有的键
        items = [(f, w[f]) for f in feats if f in w]
    elif w is not None and len(feats) == len(w):
        items = list(zip(feats, w))  # 旧版按位置对齐的 list/tuple
    else:
        items = []
    for f, wi in items:
        try:
            v = float(wi)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            rows.append((str(f), "ic_linear", "ic_weight", v, asof))
    # GBM 腿：gain 重要性（腿内归一）
    for model, strs in (state.get("boosters") or {}).items():
        rows.extend(_gbm_importance_rows(list(strs or []), feat_cols, str(model), asof))
    if not rows:
        return _empty()
    return pd.DataFrame(rows, columns=COLUMNS)


def combined_scores(contrib: pd.DataFrame, leg_weights: dict | None = None) -> pd.Series:
    """把长表合成「因子 → 综合贡献分」：腿内归一 |值| × 腿权，跨腿求和。

    leg_weights 为 None 时各腿等权。返回按分值降序的 Series（index=因子名）。
    """
    if contrib is None or contrib.empty:
        return pd.Series(dtype=float)
    df = contrib.copy()
    df["abs_v"] = df["value"].astype(float).abs()
    grp = df.groupby(["source", "kind"])["abs_v"].transform("sum")
    df["norm"] = np.where(grp > 0, df["abs_v"] / grp.where(grp > 0, 1.0), 0.0)
    if leg_weights:
        lw = df["source"].map(lambda s: max(float(leg_weights.get(s, 0.0)), 0.0))
    else:
        lw = 1.0
    df["score"] = df["norm"] * lw
    return df.groupby("factor")["score"].sum().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# champion 工件读取
# ---------------------------------------------------------------------------
def _default_models_dir() -> str:
    try:
        import config as C

        return C.MODEL_DIR
    except Exception:
        return os.path.join(ROOT, "models")


def champion_leg_weights(models_dir: str | None = None) -> dict:
    """champion 各腿当前滚动 IC 权重（state["weights"]）；无 champion → {}。"""
    d = os.path.join(models_dir or _default_models_dir(), "champion")
    spath = os.path.join(d, "ensemble_state.pkl")
    if not os.path.exists(spath):
        return {}
    try:
        with open(spath, "rb") as f:
            state = pickle.load(f)
        return {str(m): float(w) for m, w in (state.get("weights") or {}).items()}
    except Exception:
        return {}


def load_champion_contrib(models_dir: str | None = None) -> pd.DataFrame:
    """读取 models/champion 的因子贡献长表；缺失/损坏/无信息 → 空表。"""
    d = os.path.join(models_dir or _default_models_dir(), "champion")
    mpath = os.path.join(d, "manifest.json")
    spath = os.path.join(d, "ensemble_state.pkl")
    if not (os.path.isdir(d) and os.path.exists(mpath) and os.path.exists(spath)):
        return _empty()
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(spath, "rb") as f:
            state = pickle.load(f)
    except Exception:
        return _empty()
    feat_cols = (manifest.get("config_snapshot") or {}).get("feat_cols") or []
    asof = manifest.get("train_end_date") or None
    df = contrib_from_state(state, feat_cols, asof=asof)
    if df.empty:
        return df
    if df["asof"].isna().all():
        import datetime as _dt

        df["asof"] = _dt.datetime.fromtimestamp(os.path.getmtime(spath)).strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# backtest 会话的腿权重时序（P5 面板「权重随时间」数据源）
# ---------------------------------------------------------------------------
def load_session_leg_weights(sessions_dir: str | None = None) -> pd.DataFrame:
    """最新一份 model_weights.parquet → 长表 (date, model, weight, ic)。

    扫描 outputs/sessions/*/artifacts/model_weights.parquet，取 mtime 最新者。
    宽表列约定（src/backtest_daily.py）：date, w_<model>, ic_<model>。
    缺失/损坏返回带列名的空表。
    """
    out_cols = ["date", "model", "weight", "ic"]
    sdir = sessions_dir or os.path.join(ROOT, "outputs", "sessions")
    files = glob.glob(os.path.join(sdir, "*", "artifacts", "model_weights.parquet"))
    if not files:
        return pd.DataFrame(columns=out_cols)
    try:
        df = pd.read_parquet(max(files, key=os.path.getmtime))
    except Exception:
        return pd.DataFrame(columns=out_cols)
    wcols = [c for c in df.columns if c.startswith("w_")]
    if not wcols or "date" not in df.columns:
        return pd.DataFrame(columns=out_cols)
    rows = []
    for _, r in df.iterrows():
        for c in wcols:
            model = c[2:]
            rows.append(
                {
                    "date": r["date"],
                    "model": model,
                    "weight": float(r[c]) if pd.notna(r[c]) else np.nan,
                    "ic": (
                        float(r[f"ic_{model}"])
                        if f"ic_{model}" in df.columns and pd.notna(r.get(f"ic_{model}"))
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows, columns=out_cols)
