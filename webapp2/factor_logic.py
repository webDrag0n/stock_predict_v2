"""因子页的纯逻辑：类别推断、冗余簇、主表组装、筛选。不依赖 dash，便于单测。

自旧前端 factor_workbench 原样迁入（口径不变，测试沿用同一批用例）。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 类别推断（原生因子无 YAML 元数据，按名称前缀归类——设计 5.4 第 ④ 项的轻量替代）
# ---------------------------------------------------------------------------
_EXACT_CATEGORY = {
    "ep": "价值",
    "bp": "价值",
    "sp": "价值",
    "ln_mv": "价值",
    "macd": "技术指标",
    "amihud_20": "量价",
}

# 顺序敏感：先命中先生效
_PREFIX_CATEGORY = [
    ("a158_", "Alpha158"),
    ("gtja_", "GTJA191"),
    ("mx_north", "北向交叉"),
    ("north_", "北向资金"),
    ("mx_", "宏观交叉"),
    ("opt_", "期权IV"),
    ("spx_", "海外宏观"),
    ("ndx_", "海外宏观"),
    ("dji_", "海外宏观"),
    ("us_", "海外宏观"),
    ("cn_", "中国宏观"),
    ("ep_", "价值"),
    ("bp_", "价值"),
    ("sp_", "价值"),
    ("ln_mv_", "价值"),
    ("value_", "价值"),
    ("industry_", "行业动量"),
    ("momentum_", "动量"),
    ("resid_mom", "动量"),
    ("high_52w", "动量"),
    ("return_", "反转"),
    ("volume_", "量价"),
    ("hl_", "量价"),
    ("price_to_ma", "量价"),
    ("volatility_", "波动率"),
    ("idio_vol", "波动率"),
    ("max_ret", "波动率"),
    ("ret_skew", "波动率"),
    ("rsi", "技术指标"),
    ("macd", "技术指标"),
    ("bb_", "技术指标"),
    ("stoch_", "技术指标"),
    ("limit_", "交易行为"),
    ("near_limit", "交易行为"),
]


# YAML category 英文值 → 中文（与原生前缀推断的词表对齐，筛选条不再双语并存）
_CATEGORY_ZH = {
    "momentum": "动量",
    "reversal": "反转",
    "volume": "量价",
    "value": "价值",
    "volatility": "波动率",
    "liquidity": "流动性",
    "gtja191": "GTJA191",
    "alpha158": "Alpha158",
    "north": "北向资金",
    "macro": "宏观",
    "custom": "自定义",
}


def _norm_category(cat) -> str:
    c = str(cat or "general").strip()
    return _CATEGORY_ZH.get(c.lower(), c)


def infer_category(name: str) -> str:
    """按名称前缀推断原生因子类别；无法识别 → "其他原生"。"""
    n = str(name)
    if n in _EXACT_CATEGORY:
        return _EXACT_CATEGORY[n]
    for prefix, cat in _PREFIX_CATEGORY:
        if n.startswith(prefix):
            return cat
    return "其他原生"


# ---------------------------------------------------------------------------
# 主表数据组装（纯函数，供单测）
# ---------------------------------------------------------------------------
OVERVIEW_COLUMNS = [
    "因子",
    "名称",
    "类别",
    "来源",
    "启用",
    "RankIC",
    "ICIR",
    "IC_t",
    "准入",
    "多空收益",
    "单调性",
    "换手",
    "近期衰减",
    "冗余簇",
    "贡献分位",
]

_EVAL_METRIC_COLS = ["RankIC", "ICIR", "IC_t", "多空收益", "单调性", "换手"]

# 主表列组：OVERVIEW_COLUMNS 是全量真源（测试锚定，不动）；核心组是显示层子集
CORE_COLUMNS = [
    "因子",
    "类别",
    "准入",
    "RankIC",
    "ICIR",
    "多空收益",
    "换手",
    "贡献分位",
    "近期衰减",
]

# 主表列头悬停解释（tooltip_header），术语口径与 components.GLOSSARY 一致
_COLUMN_TOOLTIPS = {
    "因子": "因子 id（列名）；点击行进入单因子详情",
    "名称": "因子显示名；原生因子与 id 相同",
    "类别": "因子类别；原生因子按名称前缀推断",
    "来源": "公式库 = factor_library/*.yaml 定义；原生 = 面板内置列",
    "启用": "公式库因子是否 enabled（草稿一律停用）；原生因子无此概念",
    "RankIC": "预测排序与真实收益排序的截面相关系数（Spearman）。越接近 1 选股排序越准，0 附近就是瞎猜",
    "ICIR": "RankIC 的均值除以标准差，相当于 IC 的「夏普」；衡量信号稳不稳",
    "IC_t": "单日 IC 的 t 统计量；绝对值越大越显著",
    "准入": "是否通过评估准入门槛；✓ = 通过",
    "多空收益": "做多最看好一组、做空最不看好一组的收益差；衡量排序能力，不代表实盘可执行",
    "单调性": "分层收益随分位的单调程度，越接近 1 越好",
    "换手": "调仓换手率；越高交易成本越大",
    "近期衰减": "近期窗口 RankIC 迷你条形（▁→█ 低到高）；看信号是否正在失效",
    "冗余簇": "|corr|≥0.8 的高相关簇标签；同簇因子信息重复",
    "贡献分位": "该因子在 champion 模型中的综合贡献分位（%）",
}

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values) -> str:
    """一组数值 → unicode 迷你条形（如 ▂▅▇▃）；空/全 NaN → ""。"""
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return _SPARK_BLOCKS[3] * len(vals)
    last = len(_SPARK_BLOCKS) - 1
    return "".join(_SPARK_BLOCKS[min(int((v - lo) / (hi - lo) * last), last)] for v in vals)


def corr_clusters(corr_df: pd.DataFrame, threshold: float = 0.8) -> dict:
    """相关矩阵 → {因子: 簇标签}。|corr|≥threshold 的因子并查集聚簇，单因子不成簇。

    簇标签形如「簇#1·turnover_rate（3因子）」，代表元取簇内名字典序最小者。
    """
    if corr_df is None or corr_df.empty:
        return {}
    m = corr_df.copy()
    first = m.columns[0]
    if str(first).startswith("Unnamed") or first == "":
        m = m.set_index(first)
    labels = [str(c) for c in m.columns]
    try:
        mat = m.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return {}
    parent = list(range(len(labels)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            v = mat[i][j]
            if np.isfinite(v) and abs(v) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(labels[i])
    clusters = sorted((m for m in groups.values() if len(m) >= 2), key=min)
    out = {}
    for i, members in enumerate(clusters, 1):
        rep = min(members)
        label = f"簇#{i}·{rep}（{len(members)}因子）"
        for f in members:
            out[f] = label
    return out


def _decay_sparklines(decay: pd.DataFrame) -> dict:
    """diag_factor_ic_decay.csv → {factor: sparkline 字符串}（w0..wN 按数字序）。"""
    if decay is None or decay.empty or "factor" not in decay.columns:
        return {}
    wcols = sorted(
        [c for c in decay.columns if re.fullmatch(r"w\d+", str(c))],
        key=lambda c: int(str(c)[1:]),
    )
    if not wcols:
        return {}
    dk = decay.drop_duplicates("factor").set_index("factor")
    out = {}
    for f in dk.index:
        try:
            out[str(f)] = sparkline(list(dk.loc[f, wcols]))
        except (KeyError, TypeError):
            out[str(f)] = ""
    return out


def build_overview(
    lib: pd.DataFrame,
    eval_df: pd.DataFrame,
    decay: pd.DataFrame,
    corr: pd.DataFrame,
    contrib_scores: pd.Series,
    panel_cols,
) -> pd.DataFrame:
    """因子主表：YAML 库因子 + panel 原生因子并集，左连接评估/衰减/相关/贡献。

    任何一路数据缺失（None/空表/缺列）都容错：对应列留空，不抛异常。
    contrib_scores 为 {factor: 综合贡献分} 的 Series（调用方用
    src/factor_contrib.combined_scores 预先算好），本函数只负责转分位。
    """
    rows: dict[str, dict] = {}
    order: list[str] = []
    if lib is not None and not lib.empty and "id" in lib.columns:
        for _, r in lib.iterrows():
            fid = str(r["id"])
            if fid in rows:
                continue
            rows[fid] = {
                "因子": fid,
                "名称": str(r.get("name") or fid),
                "类别": _norm_category(r.get("category")),
                "来源": "公式库",
                "启用": bool(r.get("enabled", True)),
            }
            order.append(fid)
    for c in panel_cols or ():
        c = str(c)
        if c in rows:
            continue
        rows[c] = {"因子": c, "名称": c, "类别": infer_category(c), "来源": "原生", "启用": None}
        order.append(c)
    if not order:
        return pd.DataFrame(columns=OVERVIEW_COLUMNS)
    df = pd.DataFrame([rows[k] for k in order])

    # --- 评估指标（outputs/factor_eval.csv，列名以 evaluate_panel 产物为准）---
    for col in _EVAL_METRIC_COLS:
        df[col] = np.nan
    df["准入"] = ""
    if eval_df is not None and not eval_df.empty and "factor" in eval_df.columns:
        ev = eval_df.drop_duplicates("factor").set_index("factor")
        # eval CSV 的 RankICIR → 主表展示列 ICIR，其余同名映射
        for src_col, dst_col in [
            ("RankIC", "RankIC"),
            ("RankICIR", "ICIR"),
            ("IC_t", "IC_t"),
            ("多空收益", "多空收益"),
            ("单调性", "单调性"),
            ("换手", "换手"),
        ]:
            if src_col in ev.columns:
                df[dst_col] = df["因子"].map(ev[src_col])
        if "准入" in ev.columns:
            df["准入"] = df["因子"].map(ev["准入"]).fillna("").astype(str)

    # --- 近期衰减 sparkline（outputs/diag_factor_ic_decay.csv）---
    sparks = _decay_sparklines(decay)
    df["近期衰减"] = df["因子"].map(lambda f: sparks.get(f, ""))

    # --- 冗余簇（outputs/diag_factor_corr.csv，|corr|≥0.8）---
    clusters = corr_clusters(corr)
    df["冗余簇"] = df["因子"].map(lambda f: clusters.get(f, ""))

    # --- 模型内贡献分位（champion ensemble state 提取）---
    df["贡献分位"] = np.nan
    if contrib_scores is not None and len(contrib_scores):
        q = contrib_scores.rank(pct=True) * 100.0
        df["贡献分位"] = df["因子"].map(q)
    return df[OVERVIEW_COLUMNS]


def apply_filters(
    df: pd.DataFrame,
    category: str | None = None,
    gate: str | None = None,
    enabled: str | None = None,
    text: str | None = None,
) -> pd.DataFrame:
    """筛选条回调的纯过滤逻辑。gate ∈ 全部/已通过/未通过/未评估；
    enabled ∈ 全部/已启用/已停用/原生因子。"""
    if df.empty:
        return df
    out = df
    if category and category != "全部":
        out = out[out["类别"] == category]
    if gate and gate != "全部":
        evaluated = out["RankIC"].notna()
        if gate == "已通过":
            out = out[out["准入"] == "✓"]
        elif gate == "未通过":
            out = out[evaluated & (out["准入"] != "✓")]
        elif gate == "未评估":
            out = out[~evaluated]
    if enabled and enabled != "全部":
        if enabled == "原生因子":
            out = out[out["来源"] == "原生"]
        elif enabled == "已启用":
            out = out[out["启用"] == True]  # noqa: E712 —— 三态布尔列需显式比较
        elif enabled == "已停用":
            out = out[out["启用"] == False]  # noqa: E712
    if text:
        t = str(text).strip().lower()
        if t:
            mask = out["因子"].str.lower().str.contains(t, regex=False) | out[
                "名称"
            ].str.lower().str.contains(t, regex=False)
            out = out[mask]
    return out
