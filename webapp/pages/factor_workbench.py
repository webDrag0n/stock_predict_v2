"""因子工作台 Factor Workbench（/factors · /factors/<id> · /factors/new）。

设计文档 docs/frontend_redesign.html 第 5 节（页面三 · 因子工作台）：

· /factors       因子库总览：A 筛选条 + B 评估入口 + C 因子主表（YAML 库因子与
                 panel 原生因子并集 × factor_eval 指标 × IC 衰减 sparkline ×
                 冗余簇 × 模型内贡献分位）+ D 冗余簇热图 + E 模型 vs 最强单因子
· /factors/<id>  单因子详情：头部（定义/公式/版本）+ 六面板
                 P1 逐窗口 RankIC · P2 分层净值 · P3 IC 分布 ·
                 P4 相关 Top10 · P5 模型内贡献 · P6 覆盖面/缺失率
· /factors/new   公式因子编辑器：静态校验（不 import src，本地复刻
                 src/data/factors/parser.py 的 AST 白名单）+ 保存草稿到
                 factor_library/custom.yaml（enabled: false）+ 保存成功后
                 一键创建 factoreval 会话（单因子子集，730 天回看）并跳 /runs/<sid>

子路由说明：app.py 路由表精确匹配 /factors；/factors/<id> 与 /factors/new
由 app.py::_route 的前缀分派调用本模块 route_content()（纯函数）渲染。
page-content 全应用只有 app.py 一个路由回调，禁止再加 allow_duplicate
重复输出回调（客户端去重与服务端分派规则不一致，曾导致随机路由错误）。

数据契约（全部空态容错，loader 见 webapp/dataio.py）：
  factor_library / factor_eval_table / factor_ic_decay / factor_corr /
  panel_factor_columns / output_csv("diag_model_vs_factor.csv")
  outputs/factor_layer_<id>.parquet、outputs/factor_ic_daily_<id>.parquet
  （由 src/factor_layers 在 eval 流程落盘）
  src/factor_contrib.load_champion_contrib / load_session_leg_weights
  （champion 缺失时为空表）

未实现（设计 P3 延后项，页面注明）：留一法 walkforward 对比、
原生因子完整注册表（仅前缀推断类别）。
"""

from __future__ import annotations

import ast
import functools
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc

from .. import dataio
from ..components import model_label, page_header
from ..theme import DIVERGING, PALETTE, base_fig, empty_fig

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


# ---------------------------------------------------------------------------
# 公式因子编辑器：静态校验 + 草稿落盘（纯函数，供单测）
# ---------------------------------------------------------------------------
# 复刻 src/data/factors/parser.py 的 DSL 原语 / 安全内置 / AST 白名单
# （约定：编辑器不 import src，本地解析校验）。
DSL_PRIMITIVES = frozenset(
    {
        "ts_delay",
        "ts_delta",
        "ts_sum",
        "ts_mean",
        "ts_std",
        "ts_max",
        "ts_min",
        "ts_skew",
        "ts_argmax",
        "ts_argmin",
        "ts_corr",
        "ts_rank",
        "cs_rank",
        "cs_zscore",
        "signed_power",
        "safe_div",
        "ts_quantile",
        "ts_reg_resid",
        "ts_reg_r2",
    }
)
SAFE_BUILTINS = frozenset(
    {"log", "abs", "sqrt", "sign", "exp", "log2", "log10", "round", "min", "max"}
)
_FUNCTION_NAMES = DSL_PRIMITIVES | SAFE_BUILTINS

_SAFE_NODES: set[type] = {
    ast.Module,
    ast.Expr,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.IfExp,
    ast.Call,
    ast.Tuple,
    ast.List,
    ast.Slice,
}
_FORBIDDEN_NODES = {
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Return,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Dict,
    ast.Set,
}

# 公式可引用的原始行情列（面板构建前的基础列；panel 因子列与库内因子 id 另算）
BASE_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_chg",
        "vwap",
        "adj_close",
        "turnover_rate",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "total_mv",
        "circ_mv",
    }
)
# 面板元列 / 目标列，禁止作为因子 id
RESERVED_IDS = frozenset(
    {"date", "symbol", "weight", "date_id", "time_id", "symbol_id", "return_1d"}
)

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def check_formula(formula: str) -> tuple[list, set]:
    """公式 DSL 静态校验：返回 (errors, 依赖列集合)。不 import src。"""
    errors: list[str] = []
    deps: set[str] = set()
    if not formula or not str(formula).strip():
        return ["公式不能为空"], deps
    try:
        tree = ast.parse(str(formula), mode="exec")
    except SyntaxError as e:
        return [f"语法错误: {e}"], deps
    for node in ast.walk(tree):
        t = type(node)
        if t in _FORBIDDEN_NODES:
            errors.append(f"禁止的语法: {t.__name__}（line {getattr(node, 'lineno', '?')}）")
        elif t not in _SAFE_NODES:
            errors.append(f"不支持的语法节点: {t.__name__}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                errors.append("只能调用注册的 DSL 原语函数（不允许属性/下标调用）")
            elif node.func.id not in _FUNCTION_NAMES:
                errors.append(f"未知函数: {node.func.id}（仅支持 DSL 原语与安全数学函数）")
        elif isinstance(node, ast.Name) and node.id not in _FUNCTION_NAMES:
            deps.add(node.id)
    return errors, deps


def normalize_expected_sign(v):
    """'+1'/'-1'/1/-1/'' → 1 / -1 / None；其他 → ValueError。"""
    if v is None or v == "" or str(v).strip() in ("", "unknown", "未知", "None"):
        return None
    s = str(v).strip()
    if s in ("+1", "1", "1.0"):
        return 1
    if s in ("-1", "-1.0"):
        return -1
    raise ValueError(f"expected_sign 只能是 +1 / -1 / 未知，收到 {v!r}")


def validate_factor_draft(
    fid,
    name,
    category,
    formula,
    tags="",
    expected_sign=None,
    description="",
    existing_ids=frozenset(),
    known_columns=frozenset(),
) -> dict:
    """编辑器「校验」按钮的纯逻辑：返回 {errors, warnings, deps, record}。

    字段规则与 src/data/factors/schema.py::FactorDefinition 对齐：
    id/formula 必填且为字符串，id 唯一且合法命名；公式过 AST 白名单。
    """
    errors: list[str] = []
    warnings: list[str] = []
    fid = (fid or "").strip()
    name = (name or "").strip()
    category = (category or "").strip() or "custom"
    formula = (formula or "").strip()
    description = (description or "").strip()

    if not fid:
        errors.append("id 不能为空")
    elif not _ID_RE.match(fid):
        errors.append("id 须为合法列名：字母/下划线开头，仅含字母数字下划线，≤64 字符")
    elif fid in RESERVED_IDS or fid.startswith("responder_"):
        errors.append(f"id {fid!r} 与面板保留列冲突")
    elif fid in set(existing_ids):
        errors.append(f"id {fid!r} 已存在（因子库或面板原生因子）")
    if not name:
        errors.append("name 不能为空")

    ferrs, deps = check_formula(formula)
    errors.extend(ferrs)
    for d in sorted(deps):
        if d not in set(known_columns):
            warnings.append(
                f"依赖列 {d!r} 不在已知列中（若为其他公式因子，请在保存后补 explicit_deps）"
            )
    try:
        sign = normalize_expected_sign(expected_sign)
    except ValueError as e:
        errors.append(str(e))
        sign = None

    tag_list = [t.strip() for t in re.split(r"[,，\s]+", tags or "") if t.strip()]
    record = {
        "id": fid,
        "name": name or fid,
        "formula": formula,
        "description": description,
        "category": category,
        "source": "custom",
        "tags": tag_list,
        "enabled": False,  # 草稿一律停用，评估通过后才允许入库
        "expected_sign": sign,
        "version": 1,
        "author": "webapp",
        "created": "",
        "explicit_deps": [],
    }
    return {"errors": errors, "warnings": warnings, "deps": sorted(deps), "record": record}


def save_draft(record: dict, yaml_path: str) -> None:
    """把草稿追加写入 custom.yaml（保持 factors: [...] 结构，原子写）。

    id 重复抛 ValueError；文件不存在则新建。注意：重写会丢失原文件注释。
    """
    data: dict = {"factors": []}
    if os.path.exists(yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is not None:
            if not isinstance(loaded, dict) or not isinstance(loaded.get("factors", []), list):
                raise ValueError(f"{yaml_path} 结构异常：顶层应为含 factors 列表的字典")
            data = loaded
        data.setdefault("factors", [])
        if data["factors"] is None:
            data["factors"] = []
    if any(isinstance(r, dict) and r.get("id") == record["id"] for r in data["factors"]):
        raise ValueError(f"草稿 id 重复：{record['id']} 已在 {os.path.basename(yaml_path)} 中")
    data["factors"].append(record)
    tmp = f"{yaml_path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp, yaml_path)


# ---------------------------------------------------------------------------
# 保存后「立即评估」：创建 factoreval 会话（参数组装为纯函数，供单测）
# ---------------------------------------------------------------------------
# 单因子评估回看天数：比全库默认 1825 天更快（单因子分钟级出结果）
SINGLE_FACTOR_LOOKBACK_DAYS = 730


def build_factoreval_params(factor_id: str) -> dict:
    """「立即评估该因子」的会话参数：factoreval 默认参数上覆盖单因子子集与回看天数。

    pipeline 延迟到调用时 import：编辑器静态校验约定不 import src，
    registry 会连带拉起 steps_factoreval。
    """
    from pipeline import registry

    params = registry.default_params("factoreval")
    params["factor_subset"] = str(factor_id)
    params["lookback_days"] = SINGLE_FACTOR_LOOKBACK_DAYS
    return params


def launch_factoreval_session(factor_id: str) -> str:
    """创建单因子 factoreval 会话并推进调度（queued → running）。返回 session_id。"""
    from pipeline import scheduler

    sid = scheduler.create(
        "factoreval",
        f"单因子评估 {factor_id}",
        build_factoreval_params(factor_id),
    )
    scheduler.tick()
    return sid


def launch_full_eval_session() -> str:
    """「运行全库评估」：以 factoreval 默认参数（全库启用因子，覆盖写 factor_eval.csv）
    创建会话并推进调度。返回 session_id。"""
    from pipeline import registry, scheduler

    sid = scheduler.create("factoreval", "全库因子评估", registry.default_params("factoreval"))
    scheduler.tick()
    return sid


def _save_success_block(record: dict) -> dbc.Alert:
    """保存成功提示：一键评估按钮（创建 factoreval 会话并跳 /runs/<sid>）
    + CLI 逃生通道（折叠小字，不删）+ 因子详情链接。"""
    fid = record["id"]
    return dbc.Alert(
        [
            html.Div(
                f"✓ 草稿已保存（enabled: false）→ factor_library/custom.yaml · id={fid}",
                style={"fontWeight": 700},
            ),
            dcc.Store(id="fe-eval-fid", data=fid),
            html.Div(
                dbc.Button(
                    f"立即评估该因子（回看 {SINGLE_FACTOR_LOOKBACK_DAYS} 天，分钟级）",
                    id="fe-eval-btn",
                    color="success",
                    size="sm",
                ),
                style={"marginTop": "8px"},
            ),
            html.Div(
                f"点击后创建「单因子评估 {fid}」会话并跳转 /runs 查看进度；回看取 "
                f"{SINGLE_FACTOR_LOOKBACK_DAYS} 天（单因子，比全库默认 1825 天更快）。",
                style={"fontSize": "12px", "color": PALETTE["muted"], "marginTop": "4px"},
            ),
            html.Div(id="fe-eval-msg"),
            html.Details(
                [
                    html.Summary(
                        "CLI 逃生通道（等价命令）",
                        style={"fontSize": "12px", "color": PALETTE["muted"]},
                    ),
                    html.Pre(
                        f"USE_FORMULA_FACTORS=1 FORMULA_FACTOR_SUBSET={fid} \\\n"
                        "  uv run python scripts/eval_factors.py",
                        style={"fontSize": "12px", "marginTop": "6px"},
                    ),
                ],
                style={"marginTop": "6px"},
            ),
            dcc.Link(f"查看因子详情 → /factors/{fid}", href=f"/factors/{fid}"),
        ],
        color="success",
    )


# ---------------------------------------------------------------------------
# 页面级数据 loader（lru_cache 与 dataio 同风格，全部空态容错）
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _factor_contrib() -> pd.DataFrame:
    """champion 因子贡献长表（src/factor_contrib 契约）；无 champion → 空表。"""
    try:
        import factor_contrib

        return factor_contrib.load_champion_contrib()
    except Exception:
        return pd.DataFrame(columns=["factor", "source", "kind", "value", "asof"])


@functools.lru_cache(maxsize=1)
def _contrib_scores() -> pd.Series:
    """因子 → 综合贡献分（腿内归一 × 腿权跨腿求和）。"""
    contrib = _factor_contrib()
    if contrib.empty:
        return pd.Series(dtype=float)
    try:
        import factor_contrib

        return factor_contrib.combined_scores(contrib, factor_contrib.champion_leg_weights())
    except Exception:
        return pd.Series(dtype=float)


@functools.lru_cache(maxsize=1)
def _session_leg_weights() -> pd.DataFrame:
    try:
        import factor_contrib

        return factor_contrib.load_session_leg_weights()
    except Exception:
        return pd.DataFrame(columns=["date", "model", "weight", "ic"])


@functools.lru_cache(maxsize=256)
def _layer_equity(factor: str) -> pd.DataFrame:
    try:
        import factor_layers

        path = factor_layers.layer_equity_path(dataio.OUTPUTS, factor)
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        pass
    return pd.DataFrame(columns=["date", "layer", "equity"])


@functools.lru_cache(maxsize=256)
def _daily_ic(factor: str) -> pd.DataFrame:
    try:
        import factor_layers

        path = factor_layers.daily_ic_path(dataio.OUTPUTS, factor)
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        pass
    return pd.DataFrame(columns=["date", "rank_ic"])


@functools.lru_cache(maxsize=256)
def _factor_coverage(factor: str) -> pd.DataFrame:
    """按日的因子非 NaN 占比。只读 panel 的 (date, factor) 两列控制内存。"""
    path = os.path.join(dataio.CACHE, "wholemkt_panel.parquet")
    cols = ["date", "coverage"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(path, columns=["date", factor])
    except Exception:
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    cov = df.groupby("date")[factor].apply(lambda s: float(s.notna().mean()))
    return cov.rename("coverage").reset_index()[cols]


# ---------------------------------------------------------------------------
# 图形构造
# ---------------------------------------------------------------------------
def _corr_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """corr CSV → 方阵（第一列 unnamed 时设为 index）。"""
    m = corr.copy()
    first = m.columns[0]
    if str(first).startswith("Unnamed") or first == "":
        m = m.set_index(first)
    return m


def _heatmap_fig(corr: pd.DataFrame) -> go.Figure:
    if corr is None or corr.empty:
        return empty_fig(
            "无相关矩阵（outputs/diag_factor_corr.csv）\n点上方「重新评估」生成",
            height=420,
        )
    m = _corr_matrix(corr)
    labels = [str(c) for c in m.columns]
    z = m.to_numpy(dtype=float)
    fig = base_fig("因子相关矩阵 · 冗余簇热图（|corr|≥0.8 在主表中标注同簇）", height=640)
    fig.add_trace(
        go.Heatmap(z=z, x=labels, y=labels, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1)
    )
    fig.update_xaxes(tickfont=dict(size=7), tickangle=90)
    fig.update_yaxes(tickfont=dict(size=7), autorange="reversed")
    fig.update_layout(margin=dict(l=110, b=110))
    return fig


def _model_vs_factor_block() -> html.Div:
    """E. 模型 vs 最强单因子（outputs/diag_model_vs_factor.csv，可能缺失）。"""
    mvf = dataio.output_csv("diag_model_vs_factor.csv")
    if mvf.empty:
        return html.Div(
            [
                html.Div("模型 vs 最强单因子", className="control-label"),
                dbc.Alert(
                    [
                        "产物缺失：outputs/diag_model_vs_factor.csv。",
                        html.Br(),
                        "点上方「重新评估」生成「模型是否有增量」对照。",
                    ],
                    color="secondary",
                    className="mb-0",
                ),
            ],
            className="chart-card",
        )
    fig = base_fig("模型 vs 最强单因子", height=360)
    label_col = mvf.columns[0]
    num_cols = [c for c in mvf.columns[1:] if pd.api.types.is_numeric_dtype(mvf[c])]
    for c in num_cols:
        fig.add_trace(go.Bar(x=mvf[label_col].astype(str), y=mvf[c], name=str(c)))
    fig.update_layout(barmode="group")
    return html.Div(dcc.Graph(figure=fig), className="chart-card")


def _decay_fig(fid: str) -> go.Figure:
    """P1. 逐窗口 RankIC 柱状 + 累计均值滚动线。"""
    decay = dataio.factor_ic_decay()
    if decay.empty or "factor" not in decay.columns:
        return empty_fig(
            "无 IC 衰减数据（outputs/diag_factor_ic_decay.csv）\n点上方「重新评估」生成",
            height=300,
        )
    wcols = sorted(
        [c for c in decay.columns if re.fullmatch(r"w\d+", str(c))],
        key=lambda c: int(str(c)[1:]),
    )
    row = decay[decay["factor"] == fid]
    if row.empty or not wcols:
        return empty_fig(f"{fid} 不在 IC 衰减诊断中（可能未纳入诊断因子集）", height=300)
    vals = row[wcols].to_numpy().ravel().astype(float)
    x = list(range(len(vals)))
    colors = [PALETTE["strat"] if v >= 0 else PALETTE["loss"] for v in vals]
    fig = base_fig(f"P1 · 逐窗口 RankIC（{fid}）—— 它还活着吗？", height=300)
    fig.add_trace(go.Bar(x=x, y=vals, marker_color=colors, name="窗口 RankIC"))
    if len(vals) >= 2:
        cum = np.cumsum(vals) / np.arange(1, len(vals) + 1)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cum,
                mode="lines+markers",
                name="累计均值",
                line=dict(color=PALETTE["alpha"], width=1.6),
            )
        )
    if "IC_mean" in row.columns:
        mean_ic = float(row["IC_mean"].iloc[0])
        fig.add_hline(
            y=mean_ic,
            line=dict(color=PALETTE["muted"], width=1, dash="dash"),
            annotation_text=f"均值 {mean_ic:+.3f}",
        )
    fig.add_hline(y=0, line=dict(color="grey", width=0.7))
    fig.update_xaxes(title="窗口 #", tickvals=x)
    fig.update_yaxes(title="RankIC")
    return fig


def _layer_fig(fid: str) -> go.Figure:
    """P2. 5 层分层净值（outputs/factor_layer_<id>.parquet）。"""
    eq = _layer_equity(fid)
    if eq.empty:
        return empty_fig(
            "无分层净值数据\n点上方「重新评估」生成 factor_layer_<id>.parquet",
            height=300,
        )
    fig = base_fig(f"P2 · 分层净值（{fid}）—— 单调吗？头部还是尾部赚钱？", height=300)
    layers = sorted(eq["layer"].unique())
    n = len(layers)
    for i, lyr in enumerate(layers):
        sub = eq[eq["layer"] == lyr]
        if i == n - 1:
            color, width = PALETTE["strat"], 2.2
        elif i == 0:
            color, width = PALETTE["loss"], 2.2
        else:
            color, width = PALETTE["faint"], 1.0
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["equity"],
                mode="lines",
                name=str(lyr),
                line=dict(color=color, width=width),
            )
        )
    fig.update_yaxes(title="累计净值")
    return fig


def _ic_dist_fig(fid: str) -> go.Figure:
    """P3. 逐日 RankIC 分布直方图 + 正占比。"""
    ic = _daily_ic(fid)
    if ic.empty:
        return empty_fig(
            "无逐日 IC 数据\n点上方「重新评估」生成 factor_ic_daily_<id>.parquet",
            height=300,
        )
    vals = ic["rank_ic"].astype(float)
    pos = float((vals > 0).mean())
    fig = base_fig(
        f"P3 · IC 分布（{fid}）—— 正 IC 占比 {pos:.0%}，稳定还是靠少数几天？", height=300
    )
    fig.add_trace(go.Histogram(x=vals, nbinsx=40, marker_color=PALETTE["commission"], opacity=0.75))
    fig.add_vline(x=0, line=dict(color=PALETTE["loss"], width=1.2))
    fig.add_vline(
        x=float(vals.mean()),
        line=dict(color=PALETTE["strat"], width=1.2, dash="dash"),
        annotation_text=f"均值 {vals.mean():+.3f}",
    )
    fig.update_xaxes(title="逐日 RankIC")
    fig.update_yaxes(title="天数")
    return fig


def _corr_top_fig(fid: str) -> go.Figure:
    """P4. 相关邻域 Top10 横向条形。"""
    corr = dataio.factor_corr()
    if corr.empty:
        return empty_fig(
            "无相关矩阵（outputs/diag_factor_corr.csv）\n点上方「重新评估」生成",
            height=300,
        )
    m = _corr_matrix(corr)
    if fid not in m.index and fid not in m.columns:
        return empty_fig(f"{fid} 不在相关矩阵中", height=300)
    s = m.loc[fid] if fid in m.index else m[fid]
    s = pd.to_numeric(s, errors="coerce").drop(labels=[fid], errors="ignore").dropna()
    if s.empty:
        return empty_fig("无相关数据", height=300)
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    colors = [PALETTE["strat"] if v >= 0 else PALETTE["loss"] for v in top.values]
    fig = base_fig(f"P4 · 相关邻域 Top10（{fid}）—— 独立信息还是重复？", height=300)
    fig.add_trace(
        go.Bar(
            y=[str(i) for i in top.index],
            x=top.values,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in top.values],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line=dict(color="grey", width=0.8))
    fig.update_layout(margin=dict(l=150))
    return fig


def _contrib_block(fid: str) -> html.Div:
    """P5. 模型内贡献面板：champion 内该因子的权重/重要性 + 各腿权重时序。"""
    contrib = _factor_contrib()
    parts: list = [html.Div("P5 · 模型内贡献 —— 模型真的在用它吗？", className="control-label")]
    mine = contrib[contrib["factor"] == fid] if not contrib.empty else pd.DataFrame()
    if mine.empty:
        parts.append(
            dbc.Alert(
                [
                    "champion 中无该因子的贡献记录（champion 不存在、未入库，"
                    "或该因子不在模型特征集）。契约产物：models/champion/ensemble_state.pkl。",
                ],
                color="secondary",
            )
        )
    else:
        scores = _contrib_scores()
        score = float(scores.get(fid, np.nan)) if len(scores) else np.nan
        q = float((scores.rank(pct=True) * 100).get(fid, np.nan)) if len(scores) else np.nan
        chips = []
        if np.isfinite(score):
            chips.append(f"综合贡献分 {score:.4f}")
        if np.isfinite(q):
            chips.append(f"全因子分位 {q:.0f}%")
        if chips:
            parts.append(
                html.Div(
                    " · ".join(chips),
                    style={"color": PALETTE["muted"], "fontSize": "12px", "margin": "4px 0"},
                )
            )
        tbl = mine.copy()
        tbl["value"] = tbl["value"].astype(float).round(6)
        if "source" in tbl.columns:
            tbl["source"] = tbl["source"].map(lambda s: model_label(s))
        parts.append(
            dash_table.DataTable(
                data=tbl.to_dict("records"),
                columns=[{"name": c, "id": c} for c in tbl.columns],
                style_cell={"textAlign": "left", "fontSize": 12, "padding": "4px"},
                style_header={"fontWeight": 700},
            )
        )
    # 各子模型滚动 IC 权重时序（最近 backtest 会话产物）
    lw = _session_leg_weights()
    if not lw.empty:
        fig = base_fig("各子模型滚动 IC 权重（最近一次 backtest 会话）", height=220)
        for model, sub in lw.groupby("model"):
            fig.add_trace(
                go.Scatter(x=sub["date"], y=sub["weight"], mode="lines", name=model_label(model))
            )
        fig.update_yaxes(title="子模型权重", tickformat=".0%")
        parts.append(dcc.Graph(figure=fig))
    parts.append(
        html.Div(
            "留一法（该因子在/不在特征集的两次 walkforward）对比为设计 P3 延后项，暂未实现。",
            style={"color": PALETTE["muted"], "fontSize": "12px", "marginTop": "6px"},
        )
    )
    return html.Div(parts, className="chart-card")


def _coverage_fig(fid: str) -> go.Figure:
    """P6. 覆盖面：按日非 NaN 占比。"""
    cov = _factor_coverage(fid)
    if cov.empty:
        return empty_fig("面板中无该因子列（或 panel 缺失），无法计算覆盖面", height=300)
    fig = base_fig(f"P6 · 覆盖面（{fid}）—— 覆盖全市场还是只有大盘？", height=300)
    fig.add_trace(
        go.Scatter(
            x=cov["date"],
            y=cov["coverage"],
            mode="lines",
            line=dict(color=PALETTE["commission"], width=1.2),
        )
    )
    fig.update_yaxes(title="非缺失占比", tickformat=".0%", range=[0, 1.05])
    return fig


# ---------------------------------------------------------------------------
# 页面布局
# ---------------------------------------------------------------------------
def _eval_hint_alert() -> dbc.Alert:
    return dbc.Alert(
        [
            "评估产物缺失：主表仅显示因子元数据，评估列标记「未评估」。",
            "点击下方「运行全库评估」一键创建评估会话，完成后回到本页即可看到结果。",
        ],
        color="warning",
    )


def _overview_layout() -> dbc.Container:
    lib = dataio.factor_library()
    eval_df = dataio.factor_eval_table()
    decay = dataio.factor_ic_decay()
    corr = dataio.factor_corr()
    df = build_overview(lib, eval_df, decay, corr, _contrib_scores(), dataio.panel_factor_columns())

    categories = (
        ["全部"] + sorted(df["类别"].dropna().unique().tolist()) if not df.empty else ["全部"]
    )

    head_cards = []
    if not df.empty:
        n_all = len(df)
        n_pass = int((df["准入"] == "✓").sum())
        n_eval = int(df["RankIC"].notna().sum())
        n_redun = int((df["冗余簇"] != "").sum())
        head_cards = [
            html.Div(
                [
                    html.Span(f"因子 {n_all}", className="status-chip"),
                    html.Span(f"已评估 {n_eval}", className="status-chip"),
                    html.Span(f"通过准入 {n_pass}", className="status-chip"),
                    html.Span(f"冗余簇内 {n_redun}", className="status-chip"),
                ],
                className="chart-card",
                style={"padding": "10px 14px"},
            )
        ]

    blocks = [
        page_header(
            "因子工作台 Factor Workbench",
            "哪个因子有效/失效/冗余？我新写的因子能打吗？—— 三层贡献分开看："
            "独立预测力（IC）· 边际贡献（冗余簇）· 组合内贡献（champion 权重）。",
            flow_step=0,
        ),
        dcc.Link("＋ 新建公式因子", href="/factors/new", className="page-sub"),
        dcc.Location(id="fw-redirect", refresh=True),
    ]
    if eval_df.empty:
        blocks.append(_eval_hint_alert())
    blocks.extend(head_cards)

    # A. 筛选条 + B. 评估入口
    blocks.append(
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div("类别", className="control-label"),
                                            dcc.Dropdown(
                                                id="fw-f-cat",
                                                options=categories,
                                                value="全部",
                                                clearable=False,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("准入状态", className="control-label"),
                                            dcc.Dropdown(
                                                id="fw-f-gate",
                                                options=["全部", "已通过", "未通过", "未评估"],
                                                value="全部",
                                                clearable=False,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("启用状态", className="control-label"),
                                            dcc.Dropdown(
                                                id="fw-f-enabled",
                                                options=["全部", "已启用", "已停用", "原生因子"],
                                                value="全部",
                                                clearable=False,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("搜索", className="control-label"),
                                            dbc.Input(
                                                id="fw-f-search",
                                                placeholder="因子 id / 名称包含…",
                                                debounce=True,
                                            ),
                                        ],
                                        md=3,
                                    ),
                                ]
                            )
                        ],
                        className="control-card",
                    ),
                    md=9,
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.Div("评估", className="control-label"),
                            dbc.Button(
                                "运行全库评估",
                                id="fw-eval-btn",
                                color="primary",
                                size="sm",
                            ),
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("运行全库因子评估")),
                                    dbc.ModalBody(
                                        [
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "全库启用因子 × 默认回看区间，"
                                                        "预计十分钟级起。"
                                                    ),
                                                    html.Li(
                                                        "将覆盖写 outputs/factor_eval.csv"
                                                        "（并落盘分层净值/逐日 IC）。"
                                                    ),
                                                    html.Li(
                                                        "确认后立即创建 factoreval 会话并跳转 "
                                                        "/runs/<sid>，可随时在那里查看进度。"
                                                    ),
                                                ],
                                                style={"fontSize": "13px"},
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        "CLI 等价命令",
                                                        style={
                                                            "fontSize": "12px",
                                                            "color": PALETTE["muted"],
                                                        },
                                                    ),
                                                    html.Pre(
                                                        "USE_FORMULA_FACTORS=1 uv run python "
                                                        "scripts/eval_factors.py",
                                                        style={
                                                            "fontSize": "12px",
                                                            "marginTop": "6px",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            html.Div(id="fw-eval-msg"),
                                        ]
                                    ),
                                    dbc.ModalFooter(
                                        [
                                            dbc.Button(
                                                "取消",
                                                id="fw-eval-cancel",
                                                color="secondary",
                                                outline=True,
                                                size="sm",
                                                className="me-2",
                                            ),
                                            dbc.Button(
                                                "确认运行",
                                                id="fw-eval-confirm",
                                                color="primary",
                                                size="sm",
                                            ),
                                        ]
                                    ),
                                ],
                                id="fw-eval-modal",
                                is_open=False,
                            ),
                        ],
                        className="control-card",
                    ),
                    md=3,
                ),
            ],
            className="g-2",
        )
    )

    # C. 因子主表
    if df.empty:
        blocks.append(
            dbc.Alert(
                "因子库为空：factor_library/*.yaml 无定义，且 panel 因子列不可用"
                "（data_cache/wholemkt_panel.parquet 缺失）。",
                color="secondary",
            )
        )
    else:
        blocks.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "因子主表（点击行进入单因子详情；ICIR 绿>0.3 / 红<0；"
                                "列头悬停看解释）",
                                className="control-label",
                            ),
                            dbc.RadioItems(
                                id="fw-col-group",
                                options=[
                                    {"label": "核心列", "value": "core"},
                                    {"label": "全部列", "value": "all"},
                                ],
                                value="core",
                                inline=True,
                                style={"fontSize": "12px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "baseline",
                            "gap": "12px",
                        },
                    ),
                    dash_table.DataTable(
                        id="fw-table",
                        data=_table_records(df),
                        columns=_table_columns("core"),
                        tooltip_header=_COLUMN_TOOLTIPS,
                        sort_action="native",
                        page_size=20,
                        style_cell={
                            "textAlign": "center",
                            "fontSize": 12.5,
                            "padding": "6px",
                            "fontFamily": "Inter",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "因子"}, "textAlign": "left", "fontWeight": 600},
                            {"if": {"column_id": "近期衰减"}, "fontFamily": "monospace"},
                            {"if": {"column_id": "冗余簇"}, "fontSize": 11},
                        ],
                        style_header={
                            "backgroundColor": PALETTE["strat"],
                            "color": "white",
                            "fontWeight": 700,
                        },
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{ICIR} > 0.3", "column_id": "ICIR"},
                                "color": PALETTE["strat"],
                                "fontWeight": 700,
                            },
                            {
                                "if": {"filter_query": "{ICIR} < 0", "column_id": "ICIR"},
                                "color": PALETTE["loss"],
                            },
                            {
                                "if": {"filter_query": "{准入} = '✓'", "column_id": "准入"},
                                "color": PALETTE["strat"],
                                "fontWeight": 700,
                            },
                        ],
                    ),
                ],
                className="chart-card",
            )
        )

    # D + E
    blocks.append(
        dbc.Row(
            [
                dbc.Col(
                    html.Div(dcc.Graph(figure=_heatmap_fig(corr)), className="chart-card"), md=8
                ),
                dbc.Col(_model_vs_factor_block(), md=4),
            ],
            className="g-2",
        )
    )
    return dbc.Container(blocks, fluid=True)


def _table_records(df: pd.DataFrame) -> list:
    """主表展示记录：数值列四舍五入，贡献分位取整。"""
    out = df.copy()
    for c in ["RankIC", "ICIR", "IC_t", "多空收益", "单调性", "换手"]:
        out[c] = out[c].astype(float).round(4)
    out["贡献分位"] = out["贡献分位"].astype(float).round(0)
    return out.to_dict("records")


def _table_columns(group: str = "core") -> list:
    """主表列定义。group='core' 只出 CORE_COLUMNS 子集；'all' 出 OVERVIEW_COLUMNS 全量。
    OVERVIEW_COLUMNS 本身不动（测试锚定），列组只是显示层过滤。"""
    names = {
        "RankIC": "RankIC",
        "ICIR": "ICIR",
        "IC_t": "IC_t",
        "多空收益": "多空收益",
        "单调性": "单调性",
        "换手": "换手",
        "近期衰减": "近期衰减",
        "冗余簇": "冗余簇",
        "贡献分位": "贡献分位%",
    }
    cols = CORE_COLUMNS if group == "core" else OVERVIEW_COLUMNS
    return [{"name": names.get(c, c), "id": c} for c in cols]


def _sign_badge(sign) -> dbc.Badge:
    if sign in (1, 1.0, "+1"):
        return dbc.Badge("预期正向 +1", color="success", className="me-1")
    if sign in (-1, -1.0, "-1"):
        return dbc.Badge("预期负向 -1", color="danger", className="me-1")
    return dbc.Badge("方向未知", color="secondary", className="me-1")


def detail_layout(fid: str) -> dbc.Container:
    """/factors/<id> 单因子详情：头部 + 六面板（各自空态容错）。"""
    lib = dataio.factor_library()
    meta = None
    if not lib.empty and "id" in lib.columns:
        hit = lib[lib["id"] == fid]
        if not hit.empty:
            meta = hit.iloc[0].to_dict()

    panel_cols = set(dataio.panel_factor_columns())
    if meta is None and fid not in panel_cols:
        return dbc.Container(
            [
                dcc.Link("← 返回因子库", href="/factors", className="page-sub"),
                dbc.Alert(
                    f"因子 {fid!r} 不存在于因子库（factor_library/*.yaml）或面板因子列中。",
                    color="warning",
                ),
            ],
            fluid=True,
        )

    if meta is not None:
        name = meta.get("name") or fid
        category = _norm_category(meta.get("category"))
        source = "公式库"
        sign = meta.get("expected_sign")
        version = meta.get("version", 1)
        author = meta.get("author") or "—"
        created = meta.get("created") or "—"
        formula = meta.get("formula") or ""
        enabled = bool(meta.get("enabled", True))
        desc = meta.get("description") or ""
    else:
        name, category, source = fid, infer_category(fid), "原生"
        sign, version, author, created, formula, enabled, desc = None, "—", "—", "—", "", None, ""

    badges = [
        dbc.Badge(category, color="info", className="me-1"),
        dbc.Badge(source, color="primary" if source == "公式库" else "secondary", className="me-1"),
        _sign_badge(sign),
    ]
    if enabled is not None:
        badges.append(
            dbc.Badge("已启用" if enabled else "已停用", color="success" if enabled else "dark")
        )

    header_children = [
        html.Div(
            [
                html.Span(str(name), style={"fontWeight": 700, "fontSize": "20px"}),
                html.Span(f"  {fid}", style={"color": PALETTE["muted"], "fontSize": "13px"}),
                html.Span("  ", style={"marginRight": "6px"}),
                *badges,
            ]
        ),
        html.Div(
            f"版本 v{version} · 作者 {author} · 创建 {created}",
            style={"color": PALETTE["muted"], "fontSize": "12px", "marginTop": "4px"},
        ),
    ]
    if desc:
        header_children.append(html.Div(desc, style={"fontSize": "13px", "marginTop": "6px"}))
    if formula:
        header_children.append(
            html.Pre(
                formula,
                style={
                    "background": "#0f172a",
                    "color": "#e2e8f0",
                    "padding": "12px",
                    "borderRadius": "8px",
                    "fontSize": "12.5px",
                    "marginTop": "8px",
                    "whiteSpace": "pre-wrap",
                },
            )
        )
    header_children.append(
        html.Div(
            [
                dbc.Button(
                    f"重新评估该因子（回看 {SINGLE_FACTOR_LOOKBACK_DAYS} 天）",
                    id="fwd-reeval-btn",
                    color="primary",
                    size="sm",
                ),
                dcc.Store(id="fwd-fid", data=fid),
                dcc.Location(id="fwd-redirect", refresh=True),
                html.Div(id="fwd-reeval-msg"),
                html.Div(
                    "点击后创建「单因子评估」会话并跳转 /runs 查看进度。",
                    style={"color": PALETTE["muted"], "fontSize": "12px", "marginTop": "4px"},
                ),
            ],
            style={"marginTop": "8px"},
        )
    )

    panels = [
        dbc.Col(html.Div(dcc.Graph(figure=_decay_fig(fid)), className="chart-card"), md=6),
        dbc.Col(html.Div(dcc.Graph(figure=_layer_fig(fid)), className="chart-card"), md=6),
        dbc.Col(html.Div(dcc.Graph(figure=_ic_dist_fig(fid)), className="chart-card"), md=6),
        dbc.Col(html.Div(dcc.Graph(figure=_corr_top_fig(fid)), className="chart-card"), md=6),
        dbc.Col(_contrib_block(fid), md=6),
        dbc.Col(html.Div(dcc.Graph(figure=_coverage_fig(fid)), className="chart-card"), md=6),
    ]
    rows = [dbc.Row(panels[i : i + 2], className="g-2") for i in range(0, 6, 2)]
    return dbc.Container(
        [
            dcc.Link("← 返回因子库", href="/factors", className="page-sub"),
            html.Div(header_children, className="chart-card"),
            *rows,
        ],
        fluid=True,
    )


_EDITOR_CATEGORIES = ["custom", "momentum", "reversal", "volume", "value", "general"]


def editor_layout() -> dbc.Container:
    """/factors/new 公式因子编辑器：静态校验 + 保存草稿 + CLI 评估指引。"""
    form = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("因子 id（列名）*", className="control-label"),
                            dbc.Input(id="fe-id", placeholder="如 my_reversal_5d"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("名称 *", className="control-label"),
                            dbc.Input(id="fe-name", placeholder="如 5日反转"),
                        ],
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("类别", className="control-label"),
                            dcc.Dropdown(
                                id="fe-category",
                                options=_EDITOR_CATEGORIES,
                                value="custom",
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("预期方向", className="control-label"),
                            dcc.Dropdown(
                                id="fe-sign",
                                options=[
                                    {"label": "未知", "value": ""},
                                    {"label": "+1（值大→预期涨）", "value": "+1"},
                                    {"label": "-1（值大→预期跌）", "value": "-1"},
                                ],
                                value="",
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("标签（逗号分隔）", className="control-label"),
                            dbc.Input(id="fe-tags", placeholder="如 reversal, short-term"),
                        ],
                        md=4,
                    ),
                ],
                className="mb-2",
            ),
            html.Div("公式（DSL 表达式）*", className="control-label"),
            dbc.Textarea(
                id="fe-formula",
                rows=5,
                placeholder="如 -ts_delta(log(close), 5)  或  safe_div(close - ts_mean(close, 20), ts_std(close, 20))",
                style={"fontFamily": "Menlo, Consolas, monospace", "fontSize": "12.5px"},
            ),
            html.Div(
                "可用原语: " + ", ".join(sorted(DSL_PRIMITIVES | SAFE_BUILTINS)),
                style={"color": PALETTE["muted"], "fontSize": "11px", "margin": "4px 0 10px"},
            ),
            html.Div("描述", className="control-label"),
            dbc.Textarea(id="fe-desc", rows=2, placeholder="因子逻辑、来源、适用环境"),
            html.Div(
                [
                    dbc.Button("校验", id="fe-btn-validate", color="secondary", className="me-2"),
                    dbc.Button("保存草稿到 custom.yaml", id="fe-btn-save", color="primary"),
                ],
                style={"marginTop": "12px"},
            ),
            html.Div(id="fe-validate-out", style={"marginTop": "12px"}),
            html.Div(id="fe-save-out", style={"marginTop": "12px"}),
        ],
        className="control-card",
    )
    right = html.Div(
        [
            html.Div("说明", className="control-label"),
            html.Ul(
                [
                    html.Li(
                        "校验在本地静态完成（AST 白名单 + 字段规则），不 import src、不执行公式。"
                    ),
                    html.Li("草稿以 enabled: false 追加到 factor_library/custom.yaml。"),
                    html.Li(
                        "保存成功后可一键创建「单因子评估」会话（factoreval 流水线，730 天回看）。"
                    ),
                    html.Li("评估通过后把 enabled 改为 true 即入库。"),
                ],
                style={"fontSize": "13px"},
            ),
        ],
        className="chart-card",
    )
    return dbc.Container(
        [
            dcc.Link("← 返回因子库", href="/factors", className="page-sub"),
            html.H4("公式因子编辑器", className="page-title"),
            html.P(
                "写公式 → 静态校验 → 保存草稿 → 一键评估 → 看三层贡献 → 入库。",
                className="page-sub",
            ),
            dcc.Location(id="fe-redirect", refresh=True),
            dbc.Row([dbc.Col(form, md=8), dbc.Col(right, md=4)], className="g-2"),
        ],
        fluid=True,
    )


def route_content(pathname: str | None):
    """子路由分派（纯函数）：/factors/new → 编辑器；/factors/<id> → 详情；
    其他（含 /factors 本身）返回 None，表示不接管（由 app.py::_route 调用）。"""
    if not pathname or not pathname.startswith("/factors/"):
        return None
    fid = pathname[len("/factors/") :].strip("/")
    if not fid:
        return None
    if fid == "new":
        return editor_layout()
    return detail_layout(fid)


def layout():
    return _overview_layout()


# ---------------------------------------------------------------------------
# 回调
# ---------------------------------------------------------------------------
def register(app):
    # 子路由（/factors/<id>、/factors/new）由 app.py::_route 的前缀分派调用
    # route_content() 实现——page-content 全应用只有 app.py 一个路由回调，
    # 不用 allow_duplicate 重复输出（客户端/服务端分派规则不一致会出随机路由错误）。

    @app.callback(
        Output("fw-table", "data"),
        Input("fw-f-cat", "value"),
        Input("fw-f-gate", "value"),
        Input("fw-f-enabled", "value"),
        Input("fw-f-search", "value"),
    )
    def _fw_filter(cat, gate, enabled, text):
        df = build_overview(
            dataio.factor_library(),
            dataio.factor_eval_table(),
            dataio.factor_ic_decay(),
            dataio.factor_corr(),
            _contrib_scores(),
            dataio.panel_factor_columns(),
        )
        return _table_records(apply_filters(df, cat, gate, enabled, text))

    @app.callback(
        Output("url", "pathname"),
        Input("fw-table", "active_cell"),
        State("fw-table", "derived_viewport_data"),
        prevent_initial_call=True,
    )
    def _fw_row_click(cell, view_records):
        # derived_viewport_data：排序/分页后当前页的行，与 active_cell.row 对齐
        if cell and view_records and 0 <= cell["row"] < len(view_records):
            return f"/factors/{view_records[cell['row']]['因子']}"
        return no_update

    @app.callback(
        Output("fw-table", "columns"),
        Input("fw-col-group", "value"),
    )
    def _fw_col_group(group):
        return _table_columns(group or "core")

    @app.callback(
        Output("fw-eval-modal", "is_open"),
        Output("fw-redirect", "pathname"),
        Output("fw-eval-msg", "children"),
        Input("fw-eval-btn", "n_clicks"),
        Input("fw-eval-confirm", "n_clicks"),
        Input("fw-eval-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def _fw_eval_modal(_open, _confirm, _cancel):
        # 「运行全库评估」→ 确认 Modal；确认即建会话并跳 /runs/<sid>，
        # 失败在 Modal 内原位报错、不跳页（参照编辑器 _fe_eval_now）
        if ctx.triggered_id == "fw-eval-confirm":
            try:
                sid = launch_full_eval_session()
            except Exception as ex:  # noqa: BLE001 —— 创建失败展示给用户
                return (
                    True,
                    no_update,
                    dbc.Alert(
                        f"创建评估会话失败：{type(ex).__name__}: {ex}",
                        color="danger",
                        className="mt-2 mb-0",
                    ),
                )
            return False, f"/runs/{sid}", no_update
        return ctx.triggered_id == "fw-eval-btn", no_update, no_update

    @app.callback(
        Output("fwd-redirect", "pathname"),
        Output("fwd-reeval-msg", "children"),
        Input("fwd-reeval-btn", "n_clicks"),
        State("fwd-fid", "data"),
        prevent_initial_call=True,
    )
    def _fwd_reeval(n_clicks, fid):
        # 详情页「重新评估该因子」：组件由 route_content 动态渲染，
        # 靠 suppress_callback_exceptions 生效（同编辑器 fe-eval-*）
        if not n_clicks or not fid:
            return no_update, no_update
        try:
            sid = launch_factoreval_session(str(fid))
        except Exception as ex:  # noqa: BLE001 —— 创建失败展示给用户，不跳页
            return no_update, dbc.Alert(
                f"创建评估会话失败：{type(ex).__name__}: {ex}",
                color="danger",
                className="mt-2 mb-0",
            )
        return f"/runs/{sid}", no_update

    def _editor_states():
        return [
            State("fe-id", "value"),
            State("fe-name", "value"),
            State("fe-category", "value"),
            State("fe-formula", "value"),
            State("fe-tags", "value"),
            State("fe-sign", "value"),
            State("fe-desc", "value"),
        ]

    def _validate_editor(fid, name, category, formula, tags, sign, desc):
        existing = (
            set(dataio.factor_library()["id"]) if not dataio.factor_library().empty else set()
        )
        known = set(dataio.panel_factor_columns()) | set(BASE_COLUMNS) | existing
        return validate_factor_draft(
            fid,
            name,
            category,
            formula,
            tags,
            sign,
            desc,
            existing_ids=existing,
            known_columns=known,
        )

    def _render_validation(res: dict) -> html.Div:
        if res["errors"]:
            return dbc.Alert(
                [html.Div(f"✗ {e}") for e in res["errors"]], color="danger", className="mb-0"
            )
        items = [html.Div("✓ 校验通过", style={"fontWeight": 700})]
        if res["deps"]:
            items.append(html.Div(f"依赖列: {', '.join(res['deps'])}"))
        for w in res["warnings"]:
            items.append(html.Div(f"⚠ {w}"))
        return dbc.Alert(
            items, color="success" if not res["warnings"] else "warning", className="mb-0"
        )

    @app.callback(
        Output("fe-validate-out", "children"),
        Input("fe-btn-validate", "n_clicks"),
        *_editor_states(),
        prevent_initial_call=True,
    )
    def _fe_validate(_n, fid, name, category, formula, tags, sign, desc):
        return _render_validation(_validate_editor(fid, name, category, formula, tags, sign, desc))

    @app.callback(
        Output("fe-save-out", "children"),
        Input("fe-btn-save", "n_clicks"),
        *_editor_states(),
        prevent_initial_call=True,
    )
    def _fe_save(_n, fid, name, category, formula, tags, sign, desc):
        res = _validate_editor(fid, name, category, formula, tags, sign, desc)
        if res["errors"]:
            return dbc.Alert(
                [html.Div("保存前请先通过校验：")] + [html.Div(f"✗ {e}") for e in res["errors"]],
                color="danger",
                className="mb-0",
            )
        record = dict(res["record"])
        record["created"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        yaml_path = os.path.join(dataio.FACTOR_LIBRARY, "custom.yaml")
        try:
            save_draft(record, yaml_path)
        except Exception as ex:  # noqa: BLE001 —— 写文件任何失败都展示给用户
            return dbc.Alert(f"保存失败：{type(ex).__name__}: {ex}", color="danger")
        dataio.refresh_caches()
        return _save_success_block(record)

    @app.callback(
        Output("fe-redirect", "pathname"),
        Output("fe-eval-msg", "children"),
        Input("fe-eval-btn", "n_clicks"),
        State("fe-eval-fid", "data"),
        prevent_initial_call=True,
    )
    def _fe_eval_now(n_clicks, fid):
        # 保存成功块里的按钮/Store 是动态渲染的，靠 suppress_callback_exceptions 生效
        if not n_clicks or not fid:
            return no_update, no_update
        try:
            sid = launch_factoreval_session(str(fid))
        except Exception as ex:  # noqa: BLE001 —— 创建失败展示给用户，不跳页
            return no_update, dbc.Alert(
                f"创建评估会话失败：{type(ex).__name__}: {ex}",
                color="danger",
                className="mt-2 mb-0",
            )
        return f"/runs/{sid}", no_update
