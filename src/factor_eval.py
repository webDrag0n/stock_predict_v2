"""单因子有效性评估框架（IC / ICIR / 分层回测 / 衰减 / 换手）。

这是持续因子优化路线的"度量衡"：任何候选因子在进入模型前，先在这里
过一遍单因子检验，只有达到准入门槛的因子才允许入库。核心指标：

  IC        —— 每个截面日，因子值与未来收益的相关系数，再对时间取均值
  RankIC    —— 用秩相关（Spearman），抗极值，A股日频更稳健
  ICIR      —— mean(IC) / std(IC)，衡量信号稳定性（年化可乘 sqrt(252/period)）
  分层回测   —— 按因子值分 N 层，看下期收益是否单调；Top-Bottom 多空收益
  衰减      —— IC 随预测期变化，决定调仓频率
  换手      —— 因子排序的稳定性 → 交易成本

设计为零外部依赖（只用 pandas/numpy/scipy），纯本地计算，不消耗任何 API。
配合 src/data.load_factor_panel() 使用——后者保留因子原名。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

import config as C

# 准入门槛（业界A股日频经验值，可按需调整）
GATE_RANKIC = 0.03      # |mean RankIC| 下限
GATE_ICIR = 0.30        # |ICIR| 下限
# 准入门槛。小横截面（如本项目仅 ~49 只股票）下 IC 噪声大，单纯卡 IC 绝对
# 值会过严，因此主门槛用 IC 序列的 t 统计量（衡量显著性，天然随样本量调整），
# 辅以放宽后的 RankICIR 与单调性。三者皆可按需调整。
GATE_IC_T = 2.0         # |IC t统计量| 下限——主门槛，统计显著性
GATE_RANKICIR = 0.15    # |RankICIR| 下限（小横截面49只，放宽至0.15；全市场常用0.3）
GATE_RANKIC = 0.02      # |mean RankIC| 下限
GATE_MONOTONIC = 0.5    # 分层单调性下限


@dataclass
class FactorReport:
    """单个因子的评估结果。"""
    name: str
    n_obs: int
    n_dates: int
    ic_mean: float
    ic_std: float
    icir: float
    icir_annual: float          # 年化 ICIR = ICIR * sqrt(252/period)
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    rank_icir_annual: float
    ic_t_stat: float            # IC 序列的 t 统计量（是否显著不为0）
    ic_positive_ratio: float    # IC > 0 的日子占比
    long_short_return: float    # 分层 Top-Bottom 多空累计收益
    monotonicity: float         # 分层收益单调性（层序与层收益的秩相关）
    turnover: float             # 因子排序的平均换手
    layer_returns: list = field(default_factory=list)  # 各层平均下期收益
    passed: bool = False        # 是否通过准入门槛

    def gate(self) -> "FactorReport":
        # 主门槛：IC 显著（t 检验）；辅以稳定性（RankICIR）与单调性。
        self.passed = bool(
            abs(self.ic_t_stat) >= GATE_IC_T
            and abs(self.rank_icir) >= GATE_RANKICIR
            and abs(self.rank_ic_mean) >= GATE_RANKIC
            and self.monotonicity >= GATE_MONOTONIC
        )
        return self


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def _daily_ic(panel: pd.DataFrame, factor: str, target: str,
              method: str) -> pd.Series:
    """每个截面日的 (Rank)IC 序列。method: 'pearson' | 'spearman'。"""
    def _one(g):
        x, y = g[factor], g[target]
        m = x.notna() & y.notna()
        if m.sum() < 5:
            return np.nan
        x, y = x[m], y[m]
        if x.nunique() < 2 or y.nunique() < 2:
            return np.nan
        if method == "spearman":
            return stats.spearmanr(x, y).correlation
        return stats.pearsonr(x, y)[0]
    return panel.groupby("date", group_keys=False).apply(_one, include_groups=False)


def _layered_returns(panel: pd.DataFrame, factor: str, target: str,
                     n_layers: int = 5) -> tuple[list[float], float, float]:
    """按因子值分 N 层，返回 (各层平均下期收益, Top-Bottom多空收益, 单调性)。

    每个截面日内把股票按因子值分 n_layers 组，组1=因子最小，组N=因子最大。
    返回各层在全样本上的平均下期收益、最高层减最低层、以及层序与层收益的秩相关。
    """
    def _assign(g):
        x = g[factor]
        if x.notna().sum() < n_layers:
            return pd.Series(np.nan, index=g.index)
        try:
            return pd.qcut(x.rank(method="first"), n_layers,
                           labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=g.index)

    layer = panel.groupby("date", group_keys=False).apply(
        _assign, include_groups=False)
    tmp = pd.DataFrame({"layer": layer.values, target: panel[target].values})
    tmp = tmp.dropna()
    if tmp.empty:
        return [np.nan] * n_layers, np.nan, np.nan
    layer_ret = tmp.groupby("layer")[target].mean()
    layer_ret = layer_ret.reindex(range(n_layers))
    vals = layer_ret.tolist()
    top, bottom = vals[-1], vals[0]
    ls = (top - bottom) if (not np.isnan(top) and not np.isnan(bottom)) else np.nan
    # 单调性：层序号与层收益的秩相关（接近1=完美单调递增）
    valid = layer_ret.dropna()
    if len(valid) >= 3:
        mono = stats.spearmanr(valid.index, valid.values).correlation
        mono = abs(mono) if not np.isnan(mono) else 0.0
    else:
        mono = 0.0
    return vals, float(ls) if not np.isnan(ls) else np.nan, float(mono)


def _turnover(panel: pd.DataFrame, factor: str, top_frac: float = 0.2) -> float:
    """因子 top 组的平均换手：相邻交易日 top 组成分的变动比例。

    用单次 groupby 取各日 top 组成分，避免在日期上重复布尔过滤（O(n*dates)）。
    """
    sub = panel[["date", "symbol", factor]].dropna()

    def _top(g):
        k = max(1, int(len(g) * top_frac))
        return set(g.nlargest(k, factor)["symbol"])

    tops = sub.groupby("date", group_keys=False).apply(_top, include_groups=False)
    prev_top, turns = None, []
    for top in tops:
        if not isinstance(top, set) or len(top) < 5:
            prev_top = top if isinstance(top, set) else prev_top
            continue
        if prev_top is not None and top:
            turns.append(len(top - prev_top) / len(top))
        prev_top = top
    return float(np.mean(turns)) if turns else np.nan


def evaluate_factor(panel: pd.DataFrame, factor: str, target: str = None,
                    n_layers: int = 5, winsorize: bool = True,
                    period: int = None) -> FactorReport:
    """对单个因子做完整评估，返回 FactorReport（已判定准入）。"""
    target = target or C.TARGET
    period = period or C.FORWARD_WINDOW
    df = panel[["date", "symbol", factor, target]].copy()
    df = df.dropna(subset=[factor, target])
    if winsorize:
        df[factor] = df.groupby("date", group_keys=False)[factor].apply(
            lambda s: _winsorize(s))

    ic = _daily_ic(df, factor, target, "pearson").dropna()
    ric = _daily_ic(df, factor, target, "spearman").dropna()

    ic_mean, ic_std = ic.mean(), ic.std()
    ric_mean, ric_std = ric.mean(), ric.std()
    icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
    ricir = ric_mean / ric_std if ric_std > 1e-12 else 0.0
    # 年化 ICIR：IC 按 period 日不重叠估计，年化乘 sqrt(252/period)
    ann = np.sqrt(252.0 / max(period, 1))
    icir_ann = icir * ann
    ricir_ann = ricir * ann
    # IC 序列 t 检验：均值是否显著不为0
    t_stat = (ic_mean / (ic_std / np.sqrt(len(ic)))) if (len(ic) > 1 and ic_std > 1e-12) else 0.0

    layer_ret, ls_ret, mono = _layered_returns(df, factor, target, n_layers)
    turn = _turnover(df, factor)

    rep = FactorReport(
        name=factor,
        n_obs=len(df),
        n_dates=int(df["date"].nunique()),
        ic_mean=float(ic_mean), ic_std=float(ic_std), icir=float(icir),
        icir_annual=float(icir_ann),
        rank_ic_mean=float(ric_mean), rank_ic_std=float(ric_std),
        rank_icir=float(ricir), rank_icir_annual=float(ricir_ann),
        ic_t_stat=float(t_stat),
        ic_positive_ratio=float((ic > 0).mean()) if len(ic) else 0.0,
        long_short_return=ls_ret,
        monotonicity=mono,
        turnover=turn,
        layer_returns=layer_ret,
    )
    return rep.gate()


def evaluate_panel(panel: pd.DataFrame, factors: list[str] = None,
                   target: str = None, n_layers: int = 5) -> pd.DataFrame:
    """评估面板中所有因子，返回按 |RankICIR| 排序的汇总表。"""
    from data import list_factor_columns
    factors = factors or list_factor_columns(panel)
    rows = []
    for f in factors:
        if panel[f].notna().sum() < 100 or panel[f].nunique() < 5:
            continue
        try:
            r = evaluate_factor(panel, f, target=target, n_layers=n_layers)
            rows.append(r)
        except Exception as ex:
            print(f"  因子 {f} 评估失败: {ex}")
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame([{
        "factor": r.name,
        "RankIC": r.rank_ic_mean,
        "RankICIR": r.rank_icir,
        "RankICIR_ann": r.rank_icir_annual,
        "IC": r.ic_mean,
        "ICIR": r.icir,
        "IC_t": r.ic_t_stat,
        "IC>0占比": r.ic_positive_ratio,
        "多空收益": r.long_short_return,
        "单调性": r.monotonicity,
        "换手": r.turnover,
        "准入": "✓" if r.passed else "",
    } for r in rows])
    # 按显著性（|IC_t|）排序——小横截面下这是最可靠的强弱指标
    table["_abs"] = table["IC_t"].abs()
    table = table.sort_values("_abs", ascending=False).drop(columns="_abs")
    return table.reset_index(drop=True)


def select_passing_factors(panel: pd.DataFrame, date_lo=None, date_hi=None,
                           target: str = None) -> list[str]:
    """在 [date_lo, date_hi] 训练区间内评估因子，返回通过准入门槛的因子名。

    用于滚动回测：每个窗口的因子选择只用该窗口的训练段，绝不碰测试段，
    从根本上消除"因子选择偷看测试集"的泄露。date_lo/hi 为 None 时用全段。
    返回命名因子列表（非 feature_NN）。
    """
    sub = panel
    if date_lo is not None:
        sub = sub[sub["date"] >= date_lo]
    if date_hi is not None:
        sub = sub[sub["date"] <= date_hi]
    if sub.empty:
        return []
    table = evaluate_panel(sub, target=target)
    if table.empty:
        return []
    return table[table["准入"] == "✓"]["factor"].tolist()


def format_panel_report(table: pd.DataFrame) -> str:
    """把评估汇总表格式化为可读文本。"""
    if table.empty:
        return "（无可评估因子）"
    n_pass = int((table["准入"] == "✓").sum())
    lines = [
        f"单因子评估汇总（共 {len(table)} 个因子，{n_pass} 个通过准入门槛）按 |IC_t| 排序",
        f"准入门槛: |IC_t|≥{GATE_IC_T}(显著) 且 |RankICIR|≥{GATE_RANKICIR} "
        f"且 |RankIC|≥{GATE_RANKIC} 且 单调性≥{GATE_MONOTONIC}",
        "-" * 100,
        f"{'因子':<22}{'RankIC':>9}{'RankICIR':>10}{'年化':>8}{'IC_t':>8}"
        f"{'多空收益':>10}{'单调性':>8}{'换手':>8}{'准入':>6}",
        "-" * 100,
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r['factor']:<22}{r['RankIC']:>+9.4f}{r['RankICIR']:>+10.3f}"
            f"{r['RankICIR_ann']:>+8.2f}{r['IC_t']:>+8.2f}{r['多空收益']:>+10.4f}"
            f"{r['单调性']:>8.2f}{r['换手']:>8.2f}{r['准入']:>6}")
    return "\n".join(lines)


def _layer_equity(panel: pd.DataFrame, factor: str, target: str,
                  n_layers: int) -> pd.DataFrame:
    """各层逐日平均收益 -> 累计净值（用于画图）。返回 index=date 的净值 DataFrame。"""
    df = panel[["date", factor, target]].dropna(subset=[factor, target]).copy()
    df[factor] = df.groupby("date", group_keys=False)[factor].apply(_winsorize)

    def _assign(g):
        if g[factor].notna().sum() < n_layers:
            return pd.Series(np.nan, index=g.index)
        try:
            return pd.qcut(g[factor].rank(method="first"), n_layers,
                           labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=g.index)

    df["layer"] = df.groupby("date", group_keys=False).apply(
        _assign, include_groups=False)
    df = df.dropna(subset=["layer"])
    # 每日各层平均收益
    daily = df.groupby(["date", "layer"])[target].mean().unstack("layer")
    daily = daily.sort_index()
    # 累计净值（按 period 重叠，这里用简单累乘近似展示趋势）
    equity = (1.0 + daily.fillna(0.0)).cumprod()
    return equity


def plot_layered_returns(panel: pd.DataFrame, factors: list[str],
                         out_path: str, target: str = None,
                         n_layers: int = 5) -> str:
    """Layered cumulative returns for up to 6 factors (2×3 subplots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"

    target = target or C.TARGET
    factors = factors[:6]
    n = len(factors)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.0 * nrow),
                             squeeze=False)
    cmap = plt.get_cmap("RdYlGn")
    for i, fac in enumerate(factors):
        ax = axes[i // ncol][i % ncol]
        eq = _layer_equity(panel, fac, target, n_layers)
        x = np.arange(len(eq))
        for li, col in enumerate(eq.columns):
            ax.plot(x, eq[col].values, lw=1.1,
                    color=cmap(li / max(n_layers - 1, 1)),
                    label=f"Q{int(col)+1}")
        ax.set_title(fac, fontsize=10)
        ax.axhline(1.0, color="grey", lw=0.6, ls="--", label="Baseline=1")
        ax.set_ylabel("Cum. return (1=start)", fontsize=7)
        ax.set_xlabel("Trading day", fontsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=n_layers + 1, loc="upper left")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(
        f"Factor Quintile Return Spreads  —  {n_layers} layers, "
        f"Q1=lowest factor value, Q{n_layers}=highest",
        fontsize=11, y=1.01)

    note = ("How to read:  Each subplot shows one factor split into quintiles "
            f"(Q1=lowest {100//n_layers}%, Q{n_layers}=highest {100//n_layers}%) "
            "sorted daily across all stocks.\n"
            "Cumulative return of each quintile is plotted over time (starting at 1.0).\n"
            "A good factor shows Q1 < Q2 < … < Q5 (monotone spread): "
            "high-ranked stocks should outperform low-ranked ones.\n"
            "Diverging fan = strong factor; flat or crossing lines = weak/noisy factor.")
    fig.text(0.01, -0.04, note, fontsize=7.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#cccccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def select_features(eval_csv: str = None) -> list[str] | None:
    """从因子评估CSV读取通过准入门槛的因子名，映射回 feature_NN 名称。

    load_data 会把命名因子按顺序重命名为 feature_00..78，这里做逆映射：
    先从 load_factor_panel 拿到因子顺序，再转成 feature_NN。
    若 CSV 不存在或无通过的因子，返回 None（不做筛选）。
    """
    import os
    eval_csv = eval_csv or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "outputs", "factor_eval.csv")
    if not os.path.exists(eval_csv):
        return None
    tbl = pd.read_csv(eval_csv)
    passed = tbl[tbl["准入"] == "✓"]["factor"].tolist()
    if not passed:
        return None

    # Build the factor → feature_NN mapping by replicating the same selection
    # logic as transform_to_jane_street_format (exclude raw/intermediate cols).
    # We only need the ordered list of candidate feature names; we can get it
    # cheaply from a tiny synthetic frame with the same column structure.
    try:
        from data import load_factor_panel, list_factor_columns
        import config as C
        # Read one cached parquet bar for column structure without hitting API
        import glob
        cache_files = glob.glob(os.path.join(
            C.CACHE_DIR, "tushare", "bar_*.parquet"))[:1]
        if not cache_files:
            return None
        sample = pd.read_parquet(cache_files[0]).head(1)
        # Add dummy columns to mirror what compute_technical_features would produce
        # Instead: read the factor names directly from the eval CSV ordering,
        # match them to C.FEATURE_NAMES by position from the panel column order.
        # The simplest robust approach: the eval CSV lists named factors; we need
        # their position in the feature_NN ordering. Use the panel itself.
        # Re-load just the column names (no data fetch — bars are cached).
        panel = load_factor_panel(verbose=False)
        factor_order = list_factor_columns(panel)
        # Build name→feature_NN map (same truncation as transform_to_js_format)
        n = C.N_FEATURES
        name_to_feat = {name: f"feature_{i:02d}"
                        for i, name in enumerate(factor_order[:n])}
        selected = [name_to_feat[f] for f in passed if f in name_to_feat]
        if not selected:
            return None
        print(f"特征筛选: {len(passed)} 个命名因子通过准入 → "
              f"{len(selected)} 个 feature_NN 特征")
        return selected
    except Exception as ex:
        print(f"特征筛选映射失败（{ex}），使用全部特征")
        return None
