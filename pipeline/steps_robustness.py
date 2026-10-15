"""多 Seed 稳健性模式的流水线步骤（随机种子离散度复核）。

引擎组合：
  · scripts/research/multiseed_robust.run_one_seed —— 逐 seed walkforward
    （自带 data_cache/seed_oos_<seed>.parquet 缓存，命中即秒回）
  · src/multiseed_eval.seed_backtest —— 合并 amount 后跑周期调仓回测
    （含平方根冲击成本），产出逐日 net/bench/成本序列与 NW alpha 指标

amount 映射取自 data_cache/wholemkt_panel.parquet 的 (date, symbol, amount)
投影列（原 plot_multiseed_dashboard 的做法）；缓存不存在时退回 load_factor_panel。

报告视图（替代已删除的 matplotlib dashboard_multiseed.png）：
  图A 逐 seed 净值淡线 + 均值粗线 + 基准 · 图B 累计 alpha 均值 ±1σ 带
  图C 均值净值滚动 63 日年化 Sharpe · 图D 回撤均值 + seed min/max 带
  图E 成本拆解（佣金 vs 冲击，累计）· 图F 21 日块收益热力图 · 表G 逐 seed 汇总

注：seed_backtest 的持仓策略固定为 long_topk（src 侧签名约束），
本模式不暴露 strategy 参数。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import config as C
from multiseed_eval import seed_backtest

from . import context as ctxmod
from .context import kpi, table, fig_line, fig_area_stacked, fig_heatmap, report

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PANEL_CACHE = os.path.join(_ROOT, "data_cache", "wholemkt_panel.parquet")
_SEED_CACHE = os.path.join(_ROOT, "data_cache", "seed_oos_{seed}.parquet")
# 与 scripts/research/multiseed_robust.META 一致：面板里非因子的 meta 列
_META = {"date", "symbol", "responder_6", "close", "return_1d", "pct_chg",
         "amount", "weight", "date_id", "time_id", "symbol_id"}

_METRIC_KEYS = ["sharpe", "calmar", "maxdd", "alpha_ann", "nw_t", "beta",
                "total_pnl", "cost_comm", "cost_impact", "hit"]
_SERIES_KEYS = ["net", "bench", "strat_cum_pct", "bench_cum_pct", "alpha_pct",
                "tc_comm", "tc_impact"]

_STRAT_COLOR = "#1a7f37"
_BENCH_COLOR = "#ff7f0e"
_ALPHA_COLOR = "#7b3fb5"
_DD_COLOR = "#d62728"
_COMM_COLOR = "#2563eb"
_IMPACT_COLOR = "#c2410c"


def _parse_seeds(raw) -> list[int]:
    """解析逗号分隔的 seed 列表；空项/非整数跳过，全空回退默认。"""
    seeds = []
    for tok in str(raw or "").replace("，", ",").split(","):
        tok = tok.strip()
        if tok:
            try:
                seeds.append(int(tok))
            except ValueError:
                continue
    return seeds or [42, 123, 7]


def _run_one_seed(panel: pd.DataFrame, factors: list[str], seed: int):
    """scripts/research/multiseed_robust.run_one_seed 的懒加载包装。

    scripts/research 不是包，按需把目录加进 sys.path 再 import（其模块顶部
    会自行补项目根/src 路径）。自带 seed_oos_<seed>.parquet 逐 seed 缓存。
    """
    d = os.path.join(_ROOT, "scripts", "research")
    if d not in sys.path:
        sys.path.insert(0, d)
    from multiseed_robust import run_one_seed
    return run_one_seed(panel, factors, seed)


def _load_amount_map() -> pd.DataFrame:
    """(date, symbol) -> amount(千元) 映射。优先面板缓存的投影列，避免整面板加载。"""
    if os.path.exists(_PANEL_CACHE):
        return pd.read_parquet(_PANEL_CACHE, columns=["date", "symbol", "amount"])
    from data import load_factor_panel
    return load_factor_panel(keep_price=True)[["date", "symbol", "amount"]]


def _stack(runs: dict, key: str) -> np.ndarray:
    """对齐各 seed 的等长序列，堆成 (n_seeds, T)。截到最短长度。"""
    arrs = [np.asarray(r[key], dtype=np.float64) for r in runs.values()]
    L = min(len(a) for a in arrs)
    return np.vstack([a[:L] for a in arrs])


# ---------------------------------------------------------------------------
def ms_run(ctx: ctxmod.StepContext) -> dict:
    """逐 seed：取 OOS（缓存优先，缺失则训练）→ seed_backtest → 落盘序列/指标。"""
    seeds = _parse_seeds(ctx.params.get("seeds"))
    amount_map = _load_amount_map()

    runs = {}
    panel = factors = None       # 懒加载：全部 seed 命中缓存时不需要训练面板
    for i, sd in enumerate(seeds):
        cache = _SEED_CACHE.format(seed=sd)
        hit_cache = os.path.exists(cache)
        ctx.report_progress("逐 seed 回测", i, len(seeds),
                            f"seed {sd}（{'缓存' if hit_cache else '训练'}）")
        if hit_cache:
            oos = pd.read_parquet(cache)
        else:
            if panel is None:
                if os.path.exists(_PANEL_CACHE):
                    panel = pd.read_parquet(_PANEL_CACHE)
                else:
                    from data import load_factor_panel
                    panel = load_factor_panel(keep_price=True)
                factors = [c for c in panel.columns if c not in _META]
            oos = _run_one_seed(panel, factors, sd)

        r = seed_backtest(oos, amount_map)
        dts = np.sort(oos["date"].unique())
        n = len(r["net"])
        runs[sd] = {**r, "dts": dts[:n]}

        # 落盘逐 seed 逐日序列（ms_report 可仅凭 artifact 重建）
        ser = pd.DataFrame({"date": dts[:n]})
        for key in _SERIES_KEYS:
            arr = np.asarray(r[key], dtype=np.float64)
            ser[key] = arr[:n] if len(arr) >= n else np.pad(
                arr, (0, n - len(arr)), constant_values=np.nan)
        ser.to_parquet(ctx.artifact_path(f"seed_{sd}_series.parquet"), index=False)

    ctx.shared["ms_runs"] = runs
    mdf = pd.DataFrame([{"seed": sd, **{k: r[k] for k in _METRIC_KEYS}}
                        for sd, r in runs.items()])
    mdf.to_parquet(ctx.artifact_path("seed_metrics.parquet"), index=False)
    ctx.report_progress("逐 seed 回测", len(seeds), len(seeds), "完成")

    sh = mdf["sharpe"].to_numpy()
    aa = mdf["alpha_ann"].to_numpy()
    n_sig = int((mdf["nw_t"].abs() >= 2).sum())
    n_pos = int((mdf["alpha_ann"] > 0).sum())
    kpis = [
        kpi("seed 数", f"{len(seeds)}", f"{seeds}"),
        kpi("Sharpe", f"{sh.mean():+.2f}±{sh.std():.2f}",
            f"min {sh.min():+.2f} / max {sh.max():+.2f}"),
        kpi("年化 alpha", f"{aa.mean():+.1%}±{aa.std():.1%}", ""),
        kpi("NW-t≥2 的 seed", f"{n_sig}/{len(seeds)}",
            f"alpha>0 的 seed {n_pos}/{len(seeds)}"),
    ]
    return report("ms_run", "多 Seed 回测",
                  summary=(f"{len(seeds)} seeds · Sharpe {sh.mean():+.2f}"
                           f"±{sh.std():.2f} · α {aa.mean():+.1%}±{aa.std():.1%}"),
                  kpis=kpis)


# ---------------------------------------------------------------------------
def _load_runs(ctx: ctxmod.StepContext) -> dict:
    """取逐 seed 结果：优先 ctx.shared（同进程），否则从 artifact 重建。"""
    runs = ctx.shared.get("ms_runs")
    if runs:
        return runs
    mdf = pd.read_parquet(ctx.artifact_path("seed_metrics.parquet"))
    runs = {}
    for _, row in mdf.iterrows():
        sd = int(row["seed"])
        ser = pd.read_parquet(ctx.artifact_path(f"seed_{sd}_series.parquet"))
        r = {k: float(row[k]) for k in _METRIC_KEYS}
        for key in _SERIES_KEYS:
            r[key] = ser[key].to_numpy(dtype=np.float64)
        r["dts"] = ser["date"].to_numpy()
        runs[sd] = r
    return runs


def ms_report(ctx: ctxmod.StepContext) -> dict:
    """稳健性报告：净值/alpha/滚动Sharpe/回撤/成本/月块热力图/汇总表。"""
    ctx.report_progress("构造报告", 0, 1)
    runs = _load_runs(ctx)
    if not runs:
        return report("ms_report", "稳健性报告", summary="无可用 seed",
                      notes=["⚠️ ms_run 未产生任何 seed 结果"])
    seeds = list(runs.keys())
    cap0 = float(ctx.params.get("capital") or C.INITIAL_CAPITAL)

    net_s = _stack(runs, "net")                      # (S, T)
    T = net_s.shape[1]
    mean_net = net_s.mean(axis=0)
    dts0 = np.asarray(runs[seeds[0]]["dts"])
    x = ([str(pd.Timestamp(d).date()) for d in dts0[:T]]
         if len(dts0) >= T else list(range(T)))
    x_is_date = len(dts0) >= T

    acct_s = cap0 * np.cumprod(1.0 + net_s, axis=1)
    mean_acct = cap0 * np.cumprod(1.0 + mean_net)
    alpha_s = _stack(runs, "alpha_pct")
    mean_alpha = alpha_s.mean(axis=0)
    std_alpha = alpha_s.std(axis=0)
    bench_cum = _stack(runs, "bench_cum_pct").mean(axis=0)
    bench_acct = cap0 * (1.0 + bench_cum / 100.0)

    # ---- 图A：逐 seed 净值淡线 + 均值粗线 + 基准 ----
    series_a = [{"name": f"seed {sd}", "y": acct_s[i], "opacity": 0.35,
                 "color": _STRAT_COLOR} for i, sd in enumerate(seeds)]
    series_a.append({"name": "均值策略", "y": mean_acct, "width": 2.5,
                     "color": _STRAT_COLOR})
    series_a.append({"name": "等权基准", "y": bench_acct,
                     "color": _BENCH_COLOR, "dash": "dash"})
    figures = [fig_line("账户净值 —— 逐 seed（淡） vs 均值（粗） vs 基准", x,
                        series_a, yaxis="账户价值 (元)", x_is_date=x_is_date)]

    # ---- 图B：累计 alpha 均值 ±1σ 带 ----
    series_b = [{"name": f"seed {sd}", "y": alpha_s[i], "opacity": 0.35,
                 "color": _ALPHA_COLOR} for i, sd in enumerate(seeds)]
    series_b.append({"name": "均值 alpha ±1σ", "y": mean_alpha, "width": 2.5,
                     "color": _ALPHA_COLOR,
                     "band": {"upper": mean_alpha + std_alpha,
                              "lower": mean_alpha - std_alpha}})
    figures.append(fig_line("累计 alpha（策略 − 基准），seed 离散带", x,
                            series_b, yaxis="累计超额 (百分点)",
                            x_is_date=x_is_date))

    # ---- 图C：均值净值的滚动 63 日年化 Sharpe ----
    roll = pd.Series(mean_net)
    rsharpe = (roll.rolling(63).mean() / roll.rolling(63).std()
               * np.sqrt(252)).to_numpy()
    figures.append(fig_line("滚动 63 日年化 Sharpe（均值净值）", x,
                            [{"name": "rolling Sharpe", "y": rsharpe,
                              "color": _STRAT_COLOR}],
                            yaxis="Sharpe", x_is_date=x_is_date))

    # ---- 图D：回撤均值 + seed min/max 带 ----
    dd_s = np.zeros_like(acct_s)
    for i in range(len(seeds)):
        pk = np.maximum.accumulate(acct_s[i])
        dd_s[i] = (acct_s[i] - pk) / pk * 100.0
    mean_pk = np.maximum.accumulate(mean_acct)
    mean_dd = (mean_acct - mean_pk) / mean_pk * 100.0
    figures.append(fig_line(
        "回撤（均值 + seed 范围带）", x,
        [{"name": "均值回撤", "y": mean_dd, "color": _DD_COLOR, "width": 2.0,
          "band": {"upper": dd_s.max(axis=0), "lower": dd_s.min(axis=0)}}],
        yaxis="回撤 %", x_is_date=x_is_date))

    # ---- 图E：成本拆解（佣金+印花税 vs 市场冲击，累计 ¥，均值） ----
    comm_cum = cap0 * np.cumsum(_stack(runs, "tc_comm").mean(axis=0))
    impact_cum = cap0 * np.cumsum(_stack(runs, "tc_impact").mean(axis=0))
    figures.append(fig_area_stacked(
        "累计交易成本拆解（各 seed 均值）", x,
        [{"name": "佣金+印花税", "y": comm_cum, "color": _COMM_COLOR},
         {"name": "市场冲击(√)", "y": impact_cum, "color": _IMPACT_COLOR}],
        yaxis="累计成本 (元)", x_is_date=x_is_date))

    # ---- 图F：21 日块收益热力图（均值净值） ----
    monthly = np.array([np.prod(1.0 + mean_net[i:i + 21]) - 1.0
                        for i in range(0, T, 21)]) * 100.0
    ncol = 6
    nrow = int(np.ceil(len(monthly) / ncol))
    grid = np.full((nrow, ncol), np.nan)
    for i, mret in enumerate(monthly):
        grid[i // ncol, i % ncol] = mret
    figures.append(fig_heatmap(
        "均值收益热力图（21 日块，%）", grid,
        x=[f"块{j + 1}" for j in range(ncol)],
        y=[f"M{i * ncol + 1}-{i * ncol + ncol}" for i in range(nrow)],
        xaxis="", yaxis=""))

    # ---- 表G：逐 seed 汇总 + MEAN±SD ----
    rows = []
    for sd in seeds:
        r = runs[sd]
        rows.append({"seed": str(sd),
                     "Sharpe": f"{r['sharpe']:+.2f}",
                     "Calmar": f"{r['calmar']:+.2f}",
                     "MaxDD": f"{r['maxdd']:+.1%}",
                     "Alpha(ann)": f"{r['alpha_ann']:+.1%}",
                     "NW-t": f"{r['nw_t']:+.2f}",
                     "Beta": f"{r['beta']:+.2f}",
                     "日胜率": f"{r['hit']:.1%}",
                     "佣金¥": f"{r['cost_comm'] * cap0:,.0f}",
                     "冲击¥": f"{r['cost_impact'] * cap0:,.0f}"})

    def _ms(key, fmt="{:.2f}"):
        v = np.array([runs[s][key] for s in seeds], dtype=np.float64)
        return fmt.format(v.mean()) + "±" + fmt.format(v.std())

    rows.append({"seed": "MEAN±SD",
                 "Sharpe": _ms("sharpe", "{:+.2f}"),
                 "Calmar": _ms("calmar", "{:+.2f}"),
                 "MaxDD": _ms("maxdd", "{:+.1%}"),
                 "Alpha(ann)": _ms("alpha_ann", "{:+.1%}"),
                 "NW-t": _ms("nw_t", "{:+.2f}"),
                 "Beta": _ms("beta", "{:+.2f}"),
                 "日胜率": _ms("hit", "{:.1%}"),
                 "佣金¥": f"{np.mean([runs[s]['cost_comm'] for s in seeds]) * cap0:,.0f}",
                 "冲击¥": f"{np.mean([runs[s]['cost_impact'] for s in seeds]) * cap0:,.0f}"})
    tbl = pd.DataFrame(rows)

    sh = np.array([runs[s]["sharpe"] for s in seeds])
    aa = np.array([runs[s]["alpha_ann"] for s in seeds])
    tt = np.array([runs[s]["nw_t"] for s in seeds])
    bt = np.array([runs[s]["beta"] for s in seeds])
    n_sig = int((np.abs(tt) >= 2).sum())
    n_pos = int((aa > 0).sum())
    robust = n_sig >= (len(seeds) + 1) // 2 and n_pos == len(seeds)

    kpis = [
        kpi("Sharpe", f"{sh.mean():+.2f}±{sh.std():.2f}", "跨 seed"),
        kpi("年化 alpha", f"{aa.mean():+.1%}±{aa.std():.1%}",
            f"NW-t {tt.mean():+.2f}±{tt.std():.2f}"),
        kpi("Beta", f"{bt.mean():+.2f}", "vs 等权基准"),
        kpi("NW-t≥2", f"{n_sig}/{len(seeds)}", f"alpha>0: {n_pos}/{len(seeds)}"),
        kpi("期末均值净值", f"¥{mean_acct[-1]:,.0f}",
            f"总收益 {mean_acct[-1] / cap0 - 1.0:+.1%}"),
        kpi("判定", "稳健 ✓" if robust else "不稳健 ⚠",
            "多数 seed 显著正 alpha → 稳健"),
    ]
    notes = [
        "淡线 = 单个随机种子；粗线 = 均值。聚拢 = 结果对 seed 稳健。",
        "图B 带 = 跨 seed ±1σ；图E 把平坦佣金与主导小盘股的平方根冲击成本拆开"
        "（平坦费率模型会漏掉后者）。NW-t≥2 = alpha 统计显著。",
    ]
    if not robust:
        notes.append("⚠️ alpha 对 seed 敏感：多数 seed NW-t<2 或存在负 alpha seed")

    ctx.report_progress("构造报告", 1, 1)
    return report("ms_report", "稳健性报告",
                  summary=(f"{len(seeds)} seeds · Sharpe {sh.mean():+.2f}"
                           f"±{sh.std():.2f} · α {aa.mean():+.1%}±{aa.std():.1%} · "
                           f"{'稳健' if robust else '不稳健'}"),
                  kpis=kpis, figures=figures, tables=[table(
                      "逐 seed 汇总（末行 MEAN±SD）", tbl, max_rows=50)],
                  notes=notes)
