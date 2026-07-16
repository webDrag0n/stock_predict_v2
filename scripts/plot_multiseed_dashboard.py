"""多seed回测仪表盘: 每个seed淡色、平均浓色, 含流动性成本模型与丰富视图。

复用缓存的逐seed OOS(data_cache/seed_oos_*.parquet), 合并回 amount(冲击成本需要),
对每个seed跑周期调仓回测(含平方根冲击成本), 收集各seed的净值/回撤/alpha曲线,
画成多面板仪表盘: 每个seed淡线 + 平均浓线。

面板:
  A. 账户净值(¥) — 各seed淡线 + 平均浓线 + 等权基准
  B. 累计Alpha(%) — 各seed淡线 + 平均浓线 ± 1个标准差带
  C. 各窗口Sharpe(分组柱: 各seed+均值) / 或滚动63日Sharpe
  D. 回撤(%) — 平均线 + seed范围带
  E. 成本拆解(佣金 vs 冲击成本)累计(¥)
  F. 月度收益热力图(平均)
  G. 汇总统计表(各seed + 均值±std)

用法: TRAIN_WINDOW=500 SEEDS=42,123,7 .venv/bin/python plot_multiseed_dashboard.py
"""
import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config as C
from walkforward import _with_ids
from strategy import generate_positions
from backtest import run_backtest_periodic
from measure_alpha import _newey_west_alpha_t

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
plt.rcParams["font.family"] = "DejaVu Sans"

PANEL_CACHE = "data_cache/wholemkt_panel.parquet"
PALETTE = {"strat": "#1a7f37", "bench": "#ff7f0e", "alpha": "#7b3fb5",
           "loss": "#d62728", "impact": "#c2410c", "commission": "#2563eb",
           "faint": "#9ca3af", "grid": "#e5e7eb"}


def _seed_backtest(oos, amount_map):
    """单seed: 合并amount → 周期调仓回测(含冲击成本) → 返回curve+metrics+alpha序列。"""
    oos = oos.merge(amount_map, on=["date", "symbol"], how="left")
    oos_ids = _with_ids(oos)
    pred = oos["prediction"].to_numpy()
    pos = generate_positions(oos_ids, pred, strategy="long_topk")
    # carry amount into the ids frame (run_backtest_periodic reads df['amount'])
    oos_ids["amount"] = oos["amount"].to_numpy()
    res = run_backtest_periodic(oos_ids, pred, pos,
                                rebalance_days=C.REBALANCE_DAYS,
                                pred_in_return_space=True)
    curve = res["curve"]; m = res["metrics"]
    net = curve["net"].to_numpy()
    bench = oos.groupby("date")["return_1d"].mean().sort_index().to_numpy()
    n = min(len(net), len(bench))
    a_d, beta, t_nw, r2 = _newey_west_alpha_t(net[:n], bench[:n], C.REBALANCE_DAYS)
    strat_cum = np.cumprod(1 + net) - 1.0
    bench_cum = np.cumprod(1 + bench[:n]) - 1.0
    return {
        "net": net, "bench": bench[:n],
        "strat_cum_pct": strat_cum * 100, "bench_cum_pct": bench_cum * 100,
        "alpha_pct": (strat_cum[:n] - bench_cum) * 100,
        "tc_comm": curve["tc_commission"].to_numpy(),
        "tc_impact": curve["tc_impact"].to_numpy(),
        "sharpe": m["sharpe"], "calmar": m["calmar"], "maxdd": m["max_drawdown"],
        "alpha_ann": a_d * 252, "nw_t": t_nw, "beta": beta,
        "total_pnl": m["total_pnl"], "cost_comm": m["total_cost_commission"],
        "cost_impact": m["total_cost_impact"], "hit": m["hit_rate"],
    }


def main():
    C.MODEL_OBJECTIVE = "ret_weighted"
    seeds = [int(s) for s in os.environ.get("SEEDS", "42,123,7").split(",")]
    cap0 = C.INITIAL_CAPITAL

    # amount map from panel (date, symbol) -> amount(千元)
    panel = pd.read_parquet(PANEL_CACHE, columns=["date", "symbol", "amount"])
    print(f"加载 amount 映射: {len(panel):,} 行")

    runs = {}
    for sd in seeds:
        cache = f"data_cache/seed_oos_{sd}.parquet"
        if not os.path.exists(cache) and sd == 42:
            cache = "data_cache/wholemkt_oos.parquet"
        if not os.path.exists(cache):
            print(f"  seed {sd}: 无缓存 {cache}, 跳过"); continue
        oos = pd.read_parquet(cache)
        runs[sd] = _seed_backtest(oos, panel)
        r = runs[sd]
        print(f"  seed {sd}: Sharpe={r['sharpe']:+.2f} αann={r['alpha_ann']:+.2%} "
              f"NW-t={r['nw_t']:+.2f} 佣金成本={r['cost_comm']:.4f} "
              f"冲击成本={r['cost_impact']:.4f} MaxDD={r['maxdd']:+.3f}", flush=True)

    if not runs:
        print("无可用seed, 退出"); return
    _plot(runs, cap0, "outputs/dashboard_multiseed.png")
    print("-> outputs/dashboard_multiseed.png")


def _stack(runs, key):
    """对齐各seed的等长序列, 堆成 (n_seeds, T)。截到最短长度。"""
    arrs = [r[key] for r in runs.values()]
    L = min(len(a) for a in arrs)
    return np.vstack([a[:L] for a in arrs])


def _plot(runs, cap0, out_path):
    seeds = list(runs.keys())
    nS = len(seeds)
    # mean curves
    net_s = _stack(runs, "net")                       # (S, T)
    mean_net = net_s.mean(axis=0)
    acct_s = cap0 * np.cumprod(1 + net_s, axis=1)
    mean_acct = cap0 * np.cumprod(1 + mean_net)
    alpha_s = _stack(runs, "alpha_pct")
    mean_alpha = alpha_s.mean(axis=0); std_alpha = alpha_s.std(axis=0)
    bench_cum = _stack(runs, "bench_cum_pct").mean(axis=0)
    bench_acct = cap0 * (1 + bench_cum / 100)
    T = net_s.shape[1]; x = np.arange(T)

    # drawdown % per seed
    dd_s = np.zeros_like(acct_s)
    for i in range(nS):
        peak = np.maximum.accumulate(acct_s[i])
        dd_s[i] = (acct_s[i] - peak) / peak * 100
    mean_peak = np.maximum.accumulate(mean_acct)
    mean_dd = (mean_acct - mean_peak) / mean_peak * 100

    # cost breakdown (mean cumulative ¥)
    comm_cum = cap0 * np.cumsum(_stack(runs, "tc_comm").mean(axis=0))
    impact_cum = cap0 * np.cumsum(_stack(runs, "tc_impact").mean(axis=0))

    fig = plt.figure(figsize=(16, 19))
    gs = gridspec.GridSpec(5, 2, height_ratios=[2.3, 1.3, 1.3, 1.3, 1.1],
                           hspace=0.42, wspace=0.20)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, :])
    axC = fig.add_subplot(gs[2, 0])
    axD = fig.add_subplot(gs[2, 1])
    axE = fig.add_subplot(gs[3, 0])
    axF = fig.add_subplot(gs[3, 1])
    axG = fig.add_subplot(gs[4, :])

    # ---- title with mean±std summary ----
    sh = np.array([runs[s]["sharpe"] for s in seeds])
    aa = np.array([runs[s]["alpha_ann"] for s in seeds])
    tt = np.array([runs[s]["nw_t"] for s in seeds])
    bt = np.array([runs[s]["beta"] for s in seeds])
    end_ret = (mean_acct[-1] / cap0 - 1) * 100
    fig.suptitle(
        f"Whole-Market Walk-forward Dashboard — {nS} seeds (ret_weighted, liquidity-cost model)\n"
        f"Start ¥{cap0:,.0f} → Mean End ¥{mean_acct[-1]:,.0f} ({end_ret:+.1f}%)   |   "
        f"Sharpe {sh.mean():.2f}±{sh.std():.2f}   "
        f"Alpha {aa.mean():+.1%}±{aa.std():.1%}   "
        f"NW-t {tt.mean():.2f}±{tt.std():.2f}   Beta {bt.mean():.2f}",
        fontsize=12.5, y=0.995)

    # ===== A. account value =====
    for i, s in enumerate(seeds):
        axA.plot(x, acct_s[i], color=PALETTE["strat"], lw=0.8, alpha=0.28,
                 label="Per-seed" if i == 0 else None)
    axA.plot(x, mean_acct, color=PALETTE["strat"], lw=2.6, label="Mean strategy")
    axA.plot(x, bench_acct, color=PALETTE["bench"], lw=1.6, ls="--",
             label="Equal-weight benchmark")
    axA.axhline(cap0, color="grey", lw=0.7, ls=":")
    axA.fill_between(x, cap0, mean_acct, where=(mean_acct >= cap0),
                     color=PALETTE["strat"], alpha=0.08)
    axA.set_title("A. Account Value — each seed (faint) vs mean (bold) vs benchmark",
                  fontsize=11, loc="left", fontweight="bold")
    axA.set_ylabel("Account (¥)")
    axA.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"¥{v/1e6:.2f}M"))
    axA.legend(loc="upper left", fontsize=9, framealpha=0.9)
    axA.grid(alpha=0.3)
    axA.annotate(f"Mean end: ¥{mean_acct[-1]/1e6:.2f}M\n"
                 f"Benchmark: ¥{bench_acct[-1]/1e6:.2f}M",
                 xy=(x[-1], mean_acct[-1]), fontsize=9, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eefaf0", ec=PALETTE["strat"]))

    # ===== B. cumulative alpha with ±1σ band across seeds =====
    for i in range(nS):
        axB.plot(x, alpha_s[i], color=PALETTE["alpha"], lw=0.7, alpha=0.25)
    axB.plot(x, mean_alpha, color=PALETTE["alpha"], lw=2.4, label="Mean alpha")
    axB.fill_between(x, mean_alpha - std_alpha, mean_alpha + std_alpha,
                     color=PALETTE["alpha"], alpha=0.15, label="±1σ across seeds")
    axB.axhline(0, color="grey", lw=0.8, ls="--")
    axB.set_title("B. Cumulative Alpha (strategy − benchmark), seed dispersion band",
                  fontsize=11, loc="left", fontweight="bold")
    axB.set_ylabel("Alpha (%)"); axB.set_xlabel("Trading day")
    axB.legend(loc="upper left", fontsize=9); axB.grid(alpha=0.3)
    axB.annotate(f"End alpha: {mean_alpha[-1]:+.1f}%",
                 xy=(x[-1], mean_alpha[-1]), fontsize=9, ha="right",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#f4f0ff", ec=PALETTE["alpha"]))

    # ===== C. rolling 63-day annualized Sharpe (mean) =====
    win = 63
    roll = pd.Series(mean_net)
    rsharpe = (roll.rolling(win).mean() / roll.rolling(win).std()
               * np.sqrt(252)).to_numpy()
    axC.plot(x, rsharpe, color=PALETTE["strat"], lw=1.5)
    axC.axhline(0, color="grey", lw=0.7); axC.axhline(1, color="#16a34a", lw=0.7, ls=":")
    axC.fill_between(x, rsharpe, 0, where=(rsharpe >= 0), color=PALETTE["strat"], alpha=0.15)
    axC.fill_between(x, rsharpe, 0, where=(rsharpe < 0), color=PALETTE["loss"], alpha=0.15)
    axC.set_title(f"C. Rolling {win}-day Annualized Sharpe (mean)",
                  fontsize=11, loc="left", fontweight="bold")
    axC.set_ylabel("Sharpe"); axC.set_xlabel("Trading day"); axC.grid(alpha=0.3)

    # ===== D. drawdown band =====
    axD.fill_between(x, dd_s.min(axis=0), dd_s.max(axis=0),
                     color=PALETTE["loss"], alpha=0.15, label="Seed range")
    axD.plot(x, mean_dd, color=PALETTE["loss"], lw=1.8, label="Mean drawdown")
    axD.set_title("D. Drawdown (% from peak) — mean + seed range",
                  fontsize=11, loc="left", fontweight="bold")
    axD.set_ylabel("Drawdown (%)"); axD.set_xlabel("Trading day")
    axD.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axD.legend(loc="lower left", fontsize=9); axD.grid(alpha=0.3)
    axD.annotate(f"Mean MaxDD: {mean_dd.min():.1f}%",
                 xy=(x[int(np.argmin(mean_dd))], mean_dd.min()), fontsize=9,
                 ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fdeeee", ec=PALETTE["loss"]))

    # ===== E. cost breakdown: commission vs impact (cumulative ¥) =====
    axE.plot(x, comm_cum, color=PALETTE["commission"], lw=1.8, label="Commission+stamp")
    axE.plot(x, impact_cum, color=PALETTE["impact"], lw=1.8, label="Market impact (sqrt)")
    axE.fill_between(x, 0, comm_cum, color=PALETTE["commission"], alpha=0.12)
    axE.fill_between(x, 0, impact_cum, color=PALETTE["impact"], alpha=0.12)
    axE.set_title("E. Cumulative Transaction Cost — commission vs liquidity impact",
                  fontsize=11, loc="left", fontweight="bold")
    axE.set_ylabel("Cumulative cost (¥)"); axE.set_xlabel("Trading day")
    axE.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"¥{v/1e3:.0f}K"))
    axE.legend(loc="upper left", fontsize=9); axE.grid(alpha=0.3)
    tot_comm = comm_cum[-1]; tot_impact = impact_cum[-1]
    axE.annotate(f"Commission: ¥{tot_comm/1e3:.0f}K\nImpact: ¥{tot_impact/1e3:.0f}K",
                 xy=(x[-1], max(tot_comm, tot_impact)), fontsize=9, ha="right",
                 va="top", bbox=dict(boxstyle="round,pad=0.3", fc="#fff7ed",
                                     ec=PALETTE["impact"]))

    # ===== F. monthly return heatmap (mean) =====
    # build a date axis: we don't have calendar here, approximate by 21-day months
    monthly = []
    for i in range(0, T, 21):
        seg = mean_net[i:i+21]
        monthly.append(np.prod(1 + seg) - 1.0)
    monthly = np.array(monthly) * 100
    nm = len(monthly)
    ncol = 6
    nrow = int(np.ceil(nm / ncol))
    grid = np.full((nrow, ncol), np.nan)
    for i, mret in enumerate(monthly):
        grid[i // ncol, i % ncol] = mret
    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
    im = axF.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(nrow):
        for j in range(ncol):
            if np.isfinite(grid[i, j]):
                axF.text(j, i, f"{grid[i,j]:+.1f}", ha="center", va="center",
                         fontsize=7.5, color="black")
    axF.set_title("F. Mean Return by ~Month (21-day blocks, %)",
                  fontsize=11, loc="left", fontweight="bold")
    axF.set_xticks([]); axF.set_yticks(range(nrow))
    axF.set_yticklabels([f"M{i*ncol+1}-{i*ncol+ncol}" for i in range(nrow)], fontsize=7)
    fig.colorbar(im, ax=axF, fraction=0.046, pad=0.04)

    # ===== G. summary stats table =====
    axG.axis("off")
    rows = []
    for s in seeds:
        r = runs[s]
        rows.append([f"{s}", f"{r['sharpe']:.2f}", f"{r['calmar']:.2f}",
                     f"{r['maxdd']*100:.1f}%", f"{r['alpha_ann']:+.1%}",
                     f"{r['nw_t']:.2f}", f"{r['beta']:.2f}",
                     f"{r['hit']:.1%}",
                     f"¥{r['cost_comm']*cap0/1e3:.0f}K",
                     f"¥{r['cost_impact']*cap0/1e3:.0f}K"])
    # mean±std row
    def ms(key, pct=False, t=False):
        v = np.array([runs[s][key] for s in seeds])
        if pct: return f"{v.mean():+.1%}±{v.std():.1%}"
        return f"{v.mean():.2f}±{v.std():.2f}"
    rows.append(["MEAN±SD", ms("sharpe"), ms("calmar"),
                 f"{np.mean([runs[s]['maxdd'] for s in seeds])*100:.1f}%",
                 ms("alpha_ann", pct=True), ms("nw_t"), ms("beta"),
                 f"{np.mean([runs[s]['hit'] for s in seeds]):.1%}",
                 f"¥{np.mean([runs[s]['cost_comm'] for s in seeds])*cap0/1e3:.0f}K",
                 f"¥{np.mean([runs[s]['cost_impact'] for s in seeds])*cap0/1e3:.0f}K"])
    cols = ["Seed", "Sharpe", "Calmar", "MaxDD", "Alpha(ann)", "NW-t", "Beta",
            "Hit", "Comm¥", "Impact¥"]
    tbl = axG.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#1a7f37"); tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for j in range(len(cols)):
        tbl[(len(rows), j)].set_facecolor("#dcfce7"); tbl[(len(rows), j)].set_text_props(fontweight="bold")
    axG.set_title("G. Per-seed & Mean Summary (whole-market, liquidity-cost model)",
                  fontsize=11, loc="left", fontweight="bold")

    note = ("How to read: faint lines = individual random seeds; bold line = their mean. Tight clustering = result robust to seed. "
            "Panel B band = ±1σ alpha across seeds. Panel E separates flat commission from the sqrt market-impact cost that dominates "
            "for illiquid small-caps — the realistic drag a flat-fee model misses. NW-t≥2 = statistically significant alpha.")
    fig.text(0.012, 0.012, note, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f9fafb", ec="#cccccc"))

    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
