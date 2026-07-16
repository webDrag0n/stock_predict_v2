"""实盘成本真实化 + 消融回测（ablation）。

在已验证的 ret_weighted 全市场 OOS 上，逐层叠加 A 股真实交易摩擦，看每一层
吃掉多少 +13.8% 的 alpha：

  L0  仅佣金+印花税            (当前回测基线)
  L1  +平方根冲击成本          (已在生产引擎)
  L2  +买卖价差半跨            (小盘 bid-ask)
  L3  +涨跌停不可成交          (涨停不能买/跌停不能卖 → 持有原仓，路径依赖)
  L4  +参与率上限              (单次调仓每股成交 ≤ X% ADV → 未成交部分留到下次)

L3/L4 让持仓变成路径依赖（被拦的买单意味着你保持旧仓），所以这里逐个调仓日
迭代、携带"实际持仓"向前，而非生产引擎的向量化 ffill。

成本/约束参数集中在顶部，便于敏感性分析。收益用 adj 后的 return_1d，与修正
后的面板一致。alpha/NW-t 对等权全市场基准、Newey-West(lag=20) 稳健。
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C

SEEDS = (42, 123, 7)
TOPK = 0.15
REBAL = 20
CAPITAL = 1_000_000.0

# --- cost / friction parameters (the levers for realism) ---
COMMISSION = 0.0002       # 双边佣金
STAMP = 0.0010            # 卖出印花税
IMPACT_K = 0.1            # 平方根冲击系数
HALF_SPREAD = 0.0010      # 买卖价差的一半（小盘 ~10-20bp，取 10bp）
PARTICIPATION = 0.10      # 单次调仓每股最多成交 10% ADV


def _load_oos(seed: int) -> pd.DataFrame:
    """Load a seed's ret_weighted OOS, merge limit_flag/pct_chg/amount from panel."""
    oos = pd.read_parquet(f"data_cache/seed_oos_{seed}.parquet")
    panel = pd.read_parquet("data_cache/wholemkt_panel.parquet",
                            columns=["date", "symbol", "limit_flag", "pct_chg", "amount"])
    return oos.merge(panel, on=["date", "symbol"], how="left")


def _positions_long_topk(oos: pd.DataFrame, topk: float) -> pd.DataFrame:
    """Per-date long_topk target weights (rank-proportional, sum=1). Returns
    a frame with date, sym, target, ret, limit_flag, pct_chg, adv."""
    df = pd.DataFrame({
        "date": oos["date"].to_numpy(),
        "sym": oos["symbol"].to_numpy(),
        "pred": oos["prediction"].to_numpy(dtype=np.float64),
        "ret": oos["return_1d"].to_numpy(dtype=np.float64),
        "limit": oos["limit_flag"].fillna(0).to_numpy(dtype=np.float64),
        "pct": oos["pct_chg"].fillna(0).to_numpy(dtype=np.float64),
        "adv": oos["amount"].fillna(0).to_numpy(dtype=np.float64) * 1000.0,  # 千元→元
    })
    rp = df.groupby("date")["pred"].rank(pct=True, method="first")
    sel = rp > (1.0 - topk)
    w = np.where(sel, rp, 0.0)
    df["w"] = w
    tot = df.groupby("date")["w"].transform("sum")
    df["target"] = np.where(tot > 1e-12, df["w"] / tot, 0.0)
    return df

def _backtest(df: pd.DataFrame, level: int, return_series: bool = False) -> dict:
    """Path-dependent periodic backtest with friction layers up to `level`.

    level: 0=comm/stamp, 1=+impact, 2=+spread, 3=+limit-block, 4=+participation.
    Carries ACTUAL (constrained) holdings forward so limit-blocks and partial
    fills are represented. Earns each held name's daily return between rebalances.
    """
    use_impact = level >= 1
    use_spread = level >= 2
    use_limit = level >= 3
    use_partic = level >= 4

    dates = np.sort(df["date"].unique())
    rebal = set(dates[::REBAL].tolist())
    by_date = {d: g for d, g in df.groupby("date")}

    held: dict[str, float] = {}        # sym -> current weight
    daily_net = []
    daily_gross = []
    daily_cost = []
    cost_breakdown = {"comm": 0.0, "impact": 0.0, "spread": 0.0}

    for d in dates:
        g = by_date[d]
        sym = g["sym"].to_numpy()
        ret = np.nan_to_num(g["ret"].to_numpy(), nan=0.0)
        # 1) earn today's return on weights held coming INTO today
        gross = sum(held.get(s, 0.0) * r for s, r in zip(sym, ret))

        cost = 0.0
        if d in rebal:
            tgt = dict(zip(sym, g["target"].to_numpy()))
            limit = dict(zip(sym, g["limit"].to_numpy()))
            pct = dict(zip(sym, g["pct"].to_numpy()))
            adv = dict(zip(sym, g["adv"].to_numpy()))
            universe = set(tgt) | set(held)
            new_held = dict(held)
            for s in universe:
                cur = held.get(s, 0.0)
                want = tgt.get(s, 0.0)
                dw = want - cur
                if abs(dw) < 1e-9:
                    continue
                # L3: limit-up can't buy, limit-down can't sell → keep cur weight
                if use_limit and limit.get(s, 0.0) >= 1.0:
                    up = pct.get(s, 0.0) > 0
                    down = pct.get(s, 0.0) < 0
                    if (dw > 0 and up) or (dw < 0 and down):
                        new_held[s] = cur          # blocked: stay put
                        continue
                # L4: cap traded fraction to PARTICIPATION * ADV
                if use_partic and adv.get(s, 0.0) > 0:
                    max_dw = PARTICIPATION * adv[s] / CAPITAL
                    if abs(dw) > max_dw:
                        dw = np.sign(dw) * max_dw
                        want = cur + dw            # partial fill
                new_held[s] = want
                # --- costs on the executed |dw| ---
                buy = max(dw, 0.0)
                sell = max(-dw, 0.0)
                comm = buy * COMMISSION + sell * (COMMISSION + STAMP)
                cost += comm
                cost_breakdown["comm"] += comm
                if use_spread:
                    sp = abs(dw) * HALF_SPREAD
                    cost += sp
                    cost_breakdown["spread"] += sp
                if use_impact and adv.get(s, 0.0) > 0:
                    partic = abs(dw) * CAPITAL / adv[s]
                    imp = IMPACT_K * np.sqrt(partic) * abs(dw)
                    cost += imp
                    cost_breakdown["impact"] += imp
            held = {s: w for s, w in new_held.items() if abs(w) > 1e-9}

        daily_gross.append(gross)
        daily_cost.append(cost)
        daily_net.append(gross - cost)

    net = np.array(daily_net)
    # benchmark: equal-weight whole market daily mean return
    bench = df.groupby("date")["ret"].mean().reindex(dates).fillna(0.0).to_numpy()
    if return_series:
        return {"net_series": net, "dates": dates, "bench": bench}
    return _metrics(net, bench, dates, cost_breakdown)


def _metrics(net, bench, dates, cb) -> dict:
    from measure_alpha import _newey_west_alpha_t
    n = len(net)
    ann = 252
    cum = float(np.prod(1.0 + net) - 1.0)
    cagr = (1.0 + cum) ** (ann / n) - 1.0 if n else 0.0
    sharpe = net.mean() / net.std(ddof=1) * np.sqrt(ann) if net.std() > 1e-12 else 0.0
    # CAPM alpha vs equal-weight benchmark, Newey-West (lag=20) robust t (manual,
    # no statsmodels dependency — same estimator measure_alpha/multiseed use).
    alpha_d, beta, nw_t, _r2 = _newey_west_alpha_t(net, bench, 20)
    alpha_ann = alpha_d * ann
    eq = np.cumprod(1.0 + net)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    tot_cost = sum(cb.values())
    return {"cum": cum, "cagr": cagr, "sharpe": sharpe, "alpha_ann": alpha_ann,
            "nw_t": nw_t, "beta": beta, "maxdd": dd,
            "cost_comm": cb["comm"], "cost_impact": cb["impact"],
            "cost_spread": cb["spread"], "cost_total": tot_cost}


LEVELS = {0: "L0 仅佣金+印花税", 1: "L1 +冲击成本", 2: "L2 +买卖价差",
          3: "L3 +涨跌停拦截", 4: "L4 +参与率上限(完整真实)"}


def main():
    print(f"实盘成本真实化消融 | ret_weighted 全市场 | topk={TOPK} rebal={REBAL}d "
          f"| seeds={list(SEEDS)}")
    print(f"参数: 佣金{COMMISSION} 印花{STAMP} 冲击k{IMPACT_K} 半价差{HALF_SPREAD} "
          f"参与率{PARTICIPATION}\n")
    # positions per seed (cached prediction → deterministic)
    pos_by_seed = {}
    for s in SEEDS:
        pos_by_seed[s] = _positions_long_topk(_load_oos(s), TOPK)
        print(f"  seed {s}: {pos_by_seed[s]['date'].nunique()} 日 已就绪")
    print()

    rows = []
    for lvl in range(5):
        per_seed = [_backtest(pos_by_seed[s], lvl) for s in SEEDS]
        agg = {k: float(np.mean([r[k] for r in per_seed]))
               for k in per_seed[0]}
        rows.append({"level": LEVELS[lvl], **agg})
        print(f"{LEVELS[lvl]:22s} α={agg['alpha_ann']*100:+6.2f}% "
              f"NW-t={agg['nw_t']:+.2f} Sharpe={agg['sharpe']:+.2f} "
              f"累计={agg['cum']*100:+6.1f}% MaxDD={agg['maxdd']*100:.1f}% "
              f"成本={agg['cost_total']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv("outputs/realistic_cost_ablation.csv", index=False)
    print("-> outputs/realistic_cost_ablation.csv")
    bd = {k: float(np.mean([_backtest(pos_by_seed[s], 4)["cost_" + k]
                            for s in SEEDS]))
          for k in ["comm", "impact", "spread"]}
    print(f"\n完整真实(L4)成本分解(seed均值): 佣金/印花={bd['comm']:.4f} "
          f"冲击={bd['impact']:.4f} 价差={bd['spread']:.4f}")


def _exposure_schedule(dates, bench, mode, regime_lb=60, regime_floor=0.3,
                       vol_lb=60, target_vol=0.20):
    """Leak-free per-date exposure multiplier in [0,1] from TRAILING market data.

    mode: 'regime' = de-risk to regime_floor when trailing regime_lb-day market
    cum-return < 0; 'vol' = scale to hit target_vol from trailing vol; 'both' =
    product. Each date d uses only bench[:d] (strictly before d) → no look-ahead.
    bench is the daily equal-weight whole-market return series aligned to dates.
    """
    bench = np.asarray(bench, dtype=np.float64)
    n = len(dates)
    exp = np.ones(n)
    for i in range(n):
        reg = vol = 1.0
        if i >= regime_lb:
            trail = bench[i - regime_lb:i]            # strictly before i
            if np.prod(1.0 + trail) - 1.0 < 0.0:
                reg = regime_floor
        if i >= vol_lb:
            rv = bench[i - vol_lb:i].std() * np.sqrt(252)
            vol = min(1.0, target_vol / rv) if rv > 1e-9 else 1.0
        if mode == "regime":
            exp[i] = reg
        elif mode == "vol":
            exp[i] = vol
        else:
            exp[i] = reg * vol
    return exp


def overlay():
    """Test leak-free exposure overlays vs the un-overlaid baseline, under FULL
    realistic cost (L4). Goal: cut MaxDD -32.7% without killing alpha. Exposure
    scales target weights (cash earns 0); the path-dependent engine charges the
    turnover cost of de-risking, so the alpha/DD tradeoff is honest."""
    global CAPITAL
    CAPITAL = 1e7   # ¥1000万: within capacity, costs realistic
    print(f"\n减回撤叠加层 | 完整真实成本(L4) | ¥{CAPITAL/1e8:.2f}亿 topk={TOPK} {REBAL}日")
    print("(leak-free: 每日暴露仅用严格过去的市场数据; cash earns 0)\n")
    oos_by_seed = {s: _load_oos(s) for s in SEEDS}
    pos_by_seed = {s: _positions_long_topk(oos_by_seed[s], TOPK) for s in SEEDS}

    configs = [("基线(无叠加)", None, {}),
               ("regime 60d 地板0.3", "regime", {"regime_lb": 60, "regime_floor": 0.3}),
               ("regime 60d 地板0.0", "regime", {"regime_lb": 60, "regime_floor": 0.0}),
               ("vol目标20%", "vol", {"vol_lb": 60, "target_vol": 0.20}),
               ("vol目标15%", "vol", {"vol_lb": 60, "target_vol": 0.15}),
               ("both(regime0.3 + vol20%)", "both",
                {"regime_lb": 60, "regime_floor": 0.3, "vol_lb": 60, "target_vol": 0.20})]
    rows = []
    for name, mode, kw in configs:
        per = []
        for s in SEEDS:
            pos = pos_by_seed[s].copy()
            if mode is not None:
                dates = np.sort(pos["date"].unique())
                bench = pos.groupby("date")["ret"].mean().reindex(dates).fillna(0).to_numpy()
                exp = _exposure_schedule(dates, bench, mode, **kw)
                exp_map = dict(zip(dates, exp))
                pos["target"] = pos["target"] * pos["date"].map(exp_map).to_numpy()
            per.append(_backtest(pos, 4))
        agg = {k: float(np.mean([r[k] for r in per])) for k in per[0]}
        rows.append({"config": name, **agg})
        print(f"  {name:24s} α={agg['alpha_ann']*100:+6.2f}% NW-t={agg['nw_t']:+.2f} "
              f"Sharpe={agg['sharpe']:+.2f} Calmar={agg['cagr']/abs(agg['maxdd']):+.2f} "
              f"MaxDD={agg['maxdd']*100:+.1f}% 累计={agg['cum']*100:+.0f}%")
    pd.DataFrame(rows).to_csv("outputs/overlay_drawdown.csv", index=False)
    print("\n-> outputs/overlay_drawdown.csv")


def hedge():
    """Principled DD fix: hedge the market-beta with a short index-future leg.

    The trailing overlays failed because the -32% DD is structural beta (long-only
    A-shares eat the crash), not a trailing-timeable regime. The textbook fix is a
    short broad-index leg → market-neutral. We proxy the index future with the
    equal-weight whole-market return (the book is mid/small-cap, so this is a rough
    but directionally honest hedge). Net daily = long_book_net - hedge_ratio*bench.
    Index-future cost ~ commission only (no stamp on futures); we add a small bps.
    Leak-free: hedge_ratio is a fixed prior (1.0 = full beta neutralize), bench is
    the same-day market return the long book is exposed to."""
    global CAPITAL
    CAPITAL = 1e7
    print(f"\n指数对冲(空头腿→市场中性) | 完整真实成本(L4) | ¥{CAPITAL/1e8:.2f}亿 topk={TOPK}")
    print("(空头腿=做空等权全市场代理指数; hedge_ratio=对冲掉的beta比例)\n")
    from measure_alpha import _newey_west_alpha_t
    oos_by_seed = {s: _load_oos(s) for s in SEEDS}
    pos_by_seed = {s: _positions_long_topk(oos_by_seed[s], TOPK) for s in SEEDS}
    FUT_COST = 0.00005  # index-future round-trip ~0.5bp/day exposure, rough
    for hr in (0.0, 0.5, 0.8, 1.0):
        agg_rows = []
        for s in SEEDS:
            pos = pos_by_seed[s]
            r = _backtest(pos, 4, return_series=True)
            net = r["net_series"]; dates = r["dates"]
            bench = pos.groupby("date")["ret"].mean().reindex(dates).fillna(0).to_numpy()
            hedged = net - hr * bench - (hr * FUT_COST if hr > 0 else 0.0)
            a_d, beta, t, _ = _newey_west_alpha_t(hedged, bench, 20)
            eq = np.cumprod(1 + hedged); dd = float((eq/np.maximum.accumulate(eq)-1).min())
            cum = float(np.prod(1+hedged)-1)
            sh = hedged.mean()/hedged.std(ddof=1)*np.sqrt(252) if hedged.std()>1e-12 else 0
            agg_rows.append({"alpha_ann": a_d*252, "nw_t": t, "beta": beta,
                             "maxdd": dd, "cum": cum, "sharpe": sh})
        agg = {k: float(np.mean([r[k] for r in agg_rows])) for k in agg_rows[0]}
        print(f"  hedge={hr:.1f}: α={agg['alpha_ann']*100:+6.2f}% NW-t={agg['nw_t']:+.2f} "
              f"β={agg['beta']:+.2f} Sharpe={agg['sharpe']:+.2f} "
              f"MaxDD={agg['maxdd']*100:+.1f}% 累计={agg['cum']*100:+.0f}%")
    print("\n注: 空头腿用等权全市场代理; 实盘需股指期货(IC/IM对中小盘更贴)且不可做空个股。")


def _positions_liq_tilt(oos: pd.DataFrame, topk: float, beta: float) -> pd.DataFrame:
    """Liquidity-tilted long_topk: among the top-K predicted names, weight each by
    rank * ADV**beta instead of rank alone. beta=0 reproduces pure rank-proportional
    (current production); beta>0 shifts capital toward liquid names, trading a little
    signal purity for much higher capacity (lower impact at scale). Selection set
    (which names are long) is unchanged — only the *within-book weighting* shifts."""
    df = pd.DataFrame({
        "date": oos["date"].to_numpy(),
        "sym": oos["symbol"].to_numpy(),
        "pred": oos["prediction"].to_numpy(dtype=np.float64),
        "ret": oos["return_1d"].to_numpy(dtype=np.float64),
        "limit": oos["limit_flag"].fillna(0).to_numpy(dtype=np.float64),
        "pct": oos["pct_chg"].fillna(0).to_numpy(dtype=np.float64),
        "adv": oos["amount"].fillna(0).to_numpy(dtype=np.float64) * 1000.0,
    })
    out = []
    for d, g in df.groupby("date"):
        g = g.copy()
        rp = g["pred"].rank(pct=True, method="first")
        sel = rp > (1.0 - topk)
        base = np.where(sel, rp, 0.0)
        liq = np.where(sel, np.power(np.clip(g["adv"].to_numpy(), 1.0, None), beta), 0.0)
        w = base * liq
        tot = w.sum()
        g["target"] = (w / tot) if tot > 1e-12 else 0.0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def capacity_optimize():
    """Compare capacity curves: pure rank (beta=0, production) vs liquidity-tilted
    (beta>0). Shows how much AUM the strategy can hold at NW-t>2 under each."""
    global CAPITAL
    print(f"\n流动性倾斜优化 | 完整真实成本(L4) | topk={TOPK} rebal={REBAL}d")
    print("(beta=0=当前生产纯排名加权; beta>0=向高流动性倾斜)\n")
    oos_by_seed = {s: _load_oos(s) for s in SEEDS}
    caps = [1e7, 5e7, 1e8, 3e8, 5e8, 1e9]
    for beta in (0.0, 0.5, 1.0):
        pos = {s: _positions_liq_tilt(oos_by_seed[s], TOPK, beta) for s in SEEDS}
        print(f"--- beta={beta} ---")
        for cap in caps:
            CAPITAL = cap
            per = [_backtest(pos[s], 4) for s in SEEDS]
            agg = {k: float(np.mean([r[k] for r in per])) for k in per[0]}
            flag = "✓" if agg["nw_t"] >= 2 else ""
            print(f"  ¥{cap/1e8:>5.2f}亿: α={agg['alpha_ann']*100:+6.2f}% "
                  f"NW-t={agg['nw_t']:+.2f} Sharpe={agg['sharpe']:+.2f} "
                  f"冲击={agg['cost_impact']:.4f} {flag}")
        print()


def capacity():
    """Capacity sweep: alpha vs AUM under the FULL realistic model (L4). At larger
    capital the participation cap binds and impact grows, eroding alpha — the real
    'how much can this strategy hold' question that ¥1M cannot answer."""
    global CAPITAL
    print(f"\n资金容量扫描 | 完整真实成本(L4) | topk={TOPK} rebal={REBAL}d\n")
    pos_by_seed = {s: _positions_long_topk(_load_oos(s), TOPK) for s in SEEDS}
    caps = [1e6, 1e7, 5e7, 1e8, 3e8, 5e8, 1e9]
    rows = []
    for cap in caps:
        CAPITAL = cap
        per_seed = [_backtest(pos_by_seed[s], 4) for s in SEEDS]
        agg = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}
        rows.append({"capital": cap, **agg})
        print(f"  资金 ¥{cap/1e8:>5.2f}亿: α={agg['alpha_ann']*100:+6.2f}% "
              f"NW-t={agg['nw_t']:+.2f} Sharpe={agg['sharpe']:+.2f} "
              f"成本={agg['cost_total']:.4f} 冲击={agg['cost_impact']:.4f}")
    pd.DataFrame(rows).to_csv("outputs/capacity_sweep.csv", index=False)
    print("-> outputs/capacity_sweep.csv")
    print("\n-> outputs/realistic_cost_ablation.csv")
    # cost decomposition at full realism
    full = [_backtest(pos_by_seed[s], 4) for s in SEEDS]
    print(f"\n完整真实(L4)成本分解(seed均值): 佣金/印花="
          f"{np.mean([r['cost_comm'] for r in full]):.4f} "
          f"冲击={np.mean([r['cost_impact'] for r in full]):.4f} "
          f"价差={np.mean([r['cost_spread'] for r in full]):.4f}")


if __name__ == "__main__":
    import sys
    if "--hedge" in sys.argv:
        hedge()
    elif "--overlay" in sys.argv:
        overlay()
    elif "--optimize" in sys.argv:
        capacity_optimize()
    elif "--capacity" in sys.argv:
        capacity()
    else:
        main()

