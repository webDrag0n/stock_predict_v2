"""多seed稳健性复核: 全市场 ret_weighted 的 +16% alpha 是否对随机种子稳健?

为每个种子设 C.RANDOM_STATE 后跑单seed walkforward(N_SEEDS=1), 逐seed缓存OOS,
计算每个seed的 top20超额/IC/Sharpe/年化alpha/NW-t, 看离散度。种子42复用已缓存
的 wholemkt_oos.parquet。判据: 多数seed的 NW-t>2 且 alpha同号 → 稳健。

用法: TRAIN_WINDOW=500 SEEDS=42,123,7 .venv/bin/python multiseed_robust.py
"""
import os, sys, time
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import _train_predict, _with_ids
from factor_eval import _windows, _daily_rank_ic
from strategy import generate_positions
from backtest import run_backtest_periodic
from measure_alpha import _newey_west_alpha_t

PANEL_CACHE = "data_cache/wholemkt_panel.parquet"
META = {"date", "symbol", "responder_6", "close", "return_1d", "pct_chg",
        "amount", "weight", "date_id", "time_id", "symbol_id"}


def top_excess(pred, test, frac=0.2):
    t = test[["date", C.TARGET]].copy(); t["p"] = pred
    def one(g):
        k = max(1, int(round(len(g) * frac)))
        return g.nlargest(k, "p")[C.TARGET].mean() - g[C.TARGET].mean()
    return t.groupby("date", group_keys=False).apply(one, include_groups=False).dropna()


def run_one_seed(panel, factors, seed):
    """单seed walkforward(N_SEEDS=1), 返回拼接OOS。逐seed缓存。

    每个seed严格用自己的 seed_oos_{seed}.parquet。曾有一个"seed42复用
    wholemkt_oos.parquet"的捷径——但那个文件会被 verify_wholemkt.py 用任意
    MODEL_OBJECTIVE(如ic_linear)覆盖,导致seed42加载到错的预测器(2026-06-27
    踩坑:seed42读到ic_linear预测,IC/alpha与123/7完全不同)。已移除该捷径。
    """
    cache = f"data_cache/seed_oos_{seed}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    C.RANDOM_STATE = seed
    os.environ["N_SEEDS"] = "1"
    dates = np.sort(panel["date"].unique())
    frames = []
    for wid, tr, te in _windows(dates):
        t0 = time.time()
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()
        test["prediction"] = _train_predict(train, test, factors)
        frames.append(test[["date", "symbol", "prediction", C.TARGET, "return_1d"]])
        print(f"    seed{seed} 窗口{wid}: {time.time()-t0:.0f}s", flush=True)
    oos = pd.concat(frames, ignore_index=True)
    oos.to_parquet(cache, index=False)
    return oos


def metrics_for(oos, seed):
    # IC over all dates + top20 excess + periodic backtest
    ic = _daily_rank_ic(oos.rename(columns={"prediction": "__p"}), "__p", C.TARGET).dropna()
    ex = top_excess(oos["prediction"].to_numpy(), oos)
    oos_ids = _with_ids(oos)
    pos = generate_positions(oos_ids, oos["prediction"].to_numpy(), strategy="long_topk")
    res = run_backtest_periodic(oos_ids, oos["prediction"].to_numpy(), pos,
                                rebalance_days=C.REBALANCE_DAYS, pred_in_return_space=True)
    m = res["metrics"]
    strat = res["curve"]["net"].to_numpy()
    bench = oos.groupby("date")["return_1d"].mean().sort_index().to_numpy()
    n = min(len(strat), len(bench))
    a_d, beta, t_nw, r2 = _newey_west_alpha_t(strat[:n], bench[:n], C.REBALANCE_DAYS)
    return {"seed": seed, "IC": float(ic.mean()),
            "ICIR": float(ic.mean()/ic.std()) if ic.std() > 1e-12 else np.nan,
            "top20_excess": float(ex.mean()), "sharpe": m["sharpe"],
            "calmar": m["calmar"], "maxdd": m["max_drawdown"],
            "beta": beta, "alpha_ann": a_d*252, "nw_t": t_nw}


def main():
    C.MODEL_OBJECTIVE = "ret_weighted"
    seeds = [int(s) for s in os.environ.get("SEEDS", "42,123,7").split(",")]
    panel = pd.read_parquet(PANEL_CACHE)
    factors = [c for c in panel.columns if c not in META]
    print(f"多seed复核 | 全市场 {len(panel):,}行 {panel['symbol'].nunique()}股 "
          f"{len(factors)}因子 | ret_weighted单seed | TRAIN={C.TRAIN_WINDOW} | seeds={seeds}",
          flush=True)

    rows, ens_frames = [], []
    for sd in seeds:
        print(f"\n--- seed {sd} ---", flush=True)
        oos = run_one_seed(panel, factors, sd)
        rows.append(metrics_for(oos, sd))
        r = rows[-1]
        print(f"  seed{sd}: IC={r['IC']:+.4f} top20={r['top20_excess']:+.5f} "
              f"Sharpe={r['sharpe']:+.2f} alpha={r['alpha_ann']:+.2%} NW-t={r['nw_t']:+.2f}",
              flush=True)
        # for ensemble: per-date percentile rank of this seed's prediction
        ens_frames.append(oos[["date", "symbol", "prediction"]].rename(
            columns={"prediction": f"p{sd}"}))

    tbl = pd.DataFrame(rows)
    tbl.to_csv("outputs/multiseed_robust.csv", index=False)
    print("\n===== 多seed离散度 =====", flush=True)
    for col in ["IC", "top20_excess", "sharpe", "alpha_ann", "nw_t"]:
        s = tbl[col]
        print(f"  {col:<13}: 均值={s.mean():+.5f} 标准差={s.std():.5f} "
              f"min={s.min():+.5f} max={s.max():+.5f}", flush=True)
    n_sig = int((tbl["nw_t"].abs() >= 2).sum())
    n_pos = int((tbl["alpha_ann"] > 0).sum())
    print(f"\n  NW-t≥2 的seed: {n_sig}/{len(tbl)} | alpha>0 的seed: {n_pos}/{len(tbl)}",
          flush=True)
    verdict = ("稳健✓: 多数seed显著正alpha" if n_sig >= (len(tbl)+1)//2 and n_pos == len(tbl)
               else "不稳健⚠: alpha对seed敏感")
    print(f"  判定: {verdict}", flush=True)
    print("-> outputs/multiseed_robust.csv", flush=True)


if __name__ == "__main__":
    main()
