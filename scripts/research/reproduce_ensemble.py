"""End-to-end reproduction: TRAIN=500 ret_weighted walk-forward, saving the
PER-SEED predictions of the internal bagging ensemble (RANDOM_STATE+0..2) in ONE
training pass, so rank-combine vs mean-combine can be A/B'd offline at 1x cost.

This validates the production change (config.SEED_COMBINE=rank) on the exact code
path production uses (internal 3-seed bagging), not the 42/123/7 robustness runs
that ensemble_opt.py used. Replicates _train_predict_ret_weighted faithfully.

Output: data_cache/repro_perseed_oos.parquet (date,symbol,responder_6,return_1d,
pred_0,pred_1,pred_2) + prints L0 alpha/NW-t for mean vs rank combine.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import _cross_sectional_norm

N_SEEDS = 3
OUT = "data_cache/repro_perseed_oos.parquet"


def _perseed_ret_weighted(train, test, feat_cols):
    """Mirror _train_predict_ret_weighted exactly, but RETURN the per-seed
    predictions (list of n_seeds arrays) instead of combining them."""
    Xtr, _ = _cross_sectional_norm(train, feat_cols)
    ytr = train[C.TARGET].to_numpy(dtype=np.float64)
    Xte, _ = _cross_sectional_norm(test, feat_cols)
    rnk = (train.groupby("date", group_keys=False)[C.TARGET].rank(pct=True)).to_numpy()
    w = 0.2 + np.nan_to_num(rnk, nan=0.5)
    if C.WEIGHT in train:
        w = w * train[C.WEIGHT].to_numpy()
    out = []
    for s in range(N_SEEDS):
        model = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.6, min_child_samples=40,
            reg_alpha=0.1, reg_lambda=0.5, objective="l2",
            random_state=C.RANDOM_STATE + s, n_jobs=-1, verbose=-1)
        model.fit(Xtr, ytr, sample_weight=w)
        out.append(np.asarray(model.predict(Xte), dtype=np.float64).ravel())
    return out

def build_perseed_oos() -> pd.DataFrame:
    """Walk-forward over the cached panel at TRAIN=500, saving per-seed preds."""
    panel = load_factor_panel(keep_price=True)
    feat = list_factor_columns(panel)
    if C.WEIGHT not in panel.columns:
        panel = panel.copy(); panel[C.WEIGHT] = 1.0
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    n_dates = len(dates)
    train_w, test_w = C.TRAIN_WINDOW, C.TEST_WINDOW
    step, embargo = C.WF_STEP, C.EMBARGO_DAYS
    print(f"TRAIN={train_w} TEST={test_w} embargo={embargo} | {len(feat)} 因子 "
          f"| {n_dates} 交易日 | N_SEEDS={N_SEEDS}", flush=True)

    start, win = train_w + embargo, 0
    frames = []
    while start + test_w <= n_dates:
        tr = dates[start - embargo - train_w:start - embargo]
        te = dates[start:start + test_w]
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()
        t0 = time.time()
        per = _perseed_ret_weighted(train, test, feat)
        f = test[["date", "symbol", C.TARGET, "return_1d"]].copy()
        for s in range(N_SEEDS):
            f[f"pred_{s}"] = per[s]
        frames.append(f)
        print(f"  窗口{win}: {pd.Timestamp(te[0]).date()}~{pd.Timestamp(te[-1]).date()} "
              f"| {len(test)} 行 | {time.time()-t0:.0f}s", flush=True)
        start += step; win += 1
    oos = pd.concat(frames, ignore_index=True)
    oos.to_parquet(OUT)
    print(f"-> {OUT}  ({win} 窗口, {len(oos):,} 行)", flush=True)
    return oos


def _combine(oos, how):
    """date-aligned combine of pred_0..N into one score column."""
    cols = [f"pred_{s}" for s in range(N_SEEDS)]
    if how == "mean":
        return oos[cols].mean(axis=1).to_numpy()
    rank_sum = np.zeros(len(oos))
    for c in cols:
        tmp = oos[["date", c]].copy()
        rank_sum += tmp.groupby("date")[c].rank(pct=True, method="average").to_numpy()
    return rank_sum / len(cols)


def measure(oos, how, topk=0.15, rebal=20):
    """Periodic long_topk backtest + NW-t alpha vs equal-weight, like realistic_cost L0."""
    from measure_alpha import _newey_west_alpha_t
    df = oos.copy()
    df["pred"] = _combine(df, how)
    df["ret"] = df["return_1d"].fillna(0.0)
    rp = df.groupby("date")["pred"].rank(pct=True, method="first")
    df["w"] = np.where(rp > 1.0 - topk, rp, 0.0)
    tot = df.groupby("date")["w"].transform("sum")
    df["target"] = np.where(tot > 1e-12, df["w"] / tot, 0.0)

    dates = np.sort(df["date"].unique())
    rebal_set = set(dates[::rebal].tolist())
    by_date = {d: g for d, g in df.groupby("date")}
    held, net = {}, []
    COMM, STAMP = 0.0002, 0.0010
    for d in dates:
        g = by_date[d]
        sym = g["symbol"].to_numpy(); ret = np.nan_to_num(g["ret"].to_numpy(), nan=0.0)
        gross = sum(held.get(s, 0.0) * r for s, r in zip(sym, ret))
        cost = 0.0
        if d in rebal_set:
            tgt = dict(zip(sym, g["target"].to_numpy()))
            new = dict(held)
            for s in set(tgt) | set(held):
                dw = tgt.get(s, 0.0) - held.get(s, 0.0)
                if abs(dw) < 1e-9:
                    continue
                new[s] = tgt.get(s, 0.0)
                cost += max(dw, 0.0) * COMM + max(-dw, 0.0) * (COMM + STAMP)
            held = {s: w for s, w in new.items() if abs(w) > 1e-9}
        net.append(gross - cost)
    net = np.array(net)
    bench = df.groupby("date")["ret"].mean().reindex(dates).fillna(0.0).to_numpy()
    alpha_d, beta, nw_t, _ = _newey_west_alpha_t(net, bench, rebal)
    cum = float(np.prod(1.0 + net) - 1.0)
    sharpe = net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std() > 1e-12 else 0.0
    eq = np.cumprod(1.0 + net); dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    # cross-section IC
    ic = df.groupby("date").apply(
        lambda x: x["pred"].corr(x[C.TARGET], method="spearman")).mean()
    return {"how": how, "alpha_ann": alpha_d * 252, "nw_t": nw_t, "beta": beta,
            "sharpe": sharpe, "cum": cum, "maxdd": dd, "ic": float(ic)}


def main():
    if os.path.exists(OUT) and os.environ.get("REUSE", "1") == "1":
        print(f"复用 {OUT}", flush=True)
        oos = pd.read_parquet(OUT)
    else:
        oos = build_perseed_oos()
    print(f"\nOOS: {oos['date'].nunique()} 日 · {len(oos):,} 行\n")
    rows = []
    for how in ("mean", "rank"):
        m = measure(oos, how)
        rows.append(m)
        print(f"{how:5s} | α年化={m['alpha_ann']*100:+6.2f}% NW-t={m['nw_t']:+.2f} "
              f"Sharpe={m['sharpe']:+.2f} 累计={m['cum']*100:+6.1f}% "
              f"MaxDD={m['maxdd']*100:.1f}% IC={m['ic']:+.4f} beta={m['beta']:.2f}")
    pd.DataFrame(rows).to_csv("outputs/reproduce_ensemble.csv", index=False)
    print("-> outputs/reproduce_ensemble.csv")


if __name__ == "__main__":
    main()
