"""滚动走动回测（walk-forward）—— 严格防泄露的样本外验证。

针对单次划分回测 Sharpe 虚高（窗口运气 + 因子选择偷看测试集）的问题，本引擎：

  1. 滚动训练：每步用前 TRAIN_WINDOW 天训练，预测其后 TEST_WINDOW 天（样本外）
  2. embargo：训练段与测试段之间剔除 EMBARGO_DAYS 天，杜绝 N 日标签跨界泄露
  3. 因子选择隔离：每个窗口的因子选择只用该窗口的训练段（select_passing_factors）
  4. 拼接所有窗口的样本外预测 → 一条长达 1~2 年的连续 OOS 净值曲线

直接在「命名因子面板」(load_factor_panel) 上操作，模型用命名因子列训练，
避免 feature_NN 映射，保证窗口内因子选择与训练用的是同一套列。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
from factor_eval import select_passing_factors
from strategy import generate_positions
from backtest import run_backtest, run_backtest_periodic, format_metrics


def _run_bt(df, pred, pos, pred_in_return_space: bool):
    """Dispatch to the periodic-rebalance engine (default) or the legacy daily-
    overlap one. The periodic engine removes the N-day-target double-count + mis-
    annualization that inflates Sharpe/PnL (see config.PERIODIC_REBALANCE)."""
    if getattr(C, "PERIODIC_REBALANCE", True):
        return run_backtest_periodic(
            df, pred, pos, rebalance_days=getattr(C, "REBALANCE_DAYS", None),
            pred_in_return_space=pred_in_return_space)
    return run_backtest(df, pred, pos, pred_in_return_space=pred_in_return_space)


def _is_return_space() -> bool:
    """l2 and ret_weighted are genuine return-space regressors (price-RMSE /
    weighted-R2 are meaningful). lambdarank / ic_linear / fusion output ranking
    scores, for which only rank-based IC/ICIR is valid. Rank-combining the seeds
    (SEED_COMBINE=rank) also moves the output out of return space, so RMSE is
    only meaningful for the return-space objectives under mean-combine."""
    return (getattr(C, "MODEL_OBJECTIVE", "l2") in ("l2", "ret_weighted")
            and getattr(C, "SEED_COMBINE", "rank") == "mean")


def _combine_seeds(preds: list[np.ndarray], dates: np.ndarray) -> np.ndarray:
    """Merge n_seeds per-model predictions into one score.

    SEED_COMBINE="mean" averages raw predictions (legacy, stays in return space).
    "rank" averages each seed's PER-DATE percentile rank — scale-robust and the
    better tradeable combine when the seeds are highly correlated (validated on
    the cached whole-market OOS, ensemble_opt.py). Returns a 1-D array aligned to
    the input rows (i.e. to `dates`)."""
    stacked = np.vstack(preds)                      # (n_seeds, n_rows)
    if getattr(C, "SEED_COMBINE", "rank") != "rank":
        return stacked.mean(axis=0)
    # per-date percentile rank each seed, then average the ranks
    df = pd.DataFrame({"date": dates})
    rank_sum = np.zeros(stacked.shape[1], dtype=np.float64)
    for row in stacked:
        df["_p"] = row
        rank_sum += df.groupby("date")["_p"].rank(pct=True, method="average").to_numpy()
    return rank_sum / stacked.shape[0]


def _regime_scale(panel: pd.DataFrame, test_dates, factors: list[str],
                  lookback: int = 10) -> tuple[float, float]:
    """Trailing IC of value factors before the test window (10-day lookback).
    Returns (scale, avg_ic): scale=0.3 if avg_ic<0, else 1.0.
    """
    from scipy import stats
    te_min = test_dates[0]
    pre = [d for d in sorted(panel["date"].unique()) if d < te_min][-lookback:]
    if len(pre) < lookback // 3:
        return 1.0, np.nan
    sub = panel[panel["date"].isin(pre)]
    ics = []
    for f in factors:
        if f not in sub.columns:
            continue
        day_ic = sub.groupby("date", group_keys=False).apply(
            lambda g, fac=f: (
                stats.spearmanr(g[fac].dropna(),
                                g.loc[g[fac].notna(), C.TARGET]).correlation
                if g[fac].notna().sum() >= 20 else np.nan),
            include_groups=False)
        ics.extend(day_ic.dropna().tolist())
    avg_ic = float(np.mean(ics)) if ics else np.nan
    scale = 1.0 if (np.isnan(avg_ic) or avg_ic >= 0.0) else 0.3
    return scale, avg_ic


def _cross_sectional_norm(df: pd.DataFrame, feat_cols: list[str],
                          stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Per-date cross-sectional MAD-winsorize + z-score of each factor.

    Two-step denoising (leak-free; each date uses only its own cross-section):
      1. MAD-winsorize: clip each factor to median ± k·MAD (k=C.WINSOR_MAD_K, the
         robust analogue of mean ± kσ). A single extreme outlier inflates the std
         used by the z-score, so winsorizing the RAW values FIRST — before
         standardizing — is the documented best practice; clipping z afterward
         (as the old code did) lets the outlier corrupt the scale first.
         1.4826·MAD ≈ σ for normal data, so k≈3 keeps ~99.7% untouched.
      2. z-score: (x − mean) / std on the winsorized values; final clip ±4 as a
         belt-and-suspenders guard. NaN → 0 = the cross-sectional mean (neutral).
    Standardizing within each day puts every factor on the same regime-free scale
    so the tree ranks stocks, not absolute levels. `stats` unused (each frame
    normalizes against itself, never the train segment).
    """
    out = np.empty((len(df), len(feat_cols)), dtype=np.float32)
    dates = df["date"].to_numpy()
    order = np.argsort(dates, kind="stable")
    k = getattr(C, "WINSOR_MAD_K", 3.0)
    do_winsor = getattr(C, "USE_WINSORIZE", True)
    grp = df["date"].to_numpy()
    for j, col in enumerate(feat_cols):
        v = df[col].to_numpy(dtype=np.float64)
        s = pd.Series(v, index=grp)
        if do_winsor and k > 0:
            # robust winsorize: per-date median ± k·1.4826·MAD
            med = s.groupby(level=0).transform("median")
            absdev = (s - med).abs()
            mad = absdev.groupby(level=0).transform("median")
            scale = 1.4826 * mad.to_numpy()
            lo = med.to_numpy() - k * scale
            hi = med.to_numpy() + k * scale
            vv = np.clip(s.to_numpy(), lo, hi)
            s = pd.Series(vv, index=grp)
        mu = s.groupby(level=0).transform("mean")
        sd = s.groupby(level=0).transform("std")
        z = (s.to_numpy() - mu.to_numpy()) / (sd.to_numpy() + 1e-9)
        z = np.clip(z, -4.0, 4.0)
        out[:, j] = np.nan_to_num(z, nan=0.0)
    _ = order, stats  # explicitly unused
    return out, {}


def _train_predict(train: pd.DataFrame, test: pd.DataFrame,
                   feat_cols: list[str], n_seeds: int = None) -> np.ndarray:
    """Train a multi-seed LGB ensemble on `train`, predict `test` out of sample.

    Objective is C.MODEL_OBJECTIVE:
      "l2"         : regression on responder_6 (prediction in return space).
      "lambdarank" : learning-to-rank — optimizes the within-date ORDERING that
                     long_topk actually consumes. Output is a ranking score.

    Improvements applied to both:
      1. cross-sectional z-score of every factor (per date, leak-free) so the
         model ranks stocks within a day instead of chasing regime-shifted
         absolute levels;
      2. NaN -> cross-sectional mean (0 after z-score), a neutral fill;
      3. multi-seed bagging (average of n_seeds models) to cut variance on a
         very noisy target;
      4. (l2) early stopping on the tail of the training segment, embargoed so
         the validation labels don't peek into the held-out training tail;
      5. stronger regularization (smaller leaves/lr, larger min_child_samples,
         L1+L2) to resist over-fitting a very-low-signal target.
    """
    objective = getattr(C, "MODEL_OBJECTIVE", "l2")
    if n_seeds is None:                       # env override for speed (e.g. relative search)
        n_seeds = int(os.environ.get("N_SEEDS", "3"))
    if objective == "lambdarank":
        return _train_predict_rank(train, test, feat_cols, n_seeds)
    if objective == "ret_weighted":
        return _train_predict_ret_weighted(train, test, feat_cols, n_seeds)
    if objective == "ic_linear":
        return _train_predict_linear(train, test, feat_cols)
    if objective == "fusion":
        return _train_predict_fusion(train, test, feat_cols, n_seeds)
    if objective == "multi_ret_weighted":
        return _train_predict_multi_ret_weighted(train, test, feat_cols, n_seeds)
    if objective == "multi_fusion":
        return _train_predict_multi_fusion(train, test, feat_cols, n_seeds)
    if objective == "attn":
        # 截面自注意力（GPU）。独立模块，仅在选用时 import torch。
        from model_attn import attn_predict
        return attn_predict(train, test, feat_cols)

    # default l2 regression leg (factored into _gbm_l2_predict so the fusion
    # objective reuses the exact same model without duplicating its config)
    return _gbm_l2_predict(train, test, feat_cols, n_seeds)


def _train_predict_rank(train: pd.DataFrame, test: pd.DataFrame,
                        feat_cols: list[str], n_seeds: int = 3,
                        n_grades: int = 16) -> np.ndarray:
    """LambdaRank variant: per-date groups, target binned into integer grades.

    LGBMRanker maximizes NDCG over each date's cross-section, so it learns the
    ordering that long_topk consumes directly. The forward excess return is
    discretized into `n_grades` per-date quantile buckets (relevance labels).
    Predictions are returned in the ORIGINAL `test` row order.
    """
    # sort train by date so each date's rows form one contiguous ranking group
    tr_order = np.argsort(train["date"].to_numpy(), kind="stable")
    tr_sorted = train.iloc[tr_order]
    Xtr_all, _ = _cross_sectional_norm(train, feat_cols)
    Xtr = Xtr_all[tr_order]
    y_raw = tr_sorted[C.TARGET].to_numpy(dtype=np.float64)

    # per-date quantile bucket -> integer relevance grade in [0, n_grades-1]
    grades = (tr_sorted.groupby("date", group_keys=False)[C.TARGET]
              .apply(lambda s: pd.qcut(s.rank(method="first"),
                                       min(n_grades, max(s.notna().sum(), 1)),
                                       labels=False, duplicates="drop"))
              .to_numpy())
    grades = np.nan_to_num(grades, nan=0.0).astype(int)
    _, group_tr = np.unique(tr_sorted["date"].to_numpy(), return_counts=True)

    # test features in original order (predictions don't need group structure)
    Xte, _ = _cross_sectional_norm(test, feat_cols)
    _ = y_raw

    preds = []
    for s in range(n_seeds):
        ranker = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=500, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.6,
            min_child_samples=40, reg_alpha=0.1, reg_lambda=0.5,
            label_gain=list(range(n_grades + 1)),
            random_state=C.RANDOM_STATE + s, n_jobs=-1, verbose=-1)
        ranker.fit(Xtr, grades, group=group_tr)
        preds.append(np.asarray(ranker.predict(Xte), dtype=np.float64).ravel())
    return _combine_seeds(preds, test["date"].to_numpy())


def _training_ic_weights(train: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Per-factor weight = mean daily cross-sectional RankIC over the TRAIN segment.

    Leak-free: uses only the training segment's own forward returns. The sign is
    kept, so a reversal factor (negative IC) automatically gets a negative weight
    and is subtracted in the blend. Factors with <2 valid IC days get weight 0.
    """
    from scipy import stats
    weights = {}
    for f in feat_cols:
        if f not in train.columns:
            weights[f] = 0.0
            continue
        day_ic = train.groupby("date", group_keys=False).apply(
            lambda g, fac=f: (
                stats.spearmanr(g[fac], g[C.TARGET]).correlation
                if (g[fac].notna() & g[C.TARGET].notna()).sum() >= 5
                and g[fac].nunique() >= 2 else np.nan),
            include_groups=False)
        day_ic = pd.Series(day_ic).dropna()
        weights[f] = float(day_ic.mean()) if len(day_ic) >= 2 else 0.0
    return weights


def _train_predict_linear(train: pd.DataFrame, test: pd.DataFrame,
                          feat_cols: list[str]) -> np.ndarray:
    """IC-weighted linear blend of factor z-scores (ranking score).

    By default (C.LINEAR_FACTORS empty) the blend uses every factor in `feat_cols`,
    so the unified factor library is fully consumed. If C.LINEAR_FACTORS is set to
    a non-empty list, only that curated subset is used.

    Weights come from the training segment only (leak-free); each factor's test
    z-score is per-date (leak-free). Output is a ranking score, not a return.
    """
    linear_feats = [f for f in getattr(C, "LINEAR_FACTORS", feat_cols)
                    if f in test.columns and f in train.columns]
    if not linear_feats:
        linear_feats = [f for f in feat_cols if f in test.columns]

    # --- Per-window factor transforms (LEAK-FREE: uses TRAINING segment only) ---
    nl_topk = getattr(C, "NL_TOPK", 0)
    fi_topk = getattr(C, "FI_TOPK", 0)
    top_by_ic = []
    if (nl_topk > 0 or fi_topk > 0) and linear_feats:
        from data.factor_transforms import _rank_factors_by_ic
        rankings = _rank_factors_by_ic(train, linear_feats, C.TARGET)
        top_by_ic = [f for f, _, _ in rankings[:max(nl_topk, fi_topk)]]

    # Non-linear transforms: z², |z|, pos/neg tails for top-N factors by training IC
    if nl_topk > 0 and top_by_ic:
        for f in top_by_ic[:nl_topk]:
            for df_side in [train, test]:
                grp = df_side.groupby("date", group_keys=False)[f]
                mu = grp.transform("mean")
                sd = grp.transform("std").replace(0, 1e-9)
                z = ((df_side[f] - mu) / sd).clip(-4.0, 4.0)
                for suffix, tfn in [("_sq", lambda x: x**2), ("_abs", np.abs),
                                    ("_pos", lambda x: np.clip(x, 0, None)),
                                    ("_neg", lambda x: np.clip(-x, 0, None))]:
                    col = f"nl_{f}{suffix}"
                    df_side[col] = tfn(z)
                    if col not in linear_feats:
                        linear_feats.append(col)

    # Factor interactions: pairwise products of top-N factor z-scores
    fi_max = getattr(C, "FI_MAX_PAIRS", 60)
    if fi_topk > 0 and len(top_by_ic) >= 2:
        inter_factors = top_by_ic[:fi_topk]
        added = 0
        for i, fa in enumerate(inter_factors):
            for fb in inter_factors[i + 1:]:
                if added >= fi_max: break
                col = f"ix_{fa}__x__{fb}"
                for df_side in [train, test]:
                    za = ((df_side[fa] - df_side.groupby("date")[fa].transform("mean"))
                          / df_side.groupby("date")[fa].transform("std").replace(0, 1e-9))
                    zb = ((df_side[fb] - df_side.groupby("date")[fb].transform("mean"))
                          / df_side.groupby("date")[fb].transform("std").replace(0, 1e-9))
                    df_side[col] = (za.clip(-4, 4) * zb.clip(-4, 4)).astype(np.float32)
                if col not in linear_feats:
                    linear_feats.append(col)
                added += 1
            if added >= fi_max: break

    # Size-neutralization
    if getattr(C, "LINEAR_SIZE_NEUTRAL", True):
        train = _size_neutralize(train, linear_feats)
        test = _size_neutralize(test, linear_feats)

    # Factor stability filter
    min_stab = getattr(C, "MIN_FACTOR_STABILITY", 0.0)
    if min_stab > 0:
        from data.factor_transforms import filter_stable_factors
        stable = set(filter_stable_factors(train, linear_feats,
                                           min_stability=min_stab))
        linear_feats = [f for f in linear_feats if f in stable]

    w = _training_ic_weights(train, linear_feats)
    Zte, _ = _cross_sectional_norm(test, linear_feats)
    score = np.zeros(len(test), dtype=np.float64)
    for j, f in enumerate(linear_feats):
        score += w[f] * np.nan_to_num(Zte[:, j], nan=0.0)
    return score


def _size_neutralize(df: pd.DataFrame, factors: list[str],
                     size_col: str = "ln_mv") -> pd.DataFrame:
    """Replace each factor with its per-date residual after regressing on ln_mv.

    Leak-free: each trading day is regressed against only that day's own
    cross-section. size_col absent (or a factor missing) → that factor is left
    untouched. Used by the ic_linear / size-neutral predictor.
    """
    if size_col not in df.columns:
        return df
    out = df.copy()
    for f in factors:
        if f == size_col or f not in out.columns:
            continue

        def _resid(g, fac=f):
            x = g[size_col].to_numpy(dtype=np.float64)
            y = g[fac].to_numpy(dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            r = np.full(len(g), np.nan)
            if mask.sum() >= 10 and np.nanstd(x[mask]) > 1e-9:
                b = np.polyfit(x[mask], y[mask], 1)
                r[mask] = y[mask] - (b[0] * x[mask] + b[1])
            return pd.Series(r, index=g.index)

        out[f] = out.groupby("date", group_keys=False).apply(
            _resid, include_groups=False)
    return out


def _train_predict_fusion(train: pd.DataFrame, test: pd.DataFrame,
                          feat_cols: list[str], n_seeds: int = 3) -> np.ndarray:
    """Rank-average of the l2 GBM and the ic_linear blend (ranking score).

    Each model's prediction is converted to a per-date percentile rank, then the
    two ranks are averaged. De-correlates the GBM's and the linear model's errors
    (the GBM captures non-linearities, the linear blend resists noise over-fit),
    which historically lifts ICIR above either model alone.
    """
    # GBM leg: reuse the l2 path by temporarily forcing the objective off the
    # dispatch (call the regressor body directly via a small replica).
    gbm = _gbm_l2_predict(train, test, feat_cols, n_seeds)
    lin = _train_predict_linear(train, test, feat_cols)

    dates = test["date"].to_numpy()
    rg = _per_date_rank(gbm, dates)
    rl = _per_date_rank(lin, dates)
    return (rg + rl) / 2.0


def _with_target_col(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Return a copy of df with C.TARGET renamed from target_col.

    Allows reusing _train_predict_* helpers (which read C.TARGET) for multi-
    horizon targets without modifying those functions.
    """
    cols = [c for c in df.columns if c not in (C.TARGET, target_col)]
    out = df[cols + [target_col]].copy()
    out = out.rename(columns={target_col: C.TARGET})
    return out


def _train_predict_multi_ret_weighted(train: pd.DataFrame, test: pd.DataFrame,
                                      feat_cols: list[str], n_seeds: int = 3) -> np.ndarray:
    """Multi-horizon ret_weighted fusion.

    Trains one ret_weighted GBM per horizon (primary FORWARD_WINDOW plus
    MULTI_HORIZON_WINDOWS), converts each horizon's predictions to per-date
    percentile ranks, then fuses by MULTI_HORIZON_WEIGHTS. Output is a ranking
    score that blends short/medium/long-term signals.
    """
    horizons = [int(getattr(C, "FORWARD_WINDOW", 20))] + \
        [int(x) for x in getattr(C, "MULTI_HORIZON_WINDOWS", [])]
    weights = [float(w) for w in getattr(C, "MULTI_HORIZON_WEIGHTS", [])]
    if len(weights) != len(horizons):
        weights = [1.0 / len(horizons)] * len(horizons)
    tot = sum(weights)
    weights = [w / tot for w in weights]

    dates = test["date"].to_numpy()
    score = np.zeros(len(test), dtype=np.float64)
    for fw, w in zip(horizons, weights):
        col = C.TARGET if fw == getattr(C, "FORWARD_WINDOW", 20) else f"responder_fw_{fw}"
        if col not in train.columns or col not in test.columns:
            continue
        tr = _with_target_col(train, col).dropna(subset=[C.TARGET])
        te = _with_target_col(test, col)
        pred = _train_predict_ret_weighted(tr, te, feat_cols, n_seeds)
        score += w * _per_date_rank(pred, dates)
    return score


def _train_predict_multi_fusion(train: pd.DataFrame, test: pd.DataFrame,
                                feat_cols: list[str], n_seeds: int = 3) -> np.ndarray:
    """Multi-horizon fusion of GBM + ic_linear per horizon.

    For each horizon, compute rank-average of the l2 GBM and ic_linear blend,
    then rank-average across horizons weighted by MULTI_HORIZON_WEIGHTS.
    Maximizes both model diversity and horizon diversity.
    """
    horizons = [int(getattr(C, "FORWARD_WINDOW", 20))] + \
        [int(x) for x in getattr(C, "MULTI_HORIZON_WINDOWS", [])]
    weights = [float(w) for w in getattr(C, "MULTI_HORIZON_WEIGHTS", [])]
    if len(weights) != len(horizons):
        weights = [1.0 / len(horizons)] * len(horizons)
    tot = sum(weights)
    weights = [w / tot for w in weights]

    dates = test["date"].to_numpy()
    score = np.zeros(len(test), dtype=np.float64)
    for fw, w in zip(horizons, weights):
        col = C.TARGET if fw == getattr(C, "FORWARD_WINDOW", 20) else f"responder_fw_{fw}"
        if col not in train.columns or col not in test.columns:
            continue
        tr = _with_target_col(train, col).dropna(subset=[C.TARGET])
        te = _with_target_col(test, col)
        gbm = _gbm_l2_predict(tr, te, feat_cols, n_seeds)
        lin = _train_predict_linear(tr, te, feat_cols)
        horizon_score = (_per_date_rank(gbm, dates) + _per_date_rank(lin, dates)) / 2.0
        score += w * horizon_score
    return score


def _per_date_rank(pred: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Per-date percentile rank in [0, 1] (ties averaged). Leak-free (within date)."""
    s = pd.Series(np.asarray(pred, dtype=np.float64))
    grp = pd.Series(dates)
    return s.groupby(grp).rank(pct=True).to_numpy()


def _gbm_l2_predict(train: pd.DataFrame, test: pd.DataFrame,
                    feat_cols: list[str], n_seeds: int = 3) -> np.ndarray:
    """The l2 regression leg of _train_predict, factored out for reuse by fusion."""
    Xtr, _ = _cross_sectional_norm(train, feat_cols)
    ytr = train[C.TARGET].to_numpy(dtype=np.float64)
    wtr = train[C.WEIGHT].to_numpy() if C.WEIGHT in train else None
    Xte, _ = _cross_sectional_norm(test, feat_cols)

    tr_dates = np.sort(train["date"].unique())
    n_val = max(int(len(tr_dates) * 0.15), 1)
    if len(tr_dates) > n_val + C.EMBARGO_DAYS + 5:
        val_dates = tr_dates[-n_val:]
        fit_dates = tr_dates[: -(n_val + C.EMBARGO_DAYS)]
        fit_mask = train["date"].isin(fit_dates).to_numpy()
        val_mask = train["date"].isin(val_dates).to_numpy()
    else:
        fit_mask = np.ones(len(train), dtype=bool)
        val_mask = None

    preds = []
    for s in range(n_seeds):
        model = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.6, min_child_samples=40,
            reg_alpha=0.1, reg_lambda=0.5, objective="l2",
            random_state=C.RANDOM_STATE + s, n_jobs=-1, verbose=-1)
        if val_mask is not None:
            model.fit(
                Xtr[fit_mask], ytr[fit_mask],
                sample_weight=(wtr[fit_mask] if wtr is not None else None),
                eval_set=[(Xtr[val_mask], ytr[val_mask])],
                eval_sample_weight=([wtr[val_mask]] if wtr is not None else None),
                callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            model.fit(Xtr, ytr, sample_weight=wtr)
        preds.append(np.asarray(model.predict(Xte), dtype=np.float64).ravel())
    return _combine_seeds(preds, test["date"].to_numpy())


def _train_predict_ret_weighted(train: pd.DataFrame, test: pd.DataFrame,
                                feat_cols: list[str], n_seeds: int = 3) -> np.ndarray:
    """l2 GBM whose sample weights are proportional to the per-date rank of the
    forward return — concentrates the fit on "who reaches the top", aligning the
    predictor with what long_topk actually harvests (the top-20% names).

    Phase-1 retargeted search (2026-06) — primary metric = top-20% OOS excess
    return, the part a long-only book can capture — found this the best predictor:
    top-20% excess +0.0035 / stability ICIR +1.00 vs the l2 GBM baseline's
    +0.0020 / +0.31. The decile diagnostic had shown full-cross-section IC is the
    wrong target (the highest-IC ic_linear has its edge in the BOTTOM decile,
    untradeable long-only); weighting the regression toward high-forward-return
    rows moves the edge into the head. Output lives in return space (it is still
    an l2 regressor), so price-RMSE / weighted-R2 stay meaningful.

    Weight = 0.2 + per-date percentile rank of responder_6 (≈0.2 at the bottom,
    ≈1.2 at the top, ~6x tilt). The rank uses only the training segment's own
    forward returns, so it is leak-free.
    """
    Xtr, _ = _cross_sectional_norm(train, feat_cols)
    ytr = train[C.TARGET].to_numpy(dtype=np.float64)
    Xte, _ = _cross_sectional_norm(test, feat_cols)

    rnk = (train.groupby("date", group_keys=False)[C.TARGET]
                .rank(pct=True)).to_numpy()
    w = 0.2 + np.nan_to_num(rnk, nan=0.5)        # head ~6x the tail
    # fold in the existing inverse-vol/recency weight if present
    if C.WEIGHT in train:
        w = w * train[C.WEIGHT].to_numpy()

    preds = []
    for s in range(n_seeds):
        model = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.6, min_child_samples=40,
            reg_alpha=0.1, reg_lambda=0.5, objective="l2",
            random_state=C.RANDOM_STATE + s, n_jobs=-1, verbose=-1)
        model.fit(Xtr, ytr, sample_weight=w)
        preds.append(np.asarray(model.predict(Xte), dtype=np.float64).ravel())
    return _combine_seeds(preds, test["date"].to_numpy())


def run_walkforward(panel: pd.DataFrame, all_factor_cols: list[str],
                    select_features: bool = False,
                    strategy: str = "long_topk",
                    verbose: bool = True) -> dict:
    """滚动回测。返回拼接后的 OOS 结果与各窗口指标分布。

    panel: 命名因子面板（含 date, symbol, 因子..., responder_6, weight）
    all_factor_cols: 全部候选因子名
    select_features: True 则每窗口在训练段内做因子准入筛选（隔离，无泄露）
    """
    # weight 列保障（load_factor_panel 默认不含 weight，这里补等权）
    if C.WEIGHT not in panel.columns:
        panel = panel.copy()
        panel[C.WEIGHT] = 1.0
    # 整数日期序号，便于按窗口切分
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    n_dates = len(dates)

    train_w, test_w = C.TRAIN_WINDOW, C.TEST_WINDOW
    step, embargo = C.WF_STEP, C.EMBARGO_DAYS

    oos_frames = []
    oos_pos_frames = []
    window_metrics = []

    start = train_w + embargo
    win_id = 0
    while start + test_w <= n_dates:
        tr_lo_i = start - embargo - train_w
        tr_hi_i = start - embargo                  # exclusive
        te_lo_i = start
        te_hi_i = start + test_w                   # exclusive

        tr_dates = dates[tr_lo_i:tr_hi_i]
        te_dates = dates[te_lo_i:te_hi_i]
        train = panel[panel["date"].isin(tr_dates)]
        test = panel[panel["date"].isin(te_dates)].copy()

        # 窗口内因子选择（仅用训练段日期）
        if select_features:
            feat = select_passing_factors(panel, date_lo=tr_dates[0],
                                          date_hi=tr_dates[-1])
            feat = [f for f in feat if f in all_factor_cols]
            if not feat:                            # 该窗口无因子达标 → 退回全特征
                feat = all_factor_cols
        else:
            feat = all_factor_cols

        # cross-sectional-normalized, multi-seed, early-stopped LGB ensemble
        pred = _train_predict(train, test, feat)

        test["prediction"] = pred

        # Method A: dynamic position scaling — reduce exposure when value IC negative
        value_factors = [f for f in ["bp", "ep", "value_composite"] if f in all_factor_cols]
        scale, avg_ic = _regime_scale(panel, te_dates, value_factors)

        pos = generate_positions(_with_ids(test), pred, strategy=strategy)
        in_ret = _is_return_space()
        res = _run_bt(_with_ids(test), pred, pos, in_ret)
        m = res["metrics"]
        window_metrics.append({
            "window": win_id,
            "train": f"{pd.Timestamp(tr_dates[0]).date()}~{pd.Timestamp(tr_dates[-1]).date()}",
            "test": f"{pd.Timestamp(te_dates[0]).date()}~{pd.Timestamp(te_dates[-1]).date()}",
            "n_feat": len(feat), "regime_ic": round(avg_ic, 4) if not np.isnan(avg_ic) else None,
            "pos_scale": scale,
            "PnL": m["total_pnl"], "Sharpe": m["sharpe"],
            "MaxDD": m["max_drawdown"], "hit": m["hit_rate"],
            "PriceRMSE": m.get("mean_price_rmse", float("nan")),
            "PredIC": m.get("pred_ic", float("nan")),
        })
        oos_frames.append(test)
        oos_pos_frames.append(pos)   # keep risk-adjusted positions
        if verbose:
            ic_str = (f"预测IC={m['pred_ic']:+.3f}"
                      if m.get("pred_ic") == m.get("pred_ic") else "")
            print(f"  窗口{win_id}: 训练{tr_dates[0].astype('datetime64[D]')}~"
                  f"{tr_dates[-1].astype('datetime64[D]')} → "
                  f"测试{te_dates[0].astype('datetime64[D]')}~"
                  f"{te_dates[-1].astype('datetime64[D]')} | "
                  f"{len(feat)}因子 | {ic_str} | OOS Sharpe={m['sharpe']:+.2f}")
        start += step
        win_id += 1

    if not oos_frames:
        raise ValueError("数据不足以构成任何滚动窗口，请减小 TRAIN_WINDOW/TEST_WINDOW")

    # 拼接所有窗口的连续 OOS 净值（使用已经过熔断/波动率调整的持仓）
    oos = pd.concat(oos_frames, ignore_index=True)
    oos_pred = oos["prediction"].to_numpy()
    oos_pos = np.concatenate(oos_pos_frames)   # risk-adjusted, not re-generated
    in_ret = _is_return_space()
    oos_res = _run_bt(_with_ids(oos), oos_pred, oos_pos, in_ret)

    return {
        "oos_result": oos_res,
        "window_metrics": pd.DataFrame(window_metrics),
        "n_windows": win_id,
        "oos": oos,
    }


def _apply_circuit_breaker(pos: np.ndarray, test: pd.DataFrame,
                            thresh1: float = -0.05, thresh2: float = -0.10,
                            scale1: float = 0.3, scale2: float = 0.0) -> np.ndarray:
    """Intra-window circuit breaker: reduce exposure when cumulative loss deepens.

    Tracks daily portfolio P&L using actual stock returns (pct_chg).
    Scale at day t is based solely on returns from test_start to day t-1 (no look-ahead).
      cumLoss < thresh1 → scale = scale1 (0.3)
      cumLoss < thresh2 → scale = scale2 (0.0, full exit)
    """
    if "pct_chg" not in test.columns:
        return pos
    dates = sorted(test["date"].unique())
    pos_out = pos.copy()
    running = 0.0
    for d in dates:
        mask = (test["date"] == d).to_numpy()
        # scale based on loss SO FAR (before today's return)
        scale = 1.0
        if running < thresh2:
            scale = scale2
        elif running < thresh1:
            scale = scale1
        pos_out[mask] *= scale
        # update with today's actual 1-day return (pct_chg is in %)
        daily_ret = (pos_out[mask] * test.loc[test["date"] == d, "pct_chg"].to_numpy() / 100.0).sum()
        running += daily_ret
    return pos_out


def _apply_vol_targeting(pos: np.ndarray, test: pd.DataFrame,
                         target_annual_vol: float = 0.15,
                         vol_window: int = 20) -> np.ndarray:
    """Volatility targeting: scale positions to hit a fixed annualised volatility.

    Uses realised daily portfolio vol (computed from pct_chg) over the past
    vol_window days to set position scale. Reduces exposure in turbulent periods.
    """
    if "pct_chg" not in test.columns:
        return pos
    target_daily = target_annual_vol / np.sqrt(252)
    dates = sorted(test["date"].unique())
    pos_out = pos.copy()
    daily_rets = []
    for d in dates:
        mask = (test["date"] == d).to_numpy()
        scale = 1.0
        if len(daily_rets) >= 5:
            vol = np.std(daily_rets[-vol_window:]) + 1e-9
            scale = min(1.0, target_daily / vol)
        pos_out[mask] *= scale
        dr = (pos_out[mask] * test.loc[test["date"] == d, "pct_chg"].to_numpy() / 100.0).sum()
        daily_rets.append(dr)
    return pos_out


def _with_ids(df: pd.DataFrame) -> pd.DataFrame:
    """给命名因子面板补 date_id/time_id/symbol_id（策略与回测按这些列分组）。"""
    out = df.copy()
    out[C.DATE_COL] = out["date"].rank(method="dense").astype(int)
    out[C.TIME_COL] = 0
    out[C.SYMBOL_COL] = out["symbol"].astype("category").cat.codes
    return out


def plot_dashboard(result: dict, out_path: str, tag: str = "",
                   initial_capital: float = None) -> str:
    """Combine all backtest views into one multi-panel dashboard figure.

    Panels:
      A. Account value in money terms (¥), compounding daily net returns
      B. Cumulative return (%) — same shape as A but unit-free
      C. Per-window out-of-sample Sharpe (bar chart) — robustness across time
      D. Drawdown in money terms (¥) — peak-to-trough loss
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"

    cap0 = initial_capital if initial_capital is not None else C.INITIAL_CAPITAL
    curve = result["oos_result"]["curve"]
    m = result["oos_result"]["metrics"]
    wm = result["window_metrics"]
    rmse_curve = result["oos_result"].get("rmse_curve")

    net = curve["net"].to_numpy()
    equity_ret = np.cumprod(1.0 + net)
    account = cap0 * equity_ret
    money_pnl = account - cap0
    cum_ret_pct = (equity_ret - 1.0) * 100.0
    peak = np.maximum.accumulate(account)
    dd_money = account - peak
    x = np.arange(len(account))

    # Benchmark: equal-weight buy-and-hold of the full universe.
    # Use return_1d (daily close-to-close pct change, decimal form) from the OOS panel.
    oos = result.get("oos")
    ret_col = "return_1d" if (oos is not None and "return_1d" in oos.columns) else None
    bench_account = bench_cum_pct = alpha_pct = None
    if ret_col and oos is not None:
        bench_daily = (oos.groupby("date")[ret_col].mean()
                          .sort_index().to_numpy())
        if len(bench_daily) == len(net):
            bench_ret = np.cumprod(1.0 + bench_daily)
            bench_account = cap0 * bench_ret
            bench_cum_pct = (bench_ret - 1.0) * 100.0
            alpha_pct = cum_ret_pct - bench_cum_pct  # excess return vs benchmark

    final_pnl = money_pnl[-1] if len(money_pnl) else 0.0
    total_ret_pct = cum_ret_pct[-1] if len(cum_ret_pct) else 0.0
    max_dd_money = dd_money.min() if len(dd_money) else 0.0
    bench_end = (f"  |  Benchmark {bench_cum_pct[-1]:+.1f}%"
                 f"  Alpha {alpha_pct[-1]:+.1f}%"
                 if bench_cum_pct is not None else "")

    fig = plt.figure(figsize=(15, 13.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 1.3, 1.3, 1.3],
                          hspace=0.45, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])
    ax_e = fig.add_subplot(gs[3, :])

    title = (f"Walk-forward Out-of-Sample Dashboard{(' — ' + tag) if tag else ''}\n"
             f"Start ¥{cap0:,.0f}  ->  End ¥{account[-1]:,.0f}   "
             f"(Net P/L ¥{final_pnl:+,.0f}, {total_ret_pct:+.1f}%){bench_end}   |   "
             f"Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  "
             f"Calmar {m['calmar']:.2f}  Hit {m['hit_rate']:.1%}")
    fig.suptitle(title, fontsize=11, y=0.99)
    _dashboard_panels(ax_a, ax_b, ax_c, ax_d, x, account, money_pnl,
                      cum_ret_pct, dd_money, cap0, wm, final_pnl, max_dd_money,
                      bench_account=bench_account, alpha_pct=alpha_pct,
                      bench_cum_pct=bench_cum_pct, ax_e=ax_e, rmse_curve=rmse_curve)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _money_fmt(v: float) -> str:
    """Format a yuan amount compactly (e.g. 1.2M, -340K)."""
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e6:
        return f"{s}¥{a/1e6:.2f}M"
    if a >= 1e3:
        return f"{s}¥{a/1e3:.0f}K"
    return f"{s}¥{a:.0f}"


def _dashboard_panels(ax_a, ax_b, ax_c, ax_d, x, account, money_pnl,
                      cum_ret_pct, dd_money, cap0, wm, final_pnl, max_dd_money,
                      bench_account=None, alpha_pct=None, bench_cum_pct=None,
                      ax_e=None, rmse_curve=None):
    import matplotlib.ticker as mticker

    # --- Panel A: account value (¥) vs benchmark ---
    ax_a.plot(x, account, color="#1a7f37", lw=1.6, label="Strategy account (¥)")
    if bench_account is not None:
        ax_a.plot(x, bench_account, color="#ff7f0e", lw=1.2, ls="--",
                  label="Equal-weight benchmark (¥)")
    ax_a.axhline(cap0, color="grey", lw=0.7, ls=":",
                 label=f"Initial capital ¥{cap0:,.0f}")
    ax_a.fill_between(x, cap0, account, where=(account >= cap0),
                      color="#1a7f37", alpha=0.10)
    ax_a.fill_between(x, cap0, account, where=(account < cap0),
                      color="#d62728", alpha=0.10)
    ax_a.set_title(
        "A. Account Value vs Equal-weight Universe Benchmark  (both starting at initial capital)",
        fontsize=10, loc="left")
    ax_a.set_ylabel("Account value (¥)")
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"¥{v/1e6:.2f}M"))
    ax_a.legend(loc="upper left", fontsize=8)
    ax_a.grid(alpha=0.3)
    bench_label = (f"\nBenchmark: {_money_fmt(bench_account[-1])}"
                   if bench_account is not None else "")
    ax_a.annotate(
        f"Strategy end: {_money_fmt(account[-1])}\nNet P/L: {_money_fmt(final_pnl)}{bench_label}",
        xy=(x[-1], account[-1]), fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eefaf0", ec="#1a7f37"))

    # --- Panel B: alpha curve (strategy return % − benchmark %) ---
    if alpha_pct is not None:
        ax_b.plot(x, alpha_pct, color="#9467bd", lw=1.3, label="Alpha (strategy − benchmark)")
        ax_b.fill_between(x, alpha_pct, 0, where=(alpha_pct >= 0),
                          color="#9467bd", alpha=0.15)
        ax_b.fill_between(x, alpha_pct, 0, where=(alpha_pct < 0),
                          color="#d62728", alpha=0.15)
        ax_b.axhline(0, color="grey", lw=0.8, ls="--")
        ax_b.set_title("B. Cumulative Alpha (Strategy % Return − Benchmark % Return)",
                       fontsize=10, loc="left")
        ax_b.set_ylabel("Alpha (%)")
        ax_b.annotate(f"End alpha: {alpha_pct[-1]:+.1f}%",
                      xy=(x[-1], alpha_pct[-1]), fontsize=8, ha="right",
                      bbox=dict(boxstyle="round,pad=0.2", fc="#f4f0ff", ec="#9467bd"))
    else:
        ax_b.plot(x, cum_ret_pct, color="#1f77b4", lw=1.3)
        ax_b.axhline(0, color="grey", lw=0.7, ls="--")
        ax_b.set_title("B. Cumulative Return (%)", fontsize=10, loc="left")
        ax_b.set_ylabel("Return (%)")
    ax_b.set_xlabel("Trading day")
    ax_b.legend(loc="upper left", fontsize=8)
    ax_b.grid(alpha=0.3)

    # --- Panel C: per-window OOS Sharpe ---
    if wm is not None and len(wm):
        colors = ["#1a7f37" if s >= 0 else "#d62728" for s in wm["Sharpe"]]
        ax_c.bar(wm["window"].astype(str), wm["Sharpe"], color=colors, alpha=0.8)
        ax_c.axhline(0, color="grey", lw=0.7)
        npos = int((wm["Sharpe"] > 0).sum())
        ax_c.set_title(f"C. Per-window OOS Sharpe  ({npos}/{len(wm)} positive)",
                       fontsize=10, loc="left")
        ax_c.set_ylabel("Sharpe")
        ax_c.set_xlabel("Walk-forward window #")
        ax_c.grid(alpha=0.3, axis="y")

    # --- Panel D: drawdown (as % of peak) ---
    dd_pct = (dd_money / peak) * 100.0     # percentage drawdown
    ax_d.fill_between(x, dd_pct, 0, color="#d62728", alpha=0.5)
    ax_d.set_title("D. Drawdown (% loss from peak account value)",
                   fontsize=10, loc="left")
    ax_d.set_ylabel("Drawdown (%)")
    ax_d.set_xlabel("Trading day")
    ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax_d.grid(alpha=0.3)
    max_dd_pct = float(np.min(dd_pct))
    ax_d.annotate(f"Max drawdown: {max_dd_pct:.2f}%",
                  xy=(x[int(np.argmin(dd_pct))], max_dd_pct),
                  fontsize=8, ha="center", va="top",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fdeeee", ec="#d62728"))

    # --- Panel E: daily price-RMSE of the predictions (¥) ---
    note_e = ""
    if ax_e is not None:
        if rmse_curve is not None and len(rmse_curve):
            rmse = rmse_curve["rmse"].to_numpy(dtype=float)
            xr = np.arange(len(rmse))
            mean_rmse = np.nanmean(rmse)
            ax_e.plot(xr, rmse, color="#8c564b", lw=1.1,
                      label="Daily price RMSE (¥)")
            ax_e.axhline(mean_rmse, color="#8c564b", lw=0.8, ls="--",
                         label=f"Mean ¥{mean_rmse:.3f}")
            if np.isfinite(rmse).any():
                imax = int(np.nanargmax(rmse))
                ax_e.annotate(f"Worst day: ¥{rmse[imax]:.3f}",
                              xy=(imax, rmse[imax]), fontsize=8, ha="center",
                              va="bottom",
                              bbox=dict(boxstyle="round,pad=0.2", fc="#f6efec",
                                        ec="#8c564b"))
            ax_e.set_title(
                "E. Daily Price-Prediction RMSE  (cross-sectional, ¥): "
                "sqrt(mean over stocks of (predicted − realized price)²)",
                fontsize=10, loc="left")
            ax_e.set_ylabel("Price RMSE (¥)")
            ax_e.legend(loc="upper left", fontsize=8)
            note_e = ("\nE. Per-day RMSE of predicted vs realized price across the held names (¥). "
                      "Lower = sharper forecasts; spikes flag the days the model misread most.")
        else:
            ax_e.text(0.5, 0.5, "Price RMSE unavailable (panel needs `close` in the factor panel)",
                      ha="center", va="center", fontsize=9, color="grey",
                      transform=ax_e.transAxes)
            ax_e.set_title("E. Daily Price-Prediction RMSE (¥)", fontsize=10, loc="left")
        ax_e.set_xlabel("Trading day")
        ax_e.grid(alpha=0.3)

    # --- How-to-read note ---
    note = ("How to read this dashboard:\n"
            "A. Strategy account value (green) vs equal-weight buy-and-hold of the same stock pool (orange dashed), both starting at the initial capital.\n"
            "   Strategy above benchmark = active selection is adding value; below = underperforming buy-and-hold.\n"
            "B. Cumulative alpha = strategy cumulative return % minus benchmark cumulative return % day by day.\n"
            "   Above zero (purple) = outperforming; below zero (red) = underperforming the universe average.\n"
            "C. Sharpe of each independent OOS window. More green bars = the edge repeats across different time periods.\n"
            "D. Drawdown (%): how far the account sits below its own historical peak, as a percentage of peak value. Shallower/shorter = better risk control."
            + note_e)
    anchor = ax_e if ax_e is not None else ax_d
    anchor.figure.text(0.01, -0.03, note, fontsize=7.5, va="top",
                       bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9",
                                 ec="#cccccc", alpha=0.9))
