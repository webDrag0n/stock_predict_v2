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

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
from factor_eval import select_passing_factors
from strategy import generate_positions
from backtest import run_backtest_periodic, format_metrics


# GBM 超参唯一真相源（sklearn 命名）。walkforward 的 _gbm_l2_predict /
# _train_predict_ret_weighted 与 ensemble.py 的增量续训（低层 lgb.train，经
# GBM_PARAMS_NATIVE 映射）共用这一处定义，杜绝两处手工镜像漂移。
GBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31,
    subsample=0.8, colsample_bytree=0.6, min_child_samples=40,
    reg_alpha=0.1, reg_lambda=0.5, objective="l2",
)
# 低层 lgb.train 命名映射。n_estimators 不在此列——ensemble 按 FULL_TREES /
# INCR_TREES 控制树数（续训语义），walkforward 用 sklearn 接口的 n_estimators。
GBM_PARAMS_NATIVE = dict(
    objective=GBM_PARAMS["objective"],
    learning_rate=GBM_PARAMS["learning_rate"],
    num_leaves=GBM_PARAMS["num_leaves"],
    bagging_fraction=GBM_PARAMS["subsample"], bagging_freq=1,
    feature_fraction=GBM_PARAMS["colsample_bytree"],
    min_child_samples=GBM_PARAMS["min_child_samples"],
    lambda_l1=GBM_PARAMS["reg_alpha"], lambda_l2=GBM_PARAMS["reg_lambda"],
    verbose=-1,
)


def _run_bt(df, pred, pos, pred_in_return_space: bool):
    """周期调仓引擎（唯一回测口径）。旧的每日重叠引擎 run_backtest 已删除
    （N 日收益被 N 个重叠 entry 日重复计 + ann=252 误年化，Sharpe/PnL 虚高）。"""
    return run_backtest_periodic(
        df, pred, pos, rebalance_days=C.REBALANCE_DAYS,
        pred_in_return_space=pred_in_return_space)


def _is_return_space() -> bool:
    """l2 and ret_weighted are genuine return-space regressors (price-RMSE /
    weighted-R2 are meaningful). lambdarank / ic_linear / fusion output ranking
    scores, for which only rank-based IC/ICIR is valid. Rank-combining the seeds
    (SEED_COMBINE=rank) also moves the output out of return space, so RMSE is
    only meaningful for the return-space objectives under mean-combine."""
    return (C.MODEL_OBJECTIVE in ("l2", "ret_weighted")
            and C.SEED_COMBINE == "mean")


def _combine_seeds(preds: list[np.ndarray], dates: np.ndarray) -> np.ndarray:
    """Merge n_seeds per-model predictions into one score.

    SEED_COMBINE="mean" averages raw predictions (default, stays in return space).
    "rank" averages each seed's PER-DATE percentile rank — scale-robust and the
    better tradeable combine when the seeds are highly correlated (validated on
    the cached whole-market OOS, ensemble_opt.py). Returns a 1-D array aligned to
    the input rows (i.e. to `dates`)."""
    stacked = np.vstack(preds)                      # (n_seeds, n_rows)
    if C.SEED_COMBINE != "rank":
        return stacked.mean(axis=0)
    # per-date percentile rank each seed, then average the ranks
    df = pd.DataFrame({"date": dates})
    rank_sum = np.zeros(stacked.shape[1], dtype=np.float64)
    for row in stacked:
        df["_p"] = row
        rank_sum += df.groupby("date")["_p"].rank(pct=True, method="average").to_numpy()
    return rank_sum / stacked.shape[0]


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
    objective = C.MODEL_OBJECTIVE
    if n_seeds is None:
        n_seeds = C.N_SEEDS
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


def _apply_nl_transforms(df: pd.DataFrame, factors: list[str]) -> list[str]:
    """对给定因子追加逐日 z 分数的非线性变换列（z²/|z|/正尾/负尾）。

    逐日截面计算（每组只用当日自身横截面），无泄露。原地写 df，返回新增列名。
    """
    cols = []
    for f in factors:
        grp = df.groupby("date", group_keys=False)[f]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, 1e-9)
        z = ((df[f] - mu) / sd).clip(-4.0, 4.0)
        for suffix, tfn in [("_sq", lambda x: x**2), ("_abs", np.abs),
                            ("_pos", lambda x: np.clip(x, 0, None)),
                            ("_neg", lambda x: np.clip(-x, 0, None))]:
            col = f"nl_{f}{suffix}"
            df[col] = tfn(z)
            cols.append(col)
    return cols


def _apply_ix_transforms(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[str]:
    """追加因子交互列（逐日 z 分数两两乘积，clip ±4）。原地写 df，返回新增列名。"""
    cols = []
    for fa, fb in pairs:
        col = f"ix_{fa}__x__{fb}"
        za = ((df[fa] - df.groupby("date")[fa].transform("mean"))
              / df.groupby("date")[fa].transform("std").replace(0, 1e-9))
        zb = ((df[fb] - df.groupby("date")[fb].transform("mean"))
              / df.groupby("date")[fb].transform("std").replace(0, 1e-9))
        df[col] = (za.clip(-4, 4) * zb.clip(-4, 4)).astype(np.float32)
        cols.append(col)
    return cols


def _linear_fit(train: pd.DataFrame, feat_cols: list[str]) -> dict:
    """ic_linear 的【训练侧】：定因子集 + 训练段 IC 权重。无泄露（只用 train）。

    返回的 state 可反复用于 _linear_score —— ensemble 按 REBUILD_DAYS 节奏缓存
    它（每日续训不重算 IC 权重与 NL_TOPK 变换列选择，全量重建时刷新）。
    输入先 .copy()：nl_/ix_ 变换列会原地写，不能污染调用方的帧。
    """
    train = train.copy()
    linear_feats = [f for f in (C.LINEAR_FACTORS or feat_cols) if f in train.columns]
    if not linear_feats:
        linear_feats = [f for f in feat_cols if f in train.columns]

    # --- Per-window factor transforms (LEAK-FREE: uses TRAINING segment only) ---
    nl_topk = C.NL_TOPK
    fi_topk = C.FI_TOPK
    top_by_ic = []
    if (nl_topk > 0 or fi_topk > 0) and linear_feats:
        from data.factor_transforms import _rank_factors_by_ic
        rankings = _rank_factors_by_ic(train, linear_feats, C.TARGET)
        top_by_ic = [f for f, _, _ in rankings[:max(nl_topk, fi_topk)]]

    # Non-linear transforms: z², |z|, pos/neg tails for top-N factors by training IC
    nl_factors = top_by_ic[:nl_topk] if nl_topk > 0 else []
    if nl_factors:
        for col in _apply_nl_transforms(train, nl_factors):
            if col not in linear_feats:
                linear_feats.append(col)

    # Factor interactions: pairwise products of top-N factor z-scores
    fi_max = C.FI_MAX_PAIRS
    ix_pairs: list[tuple[str, str]] = []
    if fi_topk > 0 and len(top_by_ic) >= 2:
        inter_factors = top_by_ic[:fi_topk]
        for i, fa in enumerate(inter_factors):
            for fb in inter_factors[i + 1:]:
                if len(ix_pairs) >= fi_max:
                    break
                ix_pairs.append((fa, fb))
            if len(ix_pairs) >= fi_max:
                break
    if ix_pairs:
        for col in _apply_ix_transforms(train, ix_pairs):
            if col not in linear_feats:
                linear_feats.append(col)

    # Size-neutralization
    if C.LINEAR_SIZE_NEUTRAL:
        train = _size_neutralize(train, linear_feats)

    # Factor stability filter
    min_stab = C.MIN_FACTOR_STABILITY
    if min_stab > 0:
        from data.factor_transforms import filter_stable_factors
        stable = set(filter_stable_factors(train, linear_feats,
                                           min_stability=min_stab))
        linear_feats = [f for f in linear_feats if f in stable]

    w = _training_ic_weights(train, linear_feats)
    return {"feats": linear_feats, "weights": w,
            "nl_factors": nl_factors, "ix_pairs": ix_pairs}


def _linear_score(test: pd.DataFrame, state: dict) -> np.ndarray:
    """ic_linear 的【预测侧】：对 test 逐日 z 分数按训练段 IC 权重加权求和。

    所有变换（nl_/ix_ 列、size 中性化、截面归一）都是逐日自包含的，因此可以
    安全地复用训练侧缓存的 state（因子集 + 权重），每天只重算 test 侧。
    """
    test = test.copy()
    _apply_nl_transforms(test, state["nl_factors"])
    _apply_ix_transforms(test, state["ix_pairs"])
    linear_feats = [f for f in state["feats"] if f in test.columns]
    if not linear_feats:
        return np.zeros(len(test), dtype=np.float64)
    if C.LINEAR_SIZE_NEUTRAL:
        test = _size_neutralize(test, linear_feats)
    Zte, _ = _cross_sectional_norm(test, linear_feats)
    w = state["weights"]
    score = np.zeros(len(test), dtype=np.float64)
    for j, f in enumerate(linear_feats):
        score += w[f] * np.nan_to_num(Zte[:, j], nan=0.0)
    return score


def _train_predict_linear(train: pd.DataFrame, test: pd.DataFrame,
                          feat_cols: list[str]) -> np.ndarray:
    """IC-weighted linear blend of factor z-scores (ranking score).

    By default (C.LINEAR_FACTORS empty) the blend uses every factor in `feat_cols`,
    so the unified factor library is fully consumed. If C.LINEAR_FACTORS is set to
    a non-empty list, only that curated subset is used.

    Weights come from the training segment only (leak-free); each factor's test
    z-score is per-date (leak-free). Output is a ranking score, not a return.
    实现 = _linear_fit(train) + _linear_score(test)（fit/score 拆分供 ensemble
    缓存训练侧状态）。
    """
    return _linear_score(test, _linear_fit(train, feat_cols))


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
            **GBM_PARAMS,
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
            **GBM_PARAMS,
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
    # 降内存：float64→float32，【逐列原地】降级以避免整帧临时副本
    # （整帧 astype 的瞬时峰值 ≈ 面板本体，formula 变体 9.4GB 面板下会先 OOM）。
    # 全市场面板 5.4M 行 × ~100 个 double 列 ≈ 4.3GB，逐窗口累积的 OOS 切片
    # 与最终 concat 叠加后峰值曾把 31GB 机器打到 OOM（rc=137，死在尾部窗口）。
    # LightGBM 内部按 bin 处理、IC/回测对日收益的精度需求远低于 float32 的
    # 7 位有效数字，降级不改变结论。
    for c in panel.columns:
        if panel[c].dtype == np.float64:
            panel[c] = panel[c].astype(np.float32)
    # weight 列保障（load_factor_panel 默认不含 weight，这里补等权）
    if C.WEIGHT not in panel.columns:
        panel[C.WEIGHT] = np.float32(1.0)
    # 整数日期序号，便于按窗口切分
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    n_dates = len(dates)

    train_w, test_w = C.TRAIN_WINDOW, C.TEST_WINDOW
    step, embargo = C.WF_STEP, C.EMBARGO_DAYS

    # 防御性断言（config._validate 已兜底，此处防模块被篡改/动态改配置）：
    # embargo 不足时训练段尾部标签（shift(-fw) 远期收益）会偷看 OOS 价格。
    assert embargo >= C.FORWARD_WINDOW, (
        f"EMBARGO_DAYS={embargo} < FORWARD_WINDOW={C.FORWARD_WINDOW}（前视泄露）")
    if C.MODEL_OBJECTIVE.startswith("multi_") and C.MULTI_HORIZON_WINDOWS:
        assert embargo >= max(C.MULTI_HORIZON_WINDOWS), (
            f"multi 目标要求 EMBARGO_DAYS >= {max(C.MULTI_HORIZON_WINDOWS)}")

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

        pos = generate_positions(_with_ids(test), pred, strategy=strategy)
        in_ret = _is_return_space()
        res = _run_bt(_with_ids(test), pred, pos, in_ret)
        m = res["metrics"]
        window_metrics.append({
            "window": win_id,
            "train": f"{pd.Timestamp(tr_dates[0]).date()}~{pd.Timestamp(tr_dates[-1]).date()}",
            "test": f"{pd.Timestamp(te_dates[0]).date()}~{pd.Timestamp(te_dates[-1]).date()}",
            "n_feat": len(feat),
            "PnL": m["total_pnl"], "Sharpe": m["sharpe"],
            "MaxDD": m["max_drawdown"], "hit": m["hit_rate"],
            "PriceRMSE": m.get("mean_price_rmse", float("nan")),
            "PredIC": m.get("pred_ic", float("nan")),
        })
        # 只保留最终拼接所需的列（回测: date/symbol/收益/amount/target +
        # prediction；仪表盘基准: date/return_1d）。全量 100+ 列逐窗口累积是
        # 运行尾部 OOM 的主要来源之一。
        keep = [c for c in ("date", "symbol", "return_1d", "pct_chg",
                            "amount", C.TARGET, "prediction")
                if c in test.columns]
        oos_frames.append(test[keep])
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

    # 拼接所有窗口的连续 OOS 净值（用各窗口已生成的持仓，不重算）
    oos = pd.concat(oos_frames, ignore_index=True)
    oos_pred = oos["prediction"].to_numpy()
    oos_pos = np.concatenate(oos_pos_frames)
    in_ret = _is_return_space()
    oos_res = _run_bt(_with_ids(oos), oos_pred, oos_pos, in_ret)

    return {
        "oos_result": oos_res,
        "window_metrics": pd.DataFrame(window_metrics),
        "n_windows": win_id,
        "oos": oos,
    }


def _with_ids(df: pd.DataFrame) -> pd.DataFrame:
    """给命名因子面板补 date_id/time_id/symbol_id（策略与回测按这些列分组）。"""
    out = df.copy()
    out[C.DATE_COL] = out["date"].rank(method="dense").astype(int)
    out[C.TIME_COL] = 0
    out[C.SYMBOL_COL] = out["symbol"].astype("category").cat.codes
    return out

