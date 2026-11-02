"""滚动走动回测（walk-forward）—— 严格防泄露的样本外验证。

针对单次划分回测 Sharpe 虚高（窗口运气 + 因子选择偷看测试集）的问题，本引擎：

  1. 滚动训练：每步用前 TRAIN_WINDOW 天训练，预测其后 TEST_WINDOW 天（样本外）
  2. embargo：训练段与测试段之间剔除 EMBARGO_DAYS 天，杜绝 N 日标签跨界泄露
  3. 因子选择隔离：每个窗口的因子选择只用该窗口的训练段（select_passing_factors）
  4. 拼接所有窗口的样本外预测 → 一条长达 1~2 年的连续 OOS 净值曲线

回测口径：唯一回测引擎是每日重放 run_daily_replay（backtest_daily.py，与实盘
共用 trading.py 的决策/撮合：次日开盘成交、T+1、整手、涨跌停、参与率上限、
五层风控）。每个窗口的 OOS 预测打包成 score_cache（带配方指纹）喂给重放引擎，
窗口内不再重训模型；窗口指标（Sharpe/PnL/MaxDD/hit）从逐笔撮合的每日账户
序列计算，比旧的周期调仓向量化近似更接近可成交结果。

直接在「命名因子面板」(load_factor_panel，需 keep_trade_cols=True 以携带
OHLC/撮合列) 上操作，模型用命名因子列训练，避免 feature_NN 映射，保证窗口内
因子选择与训练用的是同一套列。
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
from factor_eval import select_passing_factors, daily_rank_ic
from model_store import fingerprint_of
from perf_metrics import daily_replay_metrics

logger = logging.getLogger(__name__)


def _fmt_dur(sec: float) -> str:
    """秒数 → 人类可读时长（如 3m12s / 1h05m），供窗口进度日志的耗时/ETA 用。"""
    sec = max(int(sec), 0)
    if sec >= 3600:
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m"
    if sec >= 60:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec}s"


def _window_pos_ratios(wm: pd.DataFrame) -> dict:
    """逐窗口指标帧 → 「为正窗口占比」门禁信号（sharpe_pos_ratio / predic_pos_ratio）。

    分母只计非 NaN 窗口；无任何有效窗口时对应值为 None。Sharpe>0 占比衡量信号
    跨时间稳健复现，PredIC>0 占比衡量选股 alpha 是否成立（<50% 时正收益不可信）。
    """
    out = {"sharpe_pos_ratio": None, "predic_pos_ratio": None}
    for key, col in (("sharpe_pos_ratio", "Sharpe"), ("predic_pos_ratio", "PredIC")):
        if wm is None or len(wm) == 0 or col not in wm.columns:
            continue
        v = pd.to_numeric(wm[col], errors="coerce").dropna()
        if len(v):
            out[key] = float((v > 0).mean())
    return out


# GBM 超参唯一真相源（sklearn 命名）。walkforward 的 gbm_l2_predict /
# _train_predict_ret_weighted 与 ensemble.py 的增量续训（低层 lgb.train，经
# GBM_PARAMS_NATIVE 映射）共用这一处定义，杜绝两处手工镜像漂移。
GBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_samples=40,
    reg_alpha=0.1,
    reg_lambda=0.5,
    objective="l2",
)
# 低层 lgb.train 命名映射。n_estimators 不在此列——ensemble 按 FULL_TREES /
# INCR_TREES 控制树数（续训语义），walkforward 用 sklearn 接口的 n_estimators。
GBM_PARAMS_NATIVE = dict(
    objective=GBM_PARAMS["objective"],
    learning_rate=GBM_PARAMS["learning_rate"],
    num_leaves=GBM_PARAMS["num_leaves"],
    bagging_fraction=GBM_PARAMS["subsample"],
    bagging_freq=1,
    feature_fraction=GBM_PARAMS["colsample_bytree"],
    min_child_samples=GBM_PARAMS["min_child_samples"],
    lambda_l1=GBM_PARAMS["reg_alpha"],
    lambda_l2=GBM_PARAMS["reg_lambda"],
    verbose=-1,
)


def _scores_to_cache(test: pd.DataFrame) -> dict:
    """窗口 OOS 预测 → run_daily_replay 的 score_cache（{date: {symbol: score}}）。

    date 键直接取面板日期值（np.datetime64），与引擎内部 decision_dates 同源，
    避免日期 dtype/单位错位导致缓存查不中（查不中 = 当日空仓的静默偏差）。
    """
    cache = {}
    for d, g in test.groupby("date", sort=True):
        cache[np.datetime64(d)] = dict(
            zip(g["symbol"].to_numpy(), g["prediction"].to_numpy(dtype=np.float64))
        )
    return cache


def _replay_with_scores(
    panel: pd.DataFrame,
    feat_cols: list[str],
    cache: dict,
    start_date,
    end_date,
    params: dict | None,
):
    """用缓存分在 [start_date, end_date] 上跑每日重放（跳过全部模型训练）。

    score_cache 以 {"scores", "fingerprint"} 包装传入：指纹由
    model_store.fingerprint_of(feat_cols, params) 现算，与引擎内部校验值同源，
    不绕过时效性校验（指纹不匹配引擎会显式报错）。
    """
    # 延迟导入：backtest_daily → ensemble → walkforward 存在环，故在调用点导入。
    from backtest_daily import run_daily_replay

    p = dict(params or {})
    p["score_cache"] = {"scores": cache, "fingerprint": fingerprint_of(feat_cols, p)}
    return run_daily_replay(
        panel, feat_cols, params=p, start_date=start_date, end_date=end_date, progress=False
    )


def _curve_from_daily(daily: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """每日重放的 daily 帧 → OOS 净值曲线（净日收益/成本/暴露，占本金分数口径）。"""
    tv = daily["total_value"].to_numpy(dtype=np.float64)
    net = np.empty(len(tv), dtype=np.float64)
    net[0] = tv[0] / initial_cash - 1.0
    if len(tv) > 1:
        net[1:] = tv[1:] / tv[:-1] - 1.0
    tc = daily["cost"].to_numpy(dtype=np.float64) / initial_cash
    curve = pd.DataFrame(
        {
            "date": daily["date"].to_numpy(),
            "net": net,
            "gross": net + tc,
            "tc": tc,
            "tc_commission": (daily["commission"] + daily["stamp"]).to_numpy(dtype=np.float64)
            / initial_cash,
            "tc_impact": daily["impact"].to_numpy(dtype=np.float64) / initial_cash,
            "abs_pos": (daily["stock_value"] / daily["total_value"].replace(0, np.nan)).to_numpy(
                dtype=np.float64
            ),
        }
    )
    curve["equity"] = tv / initial_cash - 1.0
    return curve


def _pred_ic(frame: pd.DataFrame) -> tuple[float, float]:
    """OOS 帧的日度截面 RankIC 均值与 ICIR（prediction vs responder_6）。"""
    ic = daily_rank_ic(frame, "prediction", C.TARGET).dropna()
    if len(ic) == 0:
        return float("nan"), float("nan")
    icir = float(ic.mean() / ic.std()) if ic.std() > 1e-12 else float("nan")
    return float(ic.mean()), icir


def _combine_seeds(preds: list[np.ndarray], dates: np.ndarray) -> np.ndarray:
    """Merge n_seeds per-model predictions into one score.

    SEED_COMBINE="mean" averages raw predictions (default, stays in return space).
    "rank" averages each seed's PER-DATE percentile rank — scale-robust and the
    better tradeable combine when the seeds are highly correlated (validated on
    the cached whole-market OOS, ensemble_opt.py). Returns a 1-D array aligned to
    the input rows (i.e. to `dates`)."""
    stacked = np.vstack(preds)  # (n_seeds, n_rows)
    if C.SEED_COMBINE != "rank":
        return stacked.mean(axis=0)
    # per-date percentile rank each seed, then average the ranks
    df = pd.DataFrame({"date": dates})
    rank_sum = np.zeros(stacked.shape[1], dtype=np.float64)
    for row in stacked:
        df["_p"] = row
        rank_sum += df.groupby("date")["_p"].rank(pct=True, method="average").to_numpy()
    return rank_sum / stacked.shape[0]


def cross_sectional_norm(
    df: pd.DataFrame, feat_cols: list[str], stats: dict | None = None
) -> tuple[np.ndarray, dict]:
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


def train_predict(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], n_seeds: int = None
) -> np.ndarray:
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
        return train_predict_multi_ret_weighted(train, test, feat_cols, n_seeds)
    if objective == "multi_fusion":
        return _train_predict_multi_fusion(train, test, feat_cols, n_seeds)
    if objective == "attn":
        # 截面自注意力（GPU）。独立模块，仅在选用时 import torch。
        from model_attn import attn_predict

        return attn_predict(train, test, feat_cols)

    # default l2 regression leg (factored into gbm_l2_predict so the fusion
    # objective reuses the exact same model without duplicating its config)
    return gbm_l2_predict(train, test, feat_cols, n_seeds)


def _train_predict_rank(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feat_cols: list[str],
    n_seeds: int = 3,
    n_grades: int = 16,
) -> np.ndarray:
    """LambdaRank variant: per-date groups, target binned into integer grades.

    LGBMRanker maximizes NDCG over each date's cross-section, so it learns the
    ordering that long_topk consumes directly. The forward excess return is
    discretized into `n_grades` per-date quantile buckets (relevance labels).
    Predictions are returned in the ORIGINAL `test` row order.
    """
    # sort train by date so each date's rows form one contiguous ranking group
    tr_order = np.argsort(train["date"].to_numpy(), kind="stable")
    tr_sorted = train.iloc[tr_order]
    Xtr_all, _ = cross_sectional_norm(train, feat_cols)
    Xtr = Xtr_all[tr_order]
    y_raw = tr_sorted[C.TARGET].to_numpy(dtype=np.float64)

    # per-date quantile bucket -> integer relevance grade in [0, n_grades-1]
    grades = (
        tr_sorted.groupby("date", group_keys=False)[C.TARGET]
        .apply(
            lambda s: pd.qcut(
                s.rank(method="first"),
                min(n_grades, max(s.notna().sum(), 1)),
                labels=False,
                duplicates="drop",
            )
        )
        .to_numpy()
    )
    grades = np.nan_to_num(grades, nan=0.0).astype(int)
    _, group_tr = np.unique(tr_sorted["date"].to_numpy(), return_counts=True)

    # test features in original order (predictions don't need group structure)
    Xte, _ = cross_sectional_norm(test, feat_cols)
    _ = y_raw

    preds = []
    for s in range(n_seeds):
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=40,
            reg_alpha=0.1,
            reg_lambda=0.5,
            label_gain=list(range(n_grades + 1)),
            random_state=C.RANDOM_STATE + s,
            n_jobs=-1,
            verbose=-1,
        )
        ranker.fit(Xtr, grades, group=group_tr)
        preds.append(np.asarray(ranker.predict(Xte), dtype=np.float64).ravel())
    return _combine_seeds(preds, test["date"].to_numpy())


def _training_ic_weights(train: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Per-factor weight = mean daily cross-sectional RankIC over the TRAIN segment.

    Leak-free: uses only the training segment's own forward returns. The sign is
    kept, so a reversal factor (negative IC) automatically gets a negative weight
    and is subtracted in the blend. Factors with <2 valid IC days get weight 0.
    """
    cols = [f for f in feat_cols if f in train.columns]
    if not cols or len(train) == 0:
        return {f: 0.0 for f in feat_cols}

    # Vectorized per-day Spearman: per-date average ranks (NaN preserved), then
    # batched Pearson on ranks via grouped sums. Matches the old per-day
    # scipy.spearmanr semantics: NaN propagates (a day with any NaN in the factor
    # or the target yields no IC), >=5 pairwise-valid rows required, and the
    # factor needs >=2 distinct values that day.
    dates = train["date"].to_numpy()
    order = np.argsort(dates, kind="stable")
    _, starts, counts = np.unique(dates[order], return_index=True, return_counts=True)
    n_day = counts.astype(np.float64)
    gb = train.groupby("date", sort=False)

    yr = gb[C.TARGET].rank(method="average").to_numpy(dtype=np.float64)[order]
    y_nan = np.isnan(yr)
    yf = np.where(y_nan, 0.0, yr)
    y_ok = (
        (np.add.reduceat(y_nan, starts) == 0)
        & (counts >= 5)
        & (
            np.maximum.reduceat(np.where(y_nan, -np.inf, yr), starts)
            > np.minimum.reduceat(np.where(y_nan, np.inf, yr), starts)
        )
    )
    sy = np.add.reduceat(yf, starts)
    syy = np.add.reduceat(yf * yf, starts)
    vy = np.maximum(syy - sy * sy / n_day, 0.0)

    # 16 列一块：块内 rank/布尔/填零副本均为 f64（64 列块在全市场窗口帧上
    # 瞬时 ~4GB），数值结果与块大小无关（块间独立累计）。
    weights = {}
    for i0 in range(0, len(cols), 16):
        block = cols[i0 : i0 + 16]
        Xr = gb[block].rank(method="average").to_numpy(dtype=np.float64)[order]
        x_nan = np.isnan(Xr)
        Xf = np.where(x_nan, 0.0, Xr)
        sx = np.add.reduceat(Xf, starts, axis=0)
        sxx = np.add.reduceat(Xf * Xf, starts, axis=0)
        sxy = np.add.reduceat(Xf * yf[:, None], starts, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy - sx * (sy / n_day)[:, None]
            vx = np.maximum(sxx - sx * sx / n_day[:, None], 0.0)
            ic = cov / np.sqrt(vx * vy[:, None])
        ok = (
            y_ok[:, None]
            & (np.add.reduceat(x_nan, starts, axis=0) == 0)
            & (
                np.maximum.reduceat(np.where(x_nan, -np.inf, Xr), starts, axis=0)
                > np.minimum.reduceat(np.where(x_nan, np.inf, Xr), starts, axis=0)
            )
        )
        ic = np.where(ok, ic, np.nan)
        valid = np.isfinite(ic)
        cnt = valid.sum(axis=0)
        tot = np.where(valid, ic, 0.0).sum(axis=0)
        for j, f in enumerate(block):
            weights[f] = float(tot[j] / cnt[j]) if cnt[j] >= 2 else 0.0
    return {f: weights.get(f, 0.0) for f in feat_cols}


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
        for suffix, tfn in [
            ("_sq", lambda x: x**2),
            ("_abs", np.abs),
            ("_pos", lambda x: np.clip(x, 0, None)),
            ("_neg", lambda x: np.clip(-x, 0, None)),
        ]:
            col = f"nl_{f}{suffix}"
            # float32 存储（与 _apply_ix_transforms 一致）：计算在 f64 进行，
            # 仅落列时降精度，ic_linear 训练帧内存减半
            df[col] = tfn(z).astype(np.float32)
            cols.append(col)
    return cols


def _apply_ix_transforms(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[str]:
    """追加因子交互列（逐日 z 分数两两乘积，clip ±4）。原地写 df，返回新增列名。"""
    cols = []
    for fa, fb in pairs:
        col = f"ix_{fa}__x__{fb}"
        za = (df[fa] - df.groupby("date")[fa].transform("mean")) / df.groupby("date")[fa].transform(
            "std"
        ).replace(0, 1e-9)
        zb = (df[fb] - df.groupby("date")[fb].transform("mean")) / df.groupby("date")[fb].transform(
            "std"
        ).replace(0, 1e-9)
        df[col] = (za.clip(-4, 4) * zb.clip(-4, 4)).astype(np.float32)
        cols.append(col)
    return cols


def linear_fit(train: pd.DataFrame, feat_cols: list[str]) -> dict:
    """ic_linear 的【训练侧】：定因子集 + 训练段 IC 权重。无泄露（只用 train）。

    返回的 state 可反复用于 linear_score —— ensemble 按 REBUILD_DAYS 节奏缓存
    它（每日续训不重算 IC 权重与 NL_TOPK 变换列选择，全量重建时刷新）。
    输入先按需取列（pandas>=3 CoW 懒视图 / 旧版显式拷贝）：nl_/ix_ 变换列
    会原地写、size 中性化会改写因子列，不能污染调用方的帧。
    """
    linear_feats = [f for f in (C.LINEAR_FACTORS or feat_cols) if f in train.columns]
    if not linear_feats:
        linear_feats = [f for f in feat_cols if f in train.columns]
    keep = ["date", C.TARGET, *linear_feats]
    if C.LINEAR_SIZE_NEUTRAL:
        keep.append("ln_mv")
    # 按需取列：nl_/ix_ 变换列会原地写、size 中性化会改写因子列，不能污染
    # 调用方的帧。pandas>=3 的 CoW 下列为懒视图（写入时逐列复制，父帧不受
    # 影响），省略显式 .copy() 可在全市场窗口帧（GB 级）下省一份完整拷贝
    # （nextday OOM 修复）；pandas<3 保持显式拷贝（语义与旧版一致）。
    train = train[[c for c in dict.fromkeys(keep) if c in train.columns]]
    if int(pd.__version__.split(".", 1)[0]) < 3:
        train = train.copy()

    # --- Per-window factor transforms (LEAK-FREE: uses TRAINING segment only) ---
    nl_topk = C.NL_TOPK
    fi_topk = C.FI_TOPK
    top_by_ic = []
    if (nl_topk > 0 or fi_topk > 0) and linear_feats:
        from data.factor_transforms import _rank_factors_by_ic

        rankings = _rank_factors_by_ic(train, linear_feats, C.TARGET)
        top_by_ic = [f for f, _, _ in rankings[: max(nl_topk, fi_topk)]]

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
            for fb in inter_factors[i + 1 :]:
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

        stable = set(filter_stable_factors(train, linear_feats, min_stability=min_stab))
        linear_feats = [f for f in linear_feats if f in stable]

    w = _training_ic_weights(train, linear_feats)
    return {"feats": linear_feats, "weights": w, "nl_factors": nl_factors, "ix_pairs": ix_pairs}


def linear_score(test: pd.DataFrame, state: dict) -> np.ndarray:
    """ic_linear 的【预测侧】：对 test 逐日 z 分数按训练段 IC 权重加权求和。

    所有变换（nl_/ix_ 列、size 中性化、截面归一）都是逐日自包含的，因此可以
    安全地复用训练侧缓存的 state（因子集 + 权重），每天只重算 test 侧。
    """
    keep = {"date", *state["nl_factors"], *state["feats"]}
    for fa, fb in state["ix_pairs"]:
        keep.update((fa, fb))
    if C.LINEAR_SIZE_NEUTRAL:
        keep.add("ln_mv")
    test = test[[c for c in test.columns if c in keep]].copy()
    _apply_nl_transforms(test, state["nl_factors"])
    _apply_ix_transforms(test, state["ix_pairs"])
    linear_feats = [f for f in state["feats"] if f in test.columns]
    if not linear_feats:
        return np.zeros(len(test), dtype=np.float64)
    if C.LINEAR_SIZE_NEUTRAL:
        test = _size_neutralize(test, linear_feats)
    Zte, _ = cross_sectional_norm(test, linear_feats)
    w = state["weights"]
    score = np.zeros(len(test), dtype=np.float64)
    for j, f in enumerate(linear_feats):
        score += w[f] * np.nan_to_num(Zte[:, j], nan=0.0)
    return score


def _train_predict_linear(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str]
) -> np.ndarray:
    """IC-weighted linear blend of factor z-scores (ranking score).

    By default (C.LINEAR_FACTORS empty) the blend uses every factor in `feat_cols`,
    so the unified factor library is fully consumed. If C.LINEAR_FACTORS is set to
    a non-empty list, only that curated subset is used.

    Weights come from the training segment only (leak-free); each factor's test
    z-score is per-date (leak-free). Output is a ranking score, not a return.
    实现 = linear_fit(train) + linear_score(test)（fit/score 拆分供 ensemble
    缓存训练侧状态）。
    """
    return linear_score(test, linear_fit(train, feat_cols))


def _size_neutralize(df: pd.DataFrame, factors: list[str], size_col: str = "ln_mv") -> pd.DataFrame:
    """Replace each factor with its per-date residual after regressing on ln_mv.

    Leak-free: each trading day is regressed against only that day's own
    cross-section. size_col absent (or a factor missing) → that factor is left
    untouched. Used by the ic_linear / size-neutral predictor.
    原地改写 df（调用方持有的均为私有副本）并返回它。
    """
    if size_col not in df.columns:
        return df
    cols = [f for f in factors if f != size_col and f in df.columns]
    if not cols or len(df) == 0:
        return df

    # Batched per-day OLS on [1, ln_mv]: grouped sums solve the 2x2 normal
    # equations for all days at once. Same NaN semantics as the old per-day
    # polyfit: rows are dropped pairwise (non-finite size or factor), a day
    # needs >=10 valid rows and non-constant size, otherwise all-NaN residuals.
    dates = df["date"].to_numpy()
    order = np.argsort(dates, kind="stable")
    _, starts, counts = np.unique(dates[order], return_index=True, return_counts=True)
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order))
    day_idx = np.repeat(np.arange(len(starts)), counts)
    xs = df[size_col].to_numpy(dtype=np.float64)[order]
    x_fin = np.isfinite(xs)

    for i0 in range(0, len(cols), 16):  # 16 列一块，控制 Y/xf/Yf/R 等 f64 块副本峰值
        block = cols[i0 : i0 + 16]
        Y = df[block].to_numpy(dtype=np.float64)[order]
        V = np.isfinite(Y) & x_fin[:, None]
        xf = np.where(V, xs[:, None], 0.0)
        Yf = np.where(V, Y, 0.0)
        n = np.add.reduceat(V, starts, axis=0).astype(np.float64)
        sx = np.add.reduceat(xf, starts, axis=0)
        sxx = np.add.reduceat(xf * xf, starts, axis=0)
        sy = np.add.reduceat(Yf, starts, axis=0)
        sxy = np.add.reduceat(xf * Yf, starts, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = sxx / n - (sx / n) ** 2
            slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            intercept = (sy - slope * sx) / n
        ok = (
            (n >= 10)
            & (var > 1e-18)
            & (
                np.maximum.reduceat(np.where(V, xs[:, None], -np.inf), starts, axis=0)
                > np.minimum.reduceat(np.where(V, xs[:, None], np.inf), starts, axis=0)
            )
        )
        slope = np.where(ok, slope, 0.0)
        intercept = np.where(ok, intercept, 0.0)
        R = Y - (slope[day_idx] * xs[:, None] + intercept[day_idx])
        R = np.where(V & ok[day_idx], R, np.nan)[inv]
        # 残差按 float32 落列（回归在 f64 完成，仅存储降精度）：
        # ic_linear 训练帧全列被中性化改写，f64 存储会让帧体积翻倍。
        # clip 到 f32 值域防极端病态值 cast 溢出（排名语义不变：仍为极端尾部）。
        f32_max = np.finfo(np.float32).max
        for j, f in enumerate(block):
            df[f] = np.clip(R[:, j], -f32_max, f32_max).astype(np.float32)
    return df


def _train_predict_fusion(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], n_seeds: int = 3
) -> np.ndarray:
    """Rank-average of the l2 GBM and the ic_linear blend (ranking score).

    Each model's prediction is converted to a per-date percentile rank, then the
    two ranks are averaged. De-correlates the GBM's and the linear model's errors
    (the GBM captures non-linearities, the linear blend resists noise over-fit),
    which historically lifts ICIR above either model alone.
    """
    # GBM leg: reuse the l2 path by temporarily forcing the objective off the
    # dispatch (call the regressor body directly via a small replica).
    gbm = gbm_l2_predict(train, test, feat_cols, n_seeds)
    lin = _train_predict_linear(train, test, feat_cols)

    dates = test["date"].to_numpy()
    rg = per_date_rank(gbm, dates)
    rl = per_date_rank(lin, dates)
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


def train_predict_multi_ret_weighted(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], n_seeds: int = 3
) -> np.ndarray:
    """Multi-horizon ret_weighted fusion.

    Trains one ret_weighted GBM per horizon (primary FORWARD_WINDOW plus
    MULTI_HORIZON_WINDOWS), converts each horizon's predictions to per-date
    percentile ranks, then fuses by MULTI_HORIZON_WEIGHTS. Output is a ranking
    score that blends short/medium/long-term signals.
    """
    horizons = [int(getattr(C, "FORWARD_WINDOW", 20))] + [
        int(x) for x in getattr(C, "MULTI_HORIZON_WINDOWS", [])
    ]
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
        score += w * per_date_rank(pred, dates)
    return score


def _train_predict_multi_fusion(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], n_seeds: int = 3
) -> np.ndarray:
    """Multi-horizon fusion of GBM + ic_linear per horizon.

    For each horizon, compute rank-average of the l2 GBM and ic_linear blend,
    then rank-average across horizons weighted by MULTI_HORIZON_WEIGHTS.
    Maximizes both model diversity and horizon diversity.
    """
    horizons = [int(getattr(C, "FORWARD_WINDOW", 20))] + [
        int(x) for x in getattr(C, "MULTI_HORIZON_WINDOWS", [])
    ]
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
        gbm = gbm_l2_predict(tr, te, feat_cols, n_seeds)
        lin = _train_predict_linear(tr, te, feat_cols)
        horizon_score = (per_date_rank(gbm, dates) + per_date_rank(lin, dates)) / 2.0
        score += w * horizon_score
    return score


def per_date_rank(pred: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Per-date percentile rank in [0, 1] (ties averaged). Leak-free (within date)."""
    s = pd.Series(np.asarray(pred, dtype=np.float64))
    grp = pd.Series(dates)
    return s.groupby(grp).rank(pct=True).to_numpy()


def gbm_l2_predict(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], n_seeds: int = 3
) -> np.ndarray:
    """The l2 regression leg of train_predict, factored out for reuse by fusion."""
    Xtr, _ = cross_sectional_norm(train, feat_cols)
    ytr = train[C.TARGET].to_numpy(dtype=np.float64)
    wtr = train[C.WEIGHT].to_numpy() if C.WEIGHT in train else None
    Xte, _ = cross_sectional_norm(test, feat_cols)

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
            **GBM_PARAMS, random_state=C.RANDOM_STATE + s, n_jobs=-1, verbose=-1
        )
        if val_mask is not None:
            model.fit(
                Xtr[fit_mask],
                ytr[fit_mask],
                sample_weight=(wtr[fit_mask] if wtr is not None else None),
                eval_set=[(Xtr[val_mask], ytr[val_mask])],
                eval_sample_weight=([wtr[val_mask]] if wtr is not None else None),
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            model.fit(Xtr, ytr, sample_weight=wtr)
        preds.append(np.asarray(model.predict(Xte), dtype=np.float64).ravel())
    return _combine_seeds(preds, test["date"].to_numpy())


def _train_predict_ret_weighted(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], n_seeds: int = 3
) -> np.ndarray:
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
    Xtr, _ = cross_sectional_norm(train, feat_cols)
    ytr = train[C.TARGET].to_numpy(dtype=np.float64)
    Xte, _ = cross_sectional_norm(test, feat_cols)

    rnk = (train.groupby("date", group_keys=False)[C.TARGET].rank(pct=True)).to_numpy()
    w = 0.2 + np.nan_to_num(rnk, nan=0.5)  # head ~6x the tail
    # fold in the existing inverse-vol/recency weight if present
    if C.WEIGHT in train:
        w = w * train[C.WEIGHT].to_numpy()

    preds = []
    for s in range(n_seeds):
        model = lgb.LGBMRegressor(
            **GBM_PARAMS, random_state=C.RANDOM_STATE + s, n_jobs=-1, verbose=-1
        )
        model.fit(Xtr, ytr, sample_weight=w)
        preds.append(np.asarray(model.predict(Xte), dtype=np.float64).ravel())
    return _combine_seeds(preds, test["date"].to_numpy())


def save_walkforward_result(res: dict, outdir: str | None = None) -> str:
    """把 run_walkforward 的返回 dict 落盘到 canonical 目录（默认 <OUTPUT_DIR>/walkforward/）。

    CLI（scripts/run_walkforward.py）与 webapp 会话（pipeline wf_run）共用此
    产物契约；前端会话页直接读取该目录展示「最近一次 CLI Walk-Forward 结果」：

      oos_curve.parquet      连续 OOS 净值曲线（date/net/gross/tc/equity…）
      window_metrics.parquet 逐窗口 OOS 指标
      oos_bench.parquet      基准日收益（date, bench_ret = 逐日 return_1d 均值；
                             res["oos"] 无 return_1d 列则不写）
      meta.json              OOS 汇总指标 + 配置快照 + generated_at
                             + sharpe_pos_ratio / predic_pos_ratio（为正窗口占比门禁）
                             （numpy/pandas 类型经 pipeline.context._jsonable
                             转纯 Python，保证可 json.dump）

    返回落盘目录路径。
    """
    # 延迟导入：src 层不依赖 pipeline 包的 import 期状态（pipeline/context.py
    # 只用 stdlib+numpy+pandas，无环）
    from pipeline.context import _jsonable

    outdir = outdir or os.path.join(C.OUTPUT_DIR, "walkforward")
    os.makedirs(outdir, exist_ok=True)

    res["oos_result"]["curve"].to_parquet(os.path.join(outdir, "oos_curve.parquet"), index=False)
    res["window_metrics"].to_parquet(os.path.join(outdir, "window_metrics.parquet"), index=False)

    oos = res.get("oos")
    if oos is not None and "return_1d" in oos.columns:
        bench = (
            oos.groupby("date")["return_1d"]
            .mean()
            .sort_index()
            .reset_index()
            .rename(columns={"return_1d": "bench_ret"})
        )
        bench.to_parquet(os.path.join(outdir, "oos_bench.parquet"), index=False)

    # 「为正窗口占比」门禁信号：优先用 run_walkforward 返回 dict 里已算好的值，
    # 缺省（手搓 res dict / 旧调用方）时从 window_metrics 现算。
    ratios = _window_pos_ratios(res.get("window_metrics"))
    sharpe_pos_ratio = res.get("sharpe_pos_ratio", ratios["sharpe_pos_ratio"])
    predic_pos_ratio = res.get("predic_pos_ratio", ratios["predic_pos_ratio"])

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": res["oos_result"]["metrics"],
        "n_windows": res.get("n_windows"),
        "select_features": res.get("select_features"),
        "sharpe_pos_ratio": sharpe_pos_ratio,
        "predic_pos_ratio": predic_pos_ratio,
        "config": {
            "train_window": C.TRAIN_WINDOW,
            "test_window": C.TEST_WINDOW,
            "wf_step": C.WF_STEP,
            "embargo_days": C.EMBARGO_DAYS,
            "forward_window": C.FORWARD_WINDOW,
            "model_objective": C.MODEL_OBJECTIVE,
            "strategy": res.get("strategy") or "long_topk",
            "initial_capital": C.INITIAL_CAPITAL,
            "universe_mode": C.UNIVERSE_MODE,
            "universe_size": C.UNIVERSE_SIZE,
        },
    }
    if oos is not None and len(oos):
        meta["universe"] = {
            "n_symbols": int(oos["symbol"].nunique()),
            "n_oos_days": int(oos["date"].nunique()),
            "oos_start": oos["date"].min(),
            "oos_end": oos["date"].max(),
        }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, ensure_ascii=False, indent=2)
    return outdir


def run_walkforward(
    panel: pd.DataFrame,
    all_factor_cols: list[str],
    select_features: bool = False,
    strategy: str = "long_topk",
    verbose: bool = True,
    replay_params: dict | None = None,
    save: bool = True,
    progress_cb=None,
) -> dict:
    """滚动回测。返回拼接后的 OOS 结果与各窗口指标分布。

    panel: 命名因子面板（含 date, symbol, 因子..., responder_6, weight；
           回测撮合需 keep_trade_cols=True 的 OHLC/amount/pct_chg 等交易列）
    all_factor_cols: 全部候选因子名
    select_features: True 则每窗口在训练段内做因子准入筛选（隔离，无泄露）
    strategy: 已废弃——唯一回测引擎 run_daily_replay 固定走 decide_orders 的
              A股长多口径（无做空/整手/T+1）；保留形参仅为调用方兼容。
    replay_params: 透传给 run_daily_replay 的决策/撮合参数覆盖（通常不用动）。
    save: True 则跑完后调用 save_walkforward_result 落盘 canonical 目录
          outputs/walkforward/（CLI 默认；webapp 会话传 False，由 wf_run 在
          会话 artifacts 之后显式落盘，避免双写）。
    progress_cb: 可选细粒度进度回调，签名 cb(phase, current, total, **extras)
          （与 run_daily_replay 的 params["progress_cb"] 同一约定，由
          pipeline 步骤包装 ctx.report_progress 注入）。phase ∈
          "windows"（每完成一个窗口，extras 带 date=窗口OOS起始日/pred_ic）
          / "final_replay"（尾部拼接整体重放）。None 时零开销；回调异常一律
          吞掉——进度永远不能影响主流程。
    """
    if strategy != "long_topk":
        warnings.warn(
            f"strategy={strategy!r} 已失效：每日重放引擎（唯一回测口径）为 A股长多"
            "逐笔撮合，无空头仓位；结果等同于 long_topk。",
            stacklevel=2,
        )
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
    # 撮合所需交易列检查（缺列不报错但订单无法成交，结果静默归零，显式警告）
    missing_trade = [c for c in ("open", "high", "low", "close") if c not in panel.columns]
    if missing_trade:
        warnings.warn(
            f"面板缺交易列 {missing_trade}（需 load_factor_panel(keep_trade_cols=True)）："
            "每日重放的订单将全部无法成交，PnL/Sharpe 恒为 0。",
            stacklevel=2,
        )
    # 整数日期序号，便于按窗口切分。load_factor_panel 的缓存链路已保证
    # (date, symbol) 有序——已单调则跳过一次 13GB+ 的全帧 sort 拷贝。
    if not panel["date"].is_monotonic_increasing:
        panel = panel.sort_values("date")
    panel = panel.reset_index(drop=True)
    dates = np.sort(panel["date"].unique())
    n_dates = len(dates)

    train_w, test_w = C.TRAIN_WINDOW, C.TEST_WINDOW
    step, embargo = C.WF_STEP, C.EMBARGO_DAYS

    # 防御性断言（config._validate 已兜底，此处防模块被篡改/动态改配置）：
    # embargo 不足时训练段尾部标签（shift(-fw) 远期收益）会偷看 OOS 价格。
    assert (
        embargo >= C.FORWARD_WINDOW
    ), f"EMBARGO_DAYS={embargo} < FORWARD_WINDOW={C.FORWARD_WINDOW}（前视泄露）"
    if C.MODEL_OBJECTIVE.startswith("multi_") and C.MULTI_HORIZON_WINDOWS:
        assert embargo >= max(
            C.MULTI_HORIZON_WINDOWS
        ), f"multi 目标要求 EMBARGO_DAYS >= {max(C.MULTI_HORIZON_WINDOWS)}"

    oos_frames = []
    window_metrics = []
    # 全部窗口的 score_cache 汇总：拼接口径的连续 OOS 用同一份缓存分跑一遍
    # 完整区间的每日重放（单一组合跨越窗口边界，持仓/现金天然连续）。
    combined_cache: dict = {}
    first_te_start = last_te_end = None

    # 窗口总数可预先算出（顺推 + 互不重叠），供进度日志打 i/N 与 ETA。
    n_windows_total = max((n_dates - train_w - embargo - test_w) // step + 1, 0)
    logger.info(
        f"walk-forward 开始：共 {n_windows_total} 个窗口 "
        f"(train={train_w} test={test_w} step={step} embargo={embargo}, "
        f"select_features={select_features})"
    )
    t_wf0 = time.time()

    start = train_w + embargo
    win_id = 0
    while start + test_w <= n_dates:
        t_win0 = time.time()
        tr_lo_i = start - embargo - train_w
        tr_hi_i = start - embargo  # exclusive
        te_lo_i = start
        te_hi_i = start + test_w  # exclusive

        tr_dates = dates[tr_lo_i:tr_hi_i]
        te_dates = dates[te_lo_i:te_hi_i]
        logger.debug(
            f"窗口 {win_id + 1}/{n_windows_total} 开始：训练"
            f"{tr_dates[0].astype('datetime64[D]')}~{tr_dates[-1].astype('datetime64[D]')} → "
            f"测试{te_dates[0].astype('datetime64[D]')}~{te_dates[-1].astype('datetime64[D]')}"
        )
        # 面板按 date 有序，窗口日期是连续区间：searchsorted 行切片（视图）
        # 替代 isin 布尔拷贝——每窗口省一份 ~2.2GB 的 float32 训练段副本。
        d_all = panel["date"].to_numpy()
        train = panel.iloc[
            int(np.searchsorted(d_all, tr_dates[0], "left")) : int(
                np.searchsorted(d_all, tr_dates[-1], "right")
            )
        ]
        test = panel.iloc[
            int(np.searchsorted(d_all, te_dates[0], "left")) : int(
                np.searchsorted(d_all, te_dates[-1], "right")
            )
        ].copy()

        # 窗口内因子选择（仅用训练段日期）
        if select_features:
            feat = select_passing_factors(panel, date_lo=tr_dates[0], date_hi=tr_dates[-1])
            feat = [f for f in feat if f in all_factor_cols]
            if not feat:  # 该窗口无因子达标 → 退回全特征
                feat = all_factor_cols
        else:
            feat = all_factor_cols

        # cross-sectional-normalized, multi-seed, early-stopped LGB ensemble
        t_train0 = time.time()
        pred = train_predict(train, test, feat)
        t_train = time.time() - t_train0

        test["prediction"] = pred

        # 每日重放（唯一回测口径）：该窗口 OOS 预测作为 score_cache 喂入，
        # 跳过模型训练，只做 决策→次日撮合→估值。窗口指标来自逐笔撮合的
        # 每日账户序列（含 T+1/整手/涨跌停/参与率上限/风控叠乘）。
        t_replay0 = time.time()
        cache = _scores_to_cache(test)
        combined_cache.update(cache)  # step<=test_w 时窗口互不重叠，无键冲突
        replay = _replay_with_scores(panel, feat, cache, te_dates[0], te_dates[-1], replay_params)
        t_replay = time.time() - t_replay0
        m = daily_replay_metrics(replay["daily"], replay["meta"]["initial_cash"])
        pred_ic, _ = _pred_ic(test)
        window_metrics.append(
            {
                "window": win_id,
                "train": f"{pd.Timestamp(tr_dates[0]).date()}~{pd.Timestamp(tr_dates[-1]).date()}",
                "test": f"{pd.Timestamp(te_dates[0]).date()}~{pd.Timestamp(te_dates[-1]).date()}",
                "n_feat": len(feat),
                "PnL": m["total_pnl"],
                "Sharpe": m["sharpe"],
                "MaxDD": m["max_drawdown"],
                "hit": m["hit_rate"],
                "PriceRMSE": float("nan"),  # 排名/重放口径下无逐日价格预测，恒 NaN
                "PredIC": pred_ic,
            }
        )
        if first_te_start is None:
            first_te_start = te_dates[0]
        last_te_end = te_dates[-1]
        # 只保留最终拼接所需的列（基准/IC: date/symbol/收益/amount/target +
        # prediction；仪表盘基准: date/return_1d）。全量 100+ 列逐窗口累积是
        # 运行尾部 OOM 的主要来源之一。
        keep = [
            c
            for c in ("date", "symbol", "return_1d", "pct_chg", "amount", C.TARGET, "prediction")
            if c in test.columns
        ]
        oos_frames.append(test[keep])
        # 窗口完成进度：i/N、训练/测试区间、Sharpe/PnL、本轮耗时与 ETA
        # （ETA = 已完成窗口平均耗时 × 剩余窗口数）。info 级——全市场一次运行
        # 1~3 小时，窗口进度是估计剩余时间的唯一手段。
        win_el = time.time() - t_win0
        avg_el = (time.time() - t_wf0) / (win_id + 1)
        eta = avg_el * max(n_windows_total - win_id - 1, 0)
        ic_str = f"预测IC={pred_ic:+.3f}" if pred_ic == pred_ic else ""
        logger.info(
            f"  窗口 {win_id + 1}/{n_windows_total}: 训练"
            f"{tr_dates[0].astype('datetime64[D]')}~{tr_dates[-1].astype('datetime64[D]')} → "
            f"测试{te_dates[0].astype('datetime64[D]')}~"
            f"{te_dates[-1].astype('datetime64[D]')} | "
            f"{len(feat)}因子 | {ic_str} | OOS Sharpe={m['sharpe']:+.2f} "
            f"PnL={m['total_pnl']:+.4f} | 本轮 {_fmt_dur(win_el)} "
            f"(训练 {_fmt_dur(t_train)} / 重放 {_fmt_dur(t_replay)}, "
            f"累计 {_fmt_dur(time.time() - t_wf0)}, 预计剩余 {_fmt_dur(eta)})"
        )
        if progress_cb is not None:
            try:
                progress_cb(
                    phase="windows",
                    current=win_id + 1,
                    total=n_windows_total,
                    message=(
                        f"窗口 {win_id + 1}/{n_windows_total}: 测试"
                        f"{te_dates[0].astype('datetime64[D]')}~"
                        f"{te_dates[-1].astype('datetime64[D]')} | {ic_str} | "
                        f"OOS Sharpe={m['sharpe']:+.2f}"
                    ),
                    date=str(pd.Timestamp(te_dates[0]).date()),
                    pred_ic=pred_ic,
                )
            except Exception:
                # progress 永远不能影响主流程
                pass
        start += step
        win_id += 1

    if not oos_frames:
        raise ValueError("数据不足以构成任何滚动窗口，请减小 TRAIN_WINDOW/TEST_WINDOW")

    # 拼接所有窗口的连续 OOS：用汇总 score_cache 对整个 OOS 区间跑一遍每日重放
    # （feat_cols 仅参与指纹计算；缓存分覆盖各窗口各自的因子集，指纹随之现算）。
    logger.info(
        f"全部 {win_id} 个窗口完成（{_fmt_dur(time.time() - t_wf0)}），" "开始拼接 OOS 整体重放…"
    )
    if progress_cb is not None:
        try:
            progress_cb(
                phase="final_replay",
                current=0,
                total=1,
                message=f"拼接 {win_id} 个窗口的 OOS 整体重放…",
            )
        except Exception:
            pass
    t_replay0 = time.time()
    oos = pd.concat(oos_frames, ignore_index=True)
    replay = _replay_with_scores(
        panel, all_factor_cols, combined_cache, first_te_start, last_te_end, replay_params
    )
    daily = replay["daily"]
    cash0 = replay["meta"]["initial_cash"]
    oos_metrics = daily_replay_metrics(daily, cash0)
    ic_mean, icir = _pred_ic(oos)
    oos_metrics["pred_ic"] = ic_mean
    oos_metrics["pred_icir"] = icir
    oos_metrics["mean_price_rmse"] = float("nan")
    logger.info(
        f"拼接 OOS 重放完成（{_fmt_dur(time.time() - t_replay0)}）："
        f"Sharpe={oos_metrics['sharpe']:+.2f} PnL={oos_metrics['total_pnl']:+.4f} "
        f"PredIC={ic_mean:+.4f}"
    )
    if progress_cb is not None:
        try:
            progress_cb(
                phase="final_replay",
                current=1,
                total=1,
                message="拼接 OOS 重放完成",
                pred_ic=ic_mean,
            )
        except Exception:
            pass
    oos_res = {
        "curve": _curve_from_daily(daily, cash0),
        "metrics": oos_metrics,
        "daily": daily,
        "trades": replay["trades"],
    }

    wm_df = pd.DataFrame(window_metrics)
    pos_ratios = _window_pos_ratios(wm_df)
    result = {
        "oos_result": oos_res,
        "window_metrics": wm_df,
        "n_windows": win_id,
        "oos": oos,
        "select_features": bool(select_features),
        "strategy": "long_topk",  # 引擎唯一有效口径（见形参说明）
        # 「为正窗口占比」门禁信号（Sharpe>0 / PredIC>0 窗口占比），meta.json 同名字段
        **pos_ratios,
    }
    if save:
        outdir = save_walkforward_result(result)
        if verbose:
            logger.info(f"walkforward 结果 -> {outdir}")
    return result


def with_ids(df: pd.DataFrame) -> pd.DataFrame:
    """给命名因子面板补 date_id/time_id/symbol_id（仅 scripts/archive/** 旧调用方使用）。"""
    out = df.copy()
    out[C.DATE_COL] = out["date"].rank(method="dense").astype(int)
    out[C.TIME_COL] = 0
    out[C.SYMBOL_COL] = out["symbol"].astype("category").cat.codes
    return out


# --- 向后兼容别名（scripts/archive/** 等旧调用方仍引用私有名）---
_train_predict = train_predict  # 向后兼容别名
_with_ids = with_ids  # 向后兼容别名
_cross_sectional_norm = cross_sectional_norm  # 向后兼容别名
_linear_fit = linear_fit  # 向后兼容别名
_linear_score = linear_score  # 向后兼容别名
_gbm_l2_predict = gbm_l2_predict  # 向后兼容别名
_per_date_rank = per_date_rank  # 向后兼容别名
_train_predict_multi_ret_weighted = train_predict_multi_ret_weighted  # 向后兼容别名
