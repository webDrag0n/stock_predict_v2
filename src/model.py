"""Ensemble model: purged date-based CV, following the notebook's approach.

The notebook trained LGB + XGB + CatBoost per fold and averaged them. Here LGB
is always available; XGB and CatBoost are used only if installed. Each fold
trains on dates where (date_index %% N_FOLD != fold), so folds are disjoint by
date — no intra-day leakage across the train/validation boundary.

Scoring uses the competition's weighted R2 (a.k.a. weighted zero-mean R2):
    R2 = 1 - sum(w * (y - yhat)^2) / sum(w * y^2)
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import config as C
from features import get_feature_matrix

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cbt
    HAS_CBT = True
except ImportError:
    HAS_CBT = False


def weighted_r2(y_true, y_pred, weight) -> float:
    """Competition metric: 1 - wMSE / w*y^2. Higher is better; can be negative."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    num = np.sum(weight * (y_true - y_pred) ** 2)
    den = np.sum(weight * y_true ** 2) + 1e-38
    return float(1.0 - num / den)


def _build_models() -> dict:
    """Instantiate every available model with CPU-safe defaults."""
    models = {}
    if HAS_LGB:
        models["lgb"] = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, objective="l2",
            random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    if HAS_XGB:
        models["xgb"] = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            objective="reg:squarederror", random_state=C.RANDOM_STATE, n_jobs=-1,
        )
    if HAS_CBT:
        models["cbt"] = cbt.CatBoostRegressor(
            iterations=100, learning_rate=0.1, depth=5,
            loss_function="RMSE", random_seed=C.RANDOM_STATE, verbose=0,
        )
    return models


class EnsembleModel:
    """Average of per-fold, per-algorithm gradient-boosted regressors."""

    def __init__(self, n_fold: int = C.N_FOLD,
                 feature_cols: list[str] | None = None):
        self.n_fold = n_fold
        self.feature_cols = feature_cols   # None = use all C.FEATURE_NAMES
        self.models: list = []
        self.algos_used: list[str] = []

    def fit(self, df: pd.DataFrame, train_dates: np.ndarray,
            valid_dates: np.ndarray, verbose: bool = True) -> "EnsembleModel":
        if not (HAS_LGB or HAS_XGB or HAS_CBT):
            raise RuntimeError("No gradient boosting library available "
                               "(need lightgbm, xgboost, or catboost).")
        fc = self.feature_cols
        Xv = get_feature_matrix(df.loc[df[C.DATE_COL].isin(valid_dates)], feature_cols=fc)
        yv = df[C.TARGET].loc[df[C.DATE_COL].isin(valid_dates)].to_numpy()
        wv = df[C.WEIGHT].loc[df[C.DATE_COL].isin(valid_dates)].to_numpy()

        train_dates = np.asarray(sorted(train_dates))
        n = len(train_dates)
        embargo = C.EMBARGO_DAYS
        # Contiguous-block purged CV: split the (time-sorted) training dates into
        # N_FOLD consecutive blocks; each fold holds out one block and trains on
        # the rest MINUS an embargo gap on both sides of the held-out block, so
        # the forward-looking labels (close.shift(-FORWARD_WINDOW)) of the train
        # set never overlap the held-out block's dates.
        bounds = np.linspace(0, n, self.n_fold + 1).astype(int)
        for fold in range(self.n_fold):
            lo, hi = bounds[fold], bounds[fold + 1]       # held-out block [lo, hi)
            hold = train_dates[lo:hi]
            # train = everything outside [lo-embargo, hi+embargo)
            keep_idx = np.concatenate([
                np.arange(0, max(lo - embargo, 0)),
                np.arange(min(hi + embargo, n), n),
            ])
            sel = train_dates[keep_idx]
            if len(sel) == 0 or len(hold) == 0:
                continue
            mask = df[C.DATE_COL].isin(sel)
            Xt = get_feature_matrix(df.loc[mask], feature_cols=fc)
            yt = df[C.TARGET].loc[mask].to_numpy()
            wt = df[C.WEIGHT].loc[mask].to_numpy()

            for name, model in _build_models().items():
                model.fit(Xt, yt, sample_weight=wt)
                self.models.append(model)
                if name not in self.algos_used:
                    self.algos_used.append(name)
                if verbose:
                    r2 = weighted_r2(yv, model.predict(Xv), wv)
                    print(f"  fold {fold} {name}: valid weighted R2 = {r2:+.5f}")

        if verbose:
            r2 = weighted_r2(yv, self.predict(df.loc[df[C.DATE_COL].isin(valid_dates)]), wv)
            n_feat = len(fc) if fc else len(C.FEATURE_NAMES)
            print(f"Ensemble valid weighted R2 = {r2:+.5f} "
                  f"(algos: {', '.join(self.algos_used)}, {len(self.models)} models, "
                  f"{n_feat} features)")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = get_feature_matrix(df, feature_cols=self.feature_cols)
        preds = np.mean([m.predict(X) for m in self.models], axis=0)
        return np.asarray(preds, dtype=np.float64).ravel()

    def save(self, path: str = None) -> str:
        path = path or os.path.join(C.MODEL_DIR, "ensemble.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"models": self.models, "algos": self.algos_used,
                     "n_fold": self.n_fold, "feature_cols": self.feature_cols}, path)
        return path

    @classmethod
    def load(cls, path: str = None) -> "EnsembleModel":
        path = path or os.path.join(C.MODEL_DIR, "ensemble.joblib")
        blob = joblib.load(path)
        obj = cls(n_fold=blob["n_fold"],
                  feature_cols=blob.get("feature_cols"))
        obj.models = blob["models"]
        obj.algos_used = blob["algos"]
        return obj
        obj = cls(n_fold=blob["n_fold"])
        obj.models = blob["models"]
        obj.algos_used = blob["algos"]
        return obj
