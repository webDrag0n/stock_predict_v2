"""Feature handling: selection, NaN filling, optional cross-sectional normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def get_feature_matrix(df: pd.DataFrame, fill: float = 0.0,
                       feature_cols: list[str] | None = None) -> np.ndarray:
    """Extract the feature block as float32, filling NaNs with `fill`.

    feature_cols: optional subset of C.FEATURE_NAMES to use (for factor-selected
    models). None means use all C.FEATURE_NAMES.
    """
    cols = feature_cols if feature_cols else C.FEATURE_NAMES
    # only keep cols that actually exist in df
    cols = [c for c in cols if c in df.columns]
    X = df[cols].to_numpy(dtype=np.float32, copy=True)
    if fill is not None:
        np.nan_to_num(X, copy=False, nan=fill)
    return X


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Optional: rank each feature within (date_id, time_id) to a [-1, 1] score.

    Cross-sectional normalization is a common edge in market-neutral setups: it
    removes per-timestamp level effects and exposes relative signal. Disabled by
    default to keep the baseline faithful; call explicitly if desired.
    """
    df = df.copy()
    grp = df.groupby([C.DATE_COL, C.TIME_COL])
    for col in C.FEATURE_NAMES:
        r = grp[col].rank(pct=True)
        df[col] = (r - 0.5) * 2.0
    return df
