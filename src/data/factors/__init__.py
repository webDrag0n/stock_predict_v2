"""Unified factor library for the A-share prediction system.

This package aggregates all engineered factor families and exposes a single
dispatcher, ``compute_all_factors()``, that ``load_factor_panel()`` calls. Each
family lives in its own module and can be toggled independently via config
switches.

Families:
  - technical    : base OHLCV technical + A-share-specific factors
  - orthogonal   : anomaly factors orthogonal to momentum/value/volatility
  - alpha158     : Microsoft Qlib Alpha158 OHLCV families
  - gtja191      : curated subset of Guotai Junan 191 alpha factors
  - formula      : YAML-based formula factors (researcher-facing, no code)

Adding a new family:
  1. Create a module with ``compute_<family>_features(df) -> pd.DataFrame``.
  2. Import it here and add it to ``FACTORY``.
  3. Add ``USE_<FAMILY>_FACTORS`` to config.py.
"""
from __future__ import annotations

import pandas as pd

import config as C

from .technical import compute_technical_features
from .orthogonal import compute_orthogonal_factors
from .alpha158 import compute_alpha158_factors
from .gtja191 import compute_gtja191_factors
from .formula_factors import compute_formula_factors, get_factor_library_hash


FACTORY: dict[str, tuple] = {
    "technical": (compute_technical_features, "USE_TECHNICAL_FEATURES"),
    "orthogonal": (compute_orthogonal_factors, "USE_ORTHOGONAL_FACTORS"),
    "alpha158": (compute_alpha158_factors, "USE_ALPHA158_FACTORS"),
    "gtja191": (compute_gtja191_factors, "USE_GTJA191_FACTORS"),
    "formula": (compute_formula_factors, "USE_FORMULA_FACTORS"),
}


def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Run all enabled factor families and return the augmented DataFrame.

    The order is intentionally fixed: technical → orthogonal → alpha158 → gtja191.
    Later families may reuse columns created by earlier families (e.g. return_1d).

    To keep whole-market panels (~6M rows) from blowing up memory, each family is
    followed by a numeric downcast. Factor values do not need float64 precision.
    """
    # Local import avoids circular dependency: src.data imports this module.
    from .. import reduce_mem_usage

    for name, (fn, cfg_key) in FACTORY.items():
        if getattr(C, cfg_key, True):
            if name == "gtja191":
                subset = getattr(C, "GTJA191_ALPHA_SUBSET", None)
                if subset:
                    subset = [f"alpha{x.strip().zfill(3)}"
                              for x in str(subset).split(",") if x.strip()]
                df = fn(df, subset=subset) if subset else fn(df)
            else:
                df = fn(df)
            df = reduce_mem_usage(df, float16_as32=True)
    return df


__all__ = [
    "compute_all_factors",
    "compute_technical_features",
    "compute_orthogonal_factors",
    "compute_alpha158_factors",
    "compute_gtja191_factors",
    "compute_formula_factors",
]
