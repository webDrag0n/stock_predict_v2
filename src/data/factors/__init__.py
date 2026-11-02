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

# 所有内置因子族最长的逐股滚动窗口（交易日）：orthogonal.high_52w_prox 的
# rolling(252)；gtja191 最深嵌套链 ts_mean(v,150)→ts_corr(...,15) ≈ 164；
# alpha158/technical ≤ 60。MACD 的 ewm(span=26) 是无限记忆，但截断到 N 天
# 的暖机误差按 (1-2/27)^N 衰减，N≥292 时相对误差 ~1e-9（实测见
# tests/test_factor_panel_cache.py），远低于 float32 面板精度。
# load_factor_panel 的尾部增量重算以此确定原始数据回看窗口。
MAX_ROLLING_WINDOW = 252

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
                    subset = [
                        f"alpha{x.strip().zfill(3)}" for x in str(subset).split(",") if x.strip()
                    ]
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
    "get_factor_library_hash",
    "MAX_ROLLING_WINDOW",
]
