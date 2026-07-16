"""Formula-based factor computer — bridge between YAML definitions and the
native factor pipeline.

Exposes ``compute_formula_factors(df)`` which matches the existing family
convention (``fn(df) → pd.DataFrame``), so it can be registered in
``FACTORY`` without changes to the dispatcher.
"""

from __future__ import annotations

import hashlib
import logging
import os

import pandas as pd

from .schema import load_factor_definitions
from .dependency import resolve_execution_order
from .evaluator import evaluate_all

logger = logging.getLogger(__name__)


def get_factor_library_hash(library_dir: str | None = None) -> str:
    """MD5 hash of all YAML files in the factor library.

    Used in the data-cache signature so that any change to formula definitions
    automatically invalidates the processed panel cache.
    """
    if library_dir is None:
        try:
            import config as C

            library_dir = getattr(
                C, "FACTOR_LIBRARY_DIR",
                os.path.join(os.path.expanduser("~"), "factor_library"),
            )
        except Exception:
            return "no_library"

    lib = library_dir
    if not os.path.isdir(lib):
        return "no_library"

    h = hashlib.md5(usedforsecurity=False)
    for path in sorted(os.listdir(lib)):
        if path.endswith((".yaml", ".yml")):
            full = os.path.join(lib, path)
            try:
                with open(full, "rb") as fh:
                    h.update(fh.read())
            except OSError:
                pass
    return h.hexdigest()[:12]


def compute_formula_factors(
    df: pd.DataFrame,
    library_dir: str | None = None,
    subset: list[str] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Compute all enabled formula-based factors and return the augmented DataFrame.

    This function signature matches the existing family convention so it can
    be added to ``FACTORY`` directly:

    ``"formula": (compute_formula_factors, "USE_FORMULA_FACTORS")``

    Parameters
    ----------
    df:
        DataFrame with at least ``symbol``, ``date`` and all native-family
        columns.  Formula factors run last in the pipeline, so every native
        column is available as a dependency.
    library_dir:
        Path to ``factor_library/``.  Defaults to ``FACTOR_LIBRARY_DIR``
        from config.
    subset:
        If provided, only compute factors whose ``id`` is in this list.
        Corresponds to ``FORMULA_FACTOR_SUBSET`` in config.

    Returns
    -------
    The augmented DataFrame (columns added in-place + returned).
    """
    if library_dir is None:
        try:
            import config as C

            library_dir = getattr(
                C, "FACTOR_LIBRARY_DIR",
                os.path.join(os.path.expanduser("~"), "factor_library"),
            )
        except Exception:
            library_dir = os.path.join(os.path.expanduser("~"), "factor_library")

    # 1. Load definitions from YAML
    definitions = load_factor_definitions(library_dir, subset=subset)
    if not definitions:
        return df

    # 2. Determine available columns
    available = set(df.columns)

    # 3. Resolve execution order (validates deps, sorts)
    ordered = resolve_execution_order(definitions, available)
    if not ordered:
        return df

    # 4. Evaluate all in order
    logger.info("计算 %d 个公式因子...", len(ordered))
    df = evaluate_all(ordered, df, progress=progress)

    return df
