"""Safe formula evaluator for the formula-factor pipeline.

Works with ``pd.DataFrame`` that uses a ``(symbol, date)`` MultiIndex (the
convention expected by the DSL primitives in ``_base.py``).

Formulas are compiled once via ``compile()`` and evaluated against a restricted
namespace: zero Python builtins, only column Series and DSL primitive callables.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .schema import FactorDefinition
from .parser import validate_formula
from ._base import (
    # All DSL primitives are exposed in the evaluation context
    ts_delay,
    ts_delta,
    ts_sum,
    ts_mean,
    ts_std,
    ts_max,
    ts_min,
    ts_skew,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_rank,
    cs_rank,
    cs_zscore,
    signed_power,
    safe_div,
    ts_quantile,
    ts_reg_resid,
    ts_reg_r2,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSL primitive registry (name → callable)
# ---------------------------------------------------------------------------

_DSL_REGISTRY: dict[str, callable] = {
    "ts_delay": ts_delay,
    "ts_delta": ts_delta,
    "ts_sum": ts_sum,
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_max": ts_max,
    "ts_min": ts_min,
    "ts_skew": ts_skew,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "ts_corr": ts_corr,
    "ts_rank": ts_rank,
    "cs_rank": cs_rank,
    "cs_zscore": cs_zscore,
    "signed_power": signed_power,
    "safe_div": safe_div,
    "ts_quantile": ts_quantile,
    "ts_reg_resid": ts_reg_resid,
    "ts_reg_r2": ts_reg_r2,
}

# Safe math builtins mapped to numpy functions
_SAFE_MATH: dict[str, callable] = {
    "log": np.log,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "sign": np.sign,
    "exp": np.exp,
    "log2": lambda x: np.log2(np.abs(x) + 1e-12),
    "log10": lambda x: np.log10(np.abs(x) + 1e-12),
    "round": np.round,
    "min": np.minimum,  # element-wise (needs two args)
    "max": np.maximum,  # element-wise (needs two args)
}


class FormulaEvalError(Exception):
    """Raised when a formula evaluation fails at runtime."""


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_context(
    df_indexed: pd.DataFrame,
    extra_columns: dict[str, pd.Series] | None = None,
) -> dict:
    """Build the evaluation context dict from a MultiIndex DataFrame.

    Parameters
    ----------
    df_indexed:
        DataFrame with a ``(symbol, date)`` MultiIndex.  Every column becomes
        a named ``pd.Series`` in the context.
    extra_columns:
        Optional additional column-name → Series mappings (e.g. intermediate
        columns not yet in the DataFrame).

    Returns
    -------
    dict mapping names to either ``pd.Series`` (columns) or callables (DSL
    primitives and math functions).
    """
    context: dict = {}

    # DataFrame columns as Series
    for col in df_indexed.columns:
        context[col] = df_indexed[col]

    # Extra columns (from previously computed formula factors)
    if extra_columns:
        context.update(extra_columns)

    # DSL primitives
    context.update(_DSL_REGISTRY)

    # Safe math
    context.update(_SAFE_MATH)

    return context


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_formula(expr: str, context: dict) -> pd.Series:
    """Validate, compile, and evaluate a single formula.

    Parameters
    ----------
    expr:
        Python expression string (e.g. ``"close / ts_delay(close, 5) - 1"``).
    context:
        Context dict from ``build_context()``.

    Returns
    -------
    ``pd.Series`` with index matching the context columns.

    Raises
    ------
    FormulaEvalError
        On runtime failure (missing column, shape mismatch, NaN, division by
        zero that propagates to all values, etc.).
    """
    # 1. Safety check (AST whitelist)
    validate_formula(expr)

    # 2. Compile
    try:
        code = compile(expr, "<formula>", "eval")
    except SyntaxError as exc:
        raise FormulaEvalError(f"公式编译失败: {expr[:80]} — {exc}") from exc

    # 3. Execute in sandbox
    try:
        result = eval(code, {"__builtins__": {}}, context)
    except Exception as exc:
        raise FormulaEvalError(
            f"公式执行失败: {expr[:80]} — {type(exc).__name__}: {exc}"
        ) from exc

    # 4. Result must be Series-like
    if result is None:
        raise FormulaEvalError(f"公式返回 None: {expr[:80]}")

    if isinstance(result, (bool, int, float, np.integer, np.floating, np.bool_)):
        # Scalar broadcast — pick an arbitrary Series from context for the index
        index = None
        for v in context.values():
            if isinstance(v, pd.Series):
                index = v.index
                break
        if index is None:
            raise FormulaEvalError(f"没有 Series 可用于索引对齐: {expr[:80]}")
        return pd.Series(np.full(len(index), float(result)), index=index, dtype=np.float32)

    if isinstance(result, np.ndarray):
        # numpy array — try to find a matching Series for index
        index = None
        for v in context.values():
            if isinstance(v, pd.Series) and len(v) == len(result):
                index = v.index
                break
        if index is None:
            raise FormulaEvalError(f"没有 Series 可用于索引对齐: {expr[:80]}")
        return pd.Series(result, index=index, dtype=np.float32)

    if isinstance(result, pd.Series):
        return result.astype(np.float32)

    raise FormulaEvalError(
        f"公式返回了不支持的类型 {type(result).__name__}: {expr[:80]}"
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_all(
    definitions: list[FactorDefinition],
    df: pd.DataFrame,
    progress: bool = False,
) -> pd.DataFrame:
    """Evaluate a topologically-sorted list of factor definitions.

    For each definition: build context from current DataFrame columns + any
    previously computed formula columns, evaluate the formula, and add the
    result as a new column.

    Parameters
    ----------
    definitions:
        Pre-sorted list of ``FactorDefinition`` objects (output from
        ``dependency.resolve_execution_order``).
    df:
        DataFrame with a flat index (``symbol`` + ``date`` columns) — the
        same shape as what native families operate on.

    Returns
    -------
    The augmented DataFrame (in-place modification + return).
    """
    if not definitions:
        return df

    # Convert to MultiIndex for DSL primitives
    needs_reset = True
    if isinstance(df.index, pd.MultiIndex) and df.index.names == ["symbol", "date"]:
        indexed = df
        needs_reset = False
    elif "symbol" in df.columns and "date" in df.columns:
        indexed = df.set_index(["symbol", "date"])
    else:
        raise ValueError("DataFrame 必须有 symbol 和 date 列, 或 (symbol, date) MultiIndex")

    # Track formula columns separately (not yet in df columns, but available as
    # dependencies for later formula factors)
    formula_cache: dict[str, pd.Series] = {}

    for i, fd in enumerate(definitions):
        if progress and (i % 10 == 0 or i == len(definitions) - 1):
            logger.info("  公式因子 %d/%d: %s", i + 1, len(definitions), fd.id)

        try:
            context = build_context(indexed, extra_columns=formula_cache)
            result = evaluate_formula(fd.formula, context)
        except FormulaEvalError:
            logger.error("公式因子 %s 执行失败", fd.id, exc_info=True)
            continue
        except Exception:
            logger.error("公式因子 %s 执行失败", fd.id, exc_info=True)
            continue

        # Store for both immediate DataFrame access and later formula factors
        result.name = fd.id
        formula_cache[fd.id] = result

    # Merge formula columns back to the flat DataFrame
    if formula_cache:
        formula_df = pd.DataFrame(formula_cache)
        if needs_reset:
            # reset_index so we can merge on [symbol, date]
            formula_df = formula_df.reset_index()
            df = df.merge(formula_df, on=["symbol", "date"], how="left")
        else:
            for col_name, series in formula_cache.items():
                indexed[col_name] = series

    if needs_reset and isinstance(indexed, pd.DataFrame):
        # indexed was a view; reset df to flat
        pass  # df is already flat from the merge

    return df
