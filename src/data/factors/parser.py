"""AST-based formula validation and dependency extraction.

Uses Python's ``ast`` module to parse formula strings, validate that they
contain only safe constructs (no imports, no attribute access, no
comprehensions, etc.), and extract the set of column references each formula
depends on.
"""

from __future__ import annotations

import ast
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known function names (DSL primitives + safe math builtins)
# ---------------------------------------------------------------------------

# From src/data/factors/_base.py
DSL_PRIMITIVES = frozenset({
    "ts_delay", "ts_delta", "ts_sum", "ts_mean", "ts_std",
    "ts_max", "ts_min", "ts_skew", "ts_argmax", "ts_argmin",
    "ts_corr", "ts_rank", "cs_rank", "cs_zscore",
    "signed_power", "safe_div", "ts_quantile",
    "ts_reg_resid", "ts_reg_r2",
})

# Safe math builtins (mapped to numpy equivalents at runtime)
SAFE_BUILTINS = frozenset({
    "log", "abs", "sqrt", "sign",
    "exp", "log2", "log10", "round",
    "min", "max",
})

# All names that are functions, NOT column references
_FUNCTION_NAMES = DSL_PRIMITIVES | SAFE_BUILTINS

# ---------------------------------------------------------------------------
# AST node whitelists
# ---------------------------------------------------------------------------

# Node types that are always safe
_SAFE_NODES: set[type] = {
    # Module / Expression wrapper
    ast.Module,

    # Expression statement (wraps a bare expression in a Module body)
    ast.Expr,

    # Literals
    ast.Constant,

    # Variable reference (Name(id=..., ctx=Load))
    ast.Name,
    ast.Load,

    # Binary operators
    ast.BinOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,

    # Unary operators
    ast.UnaryOp,
    ast.UAdd, ast.USub, ast.Not, ast.Invert,

    # Boolean operators
    ast.BoolOp,
    ast.And, ast.Or,

    # Comparison
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,

    # Ternary
    ast.IfExp,

    # Function call (only to registered names)
    ast.Call,

    # Parenthesised / tuple
    ast.Tuple,
    ast.List,

    # Slicing (for literal numbers in expressions, not for subscript)
    ast.Slice,
}

# Node types that are EXPLICITLY FORBIDDEN (security-relevant)
_FORBIDDEN_NODES = {
    ast.Attribute,       # obj.method() / obj.attr — could access os.system etc.
    ast.Subscript,       # x[0] / d["key"] — could access globals
    ast.Lambda,          # lambda x: ...
    ast.DictComp,        # {k:v for ...}
    ast.SetComp,         # {x for ...}
    ast.ListComp,        # [x for ...]
    ast.GeneratorExp,    # (x for ...)
    ast.Import,           # import ...
    ast.ImportFrom,       # from ... import ...
    ast.Assign,           # x = ...
    ast.AugAssign,        # x += ...
    ast.AnnAssign,        # x: int = ...
    ast.NamedExpr,        # x := ... (walrus)
    ast.Starred,          # *args
    ast.Await,            # await ...
    ast.Yield,            # yield ...
    ast.YieldFrom,        # yield from ...
    ast.Global,           # global x
    ast.Nonlocal,         # nonlocal x
    ast.Delete,           # del x
    ast.Raise,            # raise ...
    ast.Assert,           # assert ...
    ast.Pass,             # pass
    ast.Break,            # break
    ast.Continue,         # continue
    ast.Return,           # return ...
    ast.ClassDef,         # class ...
    ast.FunctionDef,      # def ...
    ast.AsyncFunctionDef, # async def ...
    ast.For,              # for ...
    ast.AsyncFor,         # async for ...
    ast.While,            # while ...
    ast.If,               # if stmt (IfExp is the ternary, which IS safe)
    ast.Try,              # try ...
    ast.With,             # with ...
    ast.AsyncWith,        # async with ...
    ast.Match,            # match ... (Python 3.10+)
}


class FormulaValidationError(Exception):
    """Raised when a formula string fails AST safety checks."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_formula(expr: str) -> None:
    """Raise ``FormulaValidationError`` if the formula uses unsafe constructs.

    The formula must be a pure expression — no statements, imports, attribute
    access, subscripts, lambdas, comprehensions, or assignments.  Only
    arithmetic, comparisons, function calls (to registered primitives), and
    ternary expressions are allowed.
    """
    try:
        tree = ast.parse(expr, mode="exec")
    except SyntaxError as exc:
        raise FormulaValidationError(f"公式语法错误: {exc}") from exc

    _check_node(tree)


def _check_node(node: ast.AST) -> None:
    """Recursively walk the AST and validate every node.

    Raises FormulaValidationError on the first forbidden construct.
    """
    node_type = type(node)

    # Forbidden check
    if node_type in _FORBIDDEN_NODES:
        raise FormulaValidationError(
            f"公式包含禁止的语法: {node_type.__name__} "
            f"(位置: line {getattr(node, 'lineno', '?')}, "
            f"col {getattr(node, 'col_offset', '?')})"
        )

    # Safe check — if it's in the whitelist, recurse children
    if node_type in _SAFE_NODES:
        # Extra check: Call must have a simple Name as func, not an attribute or subscript
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaValidationError(
                    "只能调用注册的原语函数, 不能调用对象方法或通过属性访问的函数 "
                    f"(位置: line {getattr(node, 'lineno', '?')})"
                )
            if node.func.id not in _FUNCTION_NAMES:
                raise FormulaValidationError(
                    f"未知函数: {node.func.id} — 只能使用注册的 DSL 原语和数学函数 "
                    f"(位置: line {getattr(node, 'lineno', '?')})"
                )
        # Recursively check all children
        for _field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                _check_node(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        _check_node(item)
        return

    # Unknown node — not in whitelist and not explicitly forbidden
    raise FormulaValidationError(
        f"公式包含不支持的语法节点: {node_type.__name__} "
        f"(位置: line {getattr(node, 'lineno', '?')})"
    )


# ---------------------------------------------------------------------------
# Dependency extraction
# ---------------------------------------------------------------------------


def extract_dependencies(expr: str) -> set[str]:
    """Return the set of column names the formula references.

    Walks the AST and collects all ``Name`` nodes that are NOT function-call
    targets (i.e. column references like ``close``, ``volume``) and NOT in the
    known-function-name set.

    Examples
    --------
    >>> extract_dependencies("adj_close / ts_delay(adj_close, 5) - 1")
    {'adj_close'}
    >>> extract_dependencies("-ts_corr(cs_rank(ts_delta(log(volume), 1)), cs_rank((close-open)/open), 6)")
    {'volume', 'close', 'open'}
    """
    try:
        tree = ast.parse(expr, mode="exec")
    except SyntaxError:
        return set()

    deps: set[str] = set()

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            # Walk arguments first
            for arg in node.args:
                _walk(arg)
            for kw in node.keywords:
                _walk(kw.value)
            # Do NOT walk the func Name — it's a function reference, not a column
            return

        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in _FUNCTION_NAMES:
                deps.add(node.id)
            return

        for _field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                _walk(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        _walk(item)

    _walk(tree)
    return deps


def extract_function_calls(expr: str) -> set[str]:
    """Return the set of DSL function names called in the formula.

    Useful for documentation / analysis only.
    """
    try:
        tree = ast.parse(expr, mode="exec")
    except SyntaxError:
        return set()

    funcs: set[str] = set()

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            funcs.add(node.func.id)
        for _field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, ast.AST):
                _walk(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        _walk(item)

    _walk(tree)
    return funcs


def extract_formula_metadata(expr: str) -> dict:
    """Convenience: validate, extract deps and functions, return as dict."""
    validate_formula(expr)
    return {
        "dependencies": sorted(extract_dependencies(expr)),
        "functions": sorted(extract_function_calls(expr)),
    }
