"""Dependency graph builder and topological sorter for formula factors.

Formula factors declare their column dependencies (auto-extracted from the
formula AST plus any explicit overrides).  From this we build a DAG where each
node is a factor that produces one column and consumes zero or more columns.

The topological sort respects inter-formula dependencies and also rejects
factors whose dependencies cannot be satisfied by the native family output
columns.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from .schema import FactorDefinition
from .parser import extract_dependencies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CycleError(Exception):
    """Circular dependency detected among formula factors."""


class MissingDependencyError(Exception):
    """A factor requires a column that is neither in the base DataFrame nor
    produced by any formula factor."""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class FactorNode:
    """A node in the factor dependency graph."""

    definition: FactorDefinition
    produces: str  # column name this factor outputs (= definition.id)
    depends_on: set[str]  # column names this factor needs as input
    resolved: bool = False  # True when all deps will be available at eval time

    def __repr__(self):
        return f"FactorNode({self.produces}, deps={sorted(self.depends_on)})"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_dependency_graph(
    definitions: list[FactorDefinition],
    available_columns: set[str],
) -> list[FactorNode]:
    """Build the dependency graph for a list of factor definitions.

    Parameters
    ----------
    definitions:
        Factor definitions loaded from YAML.
    available_columns:
        Columns already present in the DataFrame (from raw bars + native
        factor families).  Dependencies that are in this set are "satisfied"
        and do NOT need to be produced by another formula factor.

    Returns
    -------
    List of ``FactorNode`` objects, each representing one factor.
    """
    nodes: list[FactorNode] = []

    for fd in definitions:
        # Auto-extract dependencies from the formula AST
        auto_deps = extract_dependencies(fd.formula)

        # Merge explicit overrides
        all_deps = auto_deps | set(fd.explicit_deps)

        # Classify: "resolved" deps are already available
        remaining = all_deps - available_columns

        node = FactorNode(
            definition=fd,
            produces=fd.id,
            depends_on=remaining,
            resolved=len(remaining) == 0,
        )
        nodes.append(node)

    return nodes


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------


def topological_sort(nodes: list[FactorNode]) -> list[FactorNode]:
    """Sort ``FactorNode`` objects into a valid execution order.

    Uses Kahn's algorithm (BFS-based, O(V+E)).  Factors whose dependencies are
    all satisfied come first.  After each factor is "executed" (emitted), any
    factor that depends on its output becomes newly resolved.

    Raises
    ------
    CycleError
        If a circular dependency is detected.
    """
    # Build lookup: column → producing FactorNode
    col_to_node: dict[str, FactorNode] = {
        node.produces: node for node in nodes
    }

    # Build reverse edges: column → list of nodes that depend on it
    reverse: dict[str, list[FactorNode]] = {}
    for node in nodes:
        for dep in node.depends_on:
            reverse.setdefault(dep, []).append(node)

    # In-degree: number of unsatisfied dependencies
    in_degree: dict[str, int] = {
        node.produces: len(node.depends_on) for node in nodes
    }

    # Queue: nodes with in_degree == 0 (all deps satisfied)
    queue = deque(
        [node for node in nodes if in_degree[node.produces] == 0]
    )

    sorted_nodes: list[FactorNode] = []

    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)

        # All nodes that depend on this node get their in_degree reduced
        for dependent in reverse.get(node.produces, []):
            in_degree[dependent.produces] -= 1
            if in_degree[dependent.produces] == 0:
                queue.append(dependent)

    if len(sorted_nodes) != len(nodes):
        remaining = [n.produces for n in nodes if n not in sorted_nodes]
        raise CycleError(
            f"公式因子之间存在循环依赖: {remaining}. "
            f"请检查因子之间的引用关系."
        )

    return sorted_nodes


# ---------------------------------------------------------------------------
# Full resolution pipeline
# ---------------------------------------------------------------------------


def resolve_execution_order(
    definitions: list[FactorDefinition],
    available_columns: set[str],
) -> list[FactorDefinition]:
    """Full pipeline: build graph → reject unsolvable → sort → return.

    Factors whose dependencies cannot be satisfied by any column (neither
    base/native columns nor other formula factors) are skipped with a warning.

    Parameters
    ----------
    definitions:
        Loaded factor definitions.
    available_columns:
        Set of column names already present in the DataFrame before formula
        factors run.

    Returns
    -------
    Topologically-sorted list of ``FactorDefinition`` ready for evaluation.
    """
    # 1. Build graph
    nodes = build_dependency_graph(definitions, available_columns)

    # 2. Check for unsolvable dependencies
    #    All columns produced by formula factors
    produced = {n.produces for n in nodes}

    valid_nodes: list[FactorNode] = []
    for node in nodes:
        unresolved = node.depends_on - produced
        if unresolved:
            logger.warning(
                "跳过因子 %s: 依赖列 %s 不存在, 且没有其他公式因子产生它们",
                node.produces, sorted(unresolved),
            )
            # Also remove from produced set so downstream factors that depend
            # on this one get caught
            produced.discard(node.produces)
        else:
            valid_nodes.append(node)

    if not valid_nodes:
        logger.warning("没有有效的公式因子可执行")
        return []

    # 3. Topological sort
    try:
        sorted_nodes = topological_sort(valid_nodes)
    except CycleError as exc:
        logger.error("公式因子排序失败: %s", exc)
        return []

    # 4. Return definitions in order
    ordered = [n.definition for n in sorted_nodes]
    if ordered:
        logger.info(
            "公式因子执行顺序 (%d 个): %s",
            len(ordered),
            " → ".join(fd.id for fd in ordered),
        )
    return ordered
