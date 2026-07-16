"""Factor definition schema and YAML loader.

Defines the FactorDefinition dataclass that researchers write in YAML, plus
the loader that scans the factor_library/ directory and returns validated
definitions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class FactorDefinition:
    """A single factor definition sourced from YAML.

    Fields match the YAML keys directly.  ``formula`` is a safe Python
    expression string (no imports, no attribute access) that will be
    evaluated against a restricted context of column Series and DSL
    primitives.
    """

    id: str  # unique column name
    name: str  # human-readable short label
    formula: str  # Python expression string
    description: str = ""
    category: str = "general"
    source: str = ""  # paper title, URL, or "custom"
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    expected_sign: float | None = None  # +1 or -1, or None if unknown
    version: int = 1
    author: str = ""
    created: str = ""
    # Explicit dependencies override auto-extracted ones.  Only needed when
    # the formula would not surface a dependency through the AST walk, e.g.
    # when using dynamic lookup.
    explicit_deps: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Factor id must be a non-empty string, got {self.id!r}")
        if not self.formula or not isinstance(self.formula, str):
            raise ValueError(f"Factor {self.id}: formula must be a non-empty string")
        # Normalise empty fields
        if not self.name:
            self.name = self.id


@dataclass
class LibraryManifest:
    """Optional top-level _manifest.yaml in the library directory."""

    schema_version: int = 1
    default_enabled: bool = True
    exclude_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _load_yaml_file(path: Path) -> list[dict]:
    """Load a single YAML file and return the list of factor dicts."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return []
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dict with a 'factors' key, got {type(data)}")
    factors = data.get("factors", [])
    if not isinstance(factors, list):
        raise ValueError(f"{path}: 'factors' must be a list, got {type(factors)}")
    return factors


def _load_manifest(library_dir: Path) -> LibraryManifest:
    """Load optional _manifest.yaml from the library directory."""
    manifest_path = library_dir / "_manifest.yaml"
    if not manifest_path.exists():
        return LibraryManifest()
    with open(manifest_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return LibraryManifest()
    return LibraryManifest(
        schema_version=int(data.get("schema_version", 1)),
        default_enabled=bool(data.get("default_enabled", True)),
        exclude_ids=[str(x) for x in data.get("exclude_ids", [])],
    )


def load_factor_definitions(
    library_dir: str | None = None,
    subset: list[str] | None = None,
) -> list[FactorDefinition]:
    """Walk ``library_dir`` and return all parsed ``FactorDefinition`` objects.

    Parameters
    ----------
    library_dir:
        Path to the factor_library/ directory.  If ``None``, reads the
        ``FACTOR_LIBRARY_DIR`` env / config value.
    subset:
        If provided, only load factors whose ``id`` is in this list.

    Returns
    -------
    List of ``FactorDefinition``, in file-system order (sorted by path).
    The caller should pass them through dependency resolution before
    evaluation.
    """
    if library_dir is None:
        # Avoid circular import at module level; config is loaded lazily.
        try:
            import config as C

            library_dir = getattr(C, "FACTOR_LIBRARY_DIR",
                                  os.path.join(os.path.expanduser("~"), "factor_library"))
        except Exception:
            library_dir = os.path.join(os.path.expanduser("~"), "factor_library")

    lib = Path(library_dir)
    if not lib.is_dir():
        logger.warning("因子库目录不存在: %s", lib)
        return []

    manifest = _load_manifest(lib)
    exclude = set(manifest.exclude_ids)

    yaml_files = sorted(lib.glob("*.yaml")) + sorted(lib.glob("*.yml"))
    # skip manifest files
    yaml_files = [p for p in yaml_files if p.name not in ("_manifest.yaml", "_manifest.yml")]

    definitions: list[FactorDefinition] = []
    seen_ids: set[str] = set()

    for path in yaml_files:
        try:
            raw_list = _load_yaml_file(path)
        except Exception:
            logger.warning("跳过无法解析的因子文件: %s", path, exc_info=True)
            continue

        for raw in raw_list:
            try:
                fd = FactorDefinition(**raw)
            except Exception:
                logger.warning("因子定义格式错误, 跳过: %s", raw.get("id", "?"),
                               exc_info=True)
                continue

            if fd.id in exclude:
                logger.info("因子 %s 在 _manifest.yaml 排除列表中, 跳过", fd.id)
                continue
            if subset is not None and fd.id not in subset:
                continue
            if not fd.enabled and not manifest.default_enabled:
                continue
            if fd.id in seen_ids:
                logger.error("因子 ID 重复: %s (文件 %s), 跳过", fd.id, path)
                continue

            seen_ids.add(fd.id)
            definitions.append(fd)

    logger.info("从 %s 加载了 %d 个公式因子定义", lib, len(definitions))
    return definitions
