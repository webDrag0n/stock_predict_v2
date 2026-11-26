"""因子草稿：校验与落盘（纯函数，前端与脚本共用，不依赖 dash）。

公式校验直接调用生产解析器 data.factors.parser（validate_formula / extract_dependencies），
不再复制一份 AST 白名单：旧前端的复刻版与解析器各自演化过，前端放行的公式生产侧可能拒绝，
反过来也一样。tests/test_factor_draft.py 断言库里现有的每个公式都能通过本校验。

草稿一律 enabled=False 写入 factor_library/custom.yaml。注意两点（均已实测）：
  1. 改动 factor_library/*.yaml 会改变因子面板缓存的签名（get_factor_library_hash 是签名的一部分，
     与 USE_FORMULA_FACTORS 是否开启无关），下一次读取面板会触发全量重建。
  2. 公式因子加载器不看 enabled：只要 USE_FORMULA_FACTORS=1，草稿也会被计算并进入面板。
     所以「enabled=False」不是安全开关，真正的开关是 USE_FORMULA_FACTORS 与 FORMULA_FACTOR_SUBSET。
"""

from __future__ import annotations

import os
import re

import yaml

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# 面板元列 / 目标列，禁止作为因子 id
RESERVED_IDS = frozenset(
    {"date", "symbol", "weight", "date_id", "time_id", "symbol_id", "return_1d"}
)


def normalize_expected_sign(v):
    """'+1'/'-1'/1/-1/''/'未知' → 1 / -1 / None；其他 → ValueError。"""
    if v is None or str(v).strip() in ("", "unknown", "未知", "None"):
        return None
    s = str(v).strip()
    if s in ("+1", "1", "1.0"):
        return 1
    if s in ("-1", "-1.0"):
        return -1
    raise ValueError(f"expected_sign 只能是 +1 / -1 / 未知，收到 {v!r}")


def check_formula(formula: str) -> tuple[list[str], set[str]]:
    """(错误列表, 依赖列集合)。错误来自生产解析器的 AST 白名单。"""
    from data.factors.parser import FormulaValidationError, extract_dependencies, validate_formula

    if not formula or not str(formula).strip():
        return ["公式不能为空"], set()
    try:
        validate_formula(str(formula))
    except FormulaValidationError as exc:
        return [str(exc)], set()
    return [], set(extract_dependencies(str(formula)))


def validate_draft(
    fid,
    name,
    category,
    formula,
    tags="",
    expected_sign=None,
    description="",
    existing_ids=frozenset(),
    known_columns=frozenset(),
) -> dict:
    """返回 {errors, warnings, deps, record}。errors 非空时不应保存。

    字段规则与 src/data/factors/schema.py::FactorDefinition 对齐：id/formula 必填，
    id 唯一且合法，公式过 AST 白名单。依赖列不在 known_columns 里只给警告（可能是其他公式因子）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    fid = (fid or "").strip()
    name = (name or "").strip()
    category = (category or "").strip() or "custom"
    formula = (formula or "").strip()
    description = (description or "").strip()

    if not fid:
        errors.append("id 不能为空")
    elif not _ID_RE.match(fid):
        errors.append("id 须为合法列名：字母/下划线开头，仅含字母数字下划线，≤64 字符")
    elif fid in RESERVED_IDS or fid.startswith("responder_"):
        errors.append(f"id {fid!r} 与面板保留列冲突")
    elif fid in set(existing_ids):
        errors.append(f"id {fid!r} 已存在（因子库或面板原生因子）")
    if not name:
        errors.append("name 不能为空")

    ferrs, deps = check_formula(formula)
    errors.extend(ferrs)
    known = set(known_columns)
    for d in sorted(deps):
        if known and d not in known:
            warnings.append(
                f"依赖列 {d!r} 不在行情缓存与面板的已知列中：若它不是其他公式因子，"
                "这个因子在计算时会被跳过"
            )
    try:
        sign = normalize_expected_sign(expected_sign)
    except ValueError as exc:
        errors.append(str(exc))
        sign = None

    tag_list = [t.strip() for t in re.split(r"[,，\s]+", tags or "") if t.strip()]
    record = {
        "id": fid,
        "name": name or fid,
        "formula": formula,
        "description": description,
        "category": category,
        "source": "custom",
        "tags": tag_list,
        "enabled": False,
        "expected_sign": sign,
        "version": 1,
        "author": "webapp",
        "created": "",
        "explicit_deps": [],
    }
    return {"errors": errors, "warnings": warnings, "deps": sorted(deps), "record": record}


def save_draft(record: dict, yaml_path: str, existing_ids=frozenset()) -> None:
    """把草稿追加写入 custom.yaml（保持 factors: [...] 结构，原子写）。

    id 已存在于本文件或 existing_ids（其他库文件里的因子）→ ValueError：
    加载器遇到重复 id 只会静默跳过后来者，用户会以为保存成功却从未生效。
    重写会丢失原文件里的注释。
    """
    data: dict = {"factors": []}
    if os.path.exists(yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is not None:
            # `factors:`（后面没写任何东西）会被 YAML 读成 None，这是合法的空库，不是结构异常
            if not isinstance(loaded, dict) or not isinstance(loaded.get("factors") or [], list):
                raise ValueError(f"{yaml_path} 结构异常：顶层应为含 factors 列表的字典")
            data = loaded
        data.setdefault("factors", [])
        if data["factors"] is None:
            data["factors"] = []
    taken = {r.get("id") for r in data["factors"] if isinstance(r, dict)} | set(existing_ids)
    if record["id"] in taken:
        raise ValueError(f"草稿 id 重复：{record['id']} 已存在于因子库")
    data["factors"].append(record)
    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
    tmp = f"{yaml_path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp, yaml_path)
