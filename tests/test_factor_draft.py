"""src/factor_draft.py：草稿校验与落盘。

校验直接调用生产解析器；这里的关键断言是「库里现有的每个公式都能通过」——旧前端复刻了一份
AST 白名单，两边各自演化后会出现前端放行、生产拒绝（或反过来）的分歧。
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

import factor_draft as fd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _library_formulas():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "factor_library", "*.yaml"))):
        if p.endswith("_manifest.yaml"):
            continue
        out += [(f["id"], f["formula"]) for f in (yaml.safe_load(open(p)) or {}).get("factors", [])]
    return out


def test_every_formula_already_in_the_library_passes_the_draft_validator():
    formulas = _library_formulas()
    assert len(formulas) >= 20
    for fid, formula in formulas:
        errors, deps = fd.check_formula(formula)
        assert errors == [], (fid, errors)
        assert deps, fid  # 每个公式至少依赖一列


@pytest.mark.parametrize(
    "formula,needle",
    [
        ("", "不能为空"),
        ("   ", "不能为空"),
        ("close +", "语法"),
        ("__import__('os').system('x')", "只能调用"),  # 属性调用被拒
        ("close.shift(1)", "只能调用"),
        ("[c for c in close]", "禁止"),
        ("lambda x: x", "禁止"),
        ("unknown_fn(close)", "未知函数"),
        ("close[0]", "Subscript"),  # 下标不在白名单内
    ],
)
def test_unsafe_or_malformed_formulas_are_rejected(formula, needle):
    errors, _ = fd.check_formula(formula)
    assert errors and needle in errors[0]


def test_dependencies_exclude_function_names():
    _, deps = fd.check_formula(
        "-ts_corr(cs_rank(ts_delta(log(volume), 1)), cs_rank((close-open)/open), 6)"
    )
    assert deps == {"volume", "close", "open"}


def _ok(**kw):
    args = dict(
        fid="my_f", name="我的因子", category="momentum", formula="close / ts_delay(close, 5) - 1"
    )
    args.update(kw)
    return fd.validate_draft(**args)


def test_valid_draft_builds_a_disabled_record():
    r = _ok(tags="a, b，c", expected_sign="+1", known_columns={"close"})
    assert r["errors"] == [] and r["warnings"] == [] and r["deps"] == ["close"]
    rec = r["record"]
    assert rec["enabled"] is False and rec["source"] == "custom" and rec["tags"] == ["a", "b", "c"]
    assert rec["expected_sign"] == 1 and rec["category"] == "momentum"


@pytest.mark.parametrize(
    "kw,needle",
    [
        ({"fid": ""}, "id 不能为空"),
        ({"fid": "1abc"}, "合法列名"),
        ({"fid": "a-b"}, "合法列名"),
        ({"fid": "x" * 65}, "合法列名"),
        ({"fid": "date"}, "保留列"),
        ({"fid": "responder_9"}, "保留列"),
        ({"fid": "dup", "existing_ids": {"dup"}}, "已存在"),
        ({"name": ""}, "name 不能为空"),
        ({"expected_sign": "2"}, "expected_sign"),
    ],
)
def test_draft_field_validation(kw, needle):
    assert any(needle in e for e in _ok(**kw)["errors"])


def test_unknown_dependency_is_a_warning_not_an_error():
    r = _ok(formula="close / ts_delay(close, 5) - nonexistent_col", known_columns={"close", "open"})
    assert r["errors"] == [] and any("nonexistent_col" in w for w in r["warnings"])
    # 没有已知列信息时（行情缓存缺失）不乱报警告
    assert _ok(formula="a + b", known_columns=set())["warnings"] == []


def test_save_draft_creates_appends_and_keeps_structure(tmp_path):
    path = tmp_path / "lib" / "custom.yaml"  # 目录也不存在
    r1, r2 = _ok(fid="f1")["record"], _ok(fid="f2")["record"]
    fd.save_draft(r1, str(path))
    fd.save_draft(r2, str(path))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert [f["id"] for f in data["factors"]] == ["f1", "f2"]
    assert data["factors"][0]["enabled"] is False and "我的因子" in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob("*.tmp-*"))  # 原子写，不留临时文件


def test_save_draft_refuses_duplicates_in_file_or_elsewhere_in_library(tmp_path):
    """加载器遇到重复 id 只会静默跳过后来者：用户会以为保存成功，其实从未生效。"""
    path = tmp_path / "custom.yaml"
    fd.save_draft(_ok(fid="f1")["record"], str(path))
    with pytest.raises(ValueError, match="重复"):
        fd.save_draft(_ok(fid="f1")["record"], str(path))
    with pytest.raises(ValueError, match="重复"):
        fd.save_draft(_ok(fid="elsewhere")["record"], str(path), existing_ids={"elsewhere"})
    assert len(yaml.safe_load(path.read_text())["factors"]) == 1


def test_save_draft_handles_empty_and_malformed_files(tmp_path):
    p = tmp_path / "custom.yaml"
    p.write_text("")  # 空文件
    fd.save_draft(_ok(fid="a")["record"], str(p))
    p.write_text("factors:\n")  # factors: null
    fd.save_draft(_ok(fid="b")["record"], str(p))
    assert [f["id"] for f in yaml.safe_load(p.read_text())["factors"]] == ["b"]
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="结构异常"):
        fd.save_draft(_ok(fid="c")["record"], str(p))
    assert p.read_text() == "- just\n- a list\n"  # 结构异常时不改动原文件


def test_saved_draft_is_loadable_by_the_production_loader(tmp_path):
    """端到端：前端写出的 yaml，必须能被生产加载器读成 FactorDefinition。"""
    from data.factors.schema import load_factor_definitions

    fd.save_draft(
        _ok(fid="roundtrip", expected_sign="-1", tags="x")["record"], str(tmp_path / "custom.yaml")
    )
    defs = load_factor_definitions(str(tmp_path))
    assert [d.id for d in defs] == ["roundtrip"]
    assert defs[0].enabled is False and defs[0].expected_sign == -1 and defs[0].tags == ["x"]
