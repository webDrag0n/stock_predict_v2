"""scripts/run_pipeline.py：参数校验（拒绝而不是静默忽略）与 run registry 登记。

步骤函数用假的，不加载面板、不训练。真实步骤的行为由 tests/test_steps_*.py 覆盖。
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

import run_registry as rr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "run_pipeline", os.path.join(ROOT, "scripts", "run_pipeline.py")
)
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

SCHEMA = [
    {"key": "factor_subset", "type": "text", "default": ""},
    {"key": "lookback_days", "type": "number", "default": 1825, "min": 120, "max": 3650},
    {"key": "n_layers", "type": "number", "default": 5, "min": 3, "max": 10},
    {"key": "ratio", "type": "number", "default": 0.5, "min": 0, "max": 1},
    {"key": "dump_layers", "type": "bool", "default": True},
    {"key": "mode", "type": "select", "default": "a", "options": ["a", "b"]},
]


# ----------------------------------------------------------------------------- 参数
def test_split_sets():
    assert rp.split_sets(["a=1", "b=x=y", "c="]) == {"a": "1", "b": "x=y", "c": ""}
    for bad in (["noequals"], ["=1"], ["  =2"]):
        with pytest.raises(rp.ParamError):
            rp.split_sets(bad)


def test_coerce_fills_defaults_and_converts_types():
    p = rp.coerce_params("x", {"lookback_days": "730", "ratio": "0.25", "dump_layers": "0"}, SCHEMA)
    assert p["lookback_days"] == 730 and isinstance(
        p["lookback_days"], int
    )  # 默认是 int → 保持 int
    assert p["ratio"] == 0.25 and p["dump_layers"] is False
    assert p["n_layers"] == 5 and p["factor_subset"] == "" and p["mode"] == "a"  # 未给的取默认


@pytest.mark.parametrize(
    "raw,needle",
    [
        ({"lookback_dayz": "7"}, "不接受参数"),  # 拼写错误必须报错，不能静默忽略
        ({"lookback_days": "abc"}, "必须是数字"),
        ({"lookback_days": "10"}, "小于下限"),
        ({"lookback_days": "99999"}, "大于上限"),
        ({"lookback_days": "nan"}, "有限数字"),
        ({"lookback_days": "inf"}, "有限数字"),
        ({"dump_layers": "maybe"}, "必须是布尔值"),
        ({"mode": "c"}, "不在可选项"),
    ],
)
def test_coerce_rejects_bad_params(raw, needle):
    with pytest.raises(rp.ParamError) as e:
        rp.coerce_params("x", raw, SCHEMA)
    assert needle in str(e.value)


def test_env_bool_params_are_real_bools_before_env_mapping():
    """env_overrides 用 `params[k]` 的真值决定写 "1"/"0"；字符串 "0" 是真值，会把「关」变成「开」。"""
    out = rp._bool_env_keys_precoerced({"use_impact": "0", "use_bb_timing": "true", "x": "0"})
    assert out["use_impact"] is False and out["use_bb_timing"] is True and out["x"] == "0"
    with pytest.raises(rp.ParamError):
        rp._bool_env_keys_precoerced({"use_impact": "perhaps"})


def test_main_exits_2_on_bad_params_without_creating_a_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUN_REGISTRY_ROOT", str(tmp_path / "runs"))
    code = rp.main(["--mode", "factoreval", "--set", "no_such_param=1"])
    assert code == 2 and "参数错误" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
    assert rp.main(["--mode", "factoreval", "--set", "lookback_days=5"]) == 2
    assert rp.main(["--mode", "factoreval", "--set", "oops"]) == 2


# ----------------------------------------------------------------------------- 登记
def _steps(order, boom=None):
    def make(sid):
        def fn(ctx):
            order.append(sid)
            ctx.report_progress("phase_" + sid, 1, 2, "halfway", metric=0.5)
            if boom == sid:
                raise RuntimeError(f"{sid} 炸了")
            return {
                "summary": f"{sid} 完成",
                "kpis": [{"label": "n", "value": 3}],
                "tables": [],
                "figures": [],
            }

        return fn

    return [(s, f"标题 {s}", make(s)) for s in ("load", "eval")]


def test_run_steps_registers_everything_a_frontend_needs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_REGISTRY_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("RUN_ID", "p1")
    order: list[str] = []
    assert (
        rp.run_steps(
            "factoreval", "因子评估", _steps(order), {"factor_subset": "ep", "lookback_days": 730}
        )
        == 0
    )
    assert order == ["load", "eval"]
    meta = rr.read_run("p1", root=str(tmp_path / "runs"))
    assert meta["status"] == "done" and meta["kind"] == "factoreval" and meta["exit_code"] == 0
    assert [s["id"] for s in meta["steps"]] == ["load", "eval"]
    assert all(s["status"] == "done" for s in meta["steps"])
    assert meta["steps"][0]["summary"] == "load 完成"
    assert meta["params"]["factor_subset"] == "ep"
    # 每步完整报告落盘并登记为产物
    rep = tmp_path / "runs" / "p1" / "reports" / "00_load.json"
    assert json.loads(rep.read_text())["kpis"][0]["label"] == "n"
    assert any(a["name"].startswith("报告") for a in meta["artifacts"])

    # 进度被桥接到 registry：进度只存在于运行中的快照里（步骤结束即清空），所以在步骤执行中读取
    seen: dict = {}

    def probing_step(ctx):
        ctx.report_progress("训练", 3, 4, "窗口 3", ic=0.05)
        seen["meta"] = rr.read_run("p1b", root=str(tmp_path / "runs"))
        return {"summary": "ok"}

    monkeypatch.setenv("RUN_ID", "p1b")
    assert rp.run_steps("factoreval", "x", [("probe", "探针", probing_step)], {}) == 0
    prog = seen["meta"]["progress"]
    assert prog["phase"] == "训练" and prog["current"] == 3 and prog["total"] == 4
    assert prog["message"] == "窗口 3" and prog["extras"]["ic"] == 0.05
    assert seen["meta"]["current_step"] == "probe"


def test_failing_step_is_recorded_with_traceback_and_reraised(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_REGISTRY_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("RUN_ID", "p2")
    order: list[str] = []
    with pytest.raises(RuntimeError, match="eval 炸了"):
        rp.run_steps("factoreval", "x", _steps(order, boom="eval"), {})
    meta = rr.read_run("p2", root=str(tmp_path / "runs"))
    assert meta["status"] == "failed" and meta["failure"]["exc_type"] == "RuntimeError"
    assert "eval 炸了" in meta["failure"]["exc_msg"]
    by = {s["id"]: s["status"] for s in meta["steps"]}
    assert by == {"load": "done", "eval": "failed"}  # 失败步骤被标红，已完成的保持完成


def test_registry_disabled_still_runs_steps(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUN_REGISTRY", "0")
    monkeypatch.setenv("RUN_REGISTRY_ROOT", str(tmp_path / "runs"))
    order: list[str] = []
    assert rp.run_steps("factoreval", "x", _steps(order), {}) == 0
    assert order == ["load", "eval"] and not (tmp_path / "runs").exists()
    order.clear()
    assert rp.run_steps("factoreval", "x", _steps(order, boom="load"), {}) == 1  # 失败 → 退出码 1


def test_every_pipeline_mode_has_a_schema_the_wrapper_can_validate():
    """真实 schema 的默认值本身必须通过校验（否则「不带任何参数」就会被自己拒绝）。"""
    from pipeline import registry

    for mode in rp.MODES:
        p = rp.coerce_params(mode, {}, registry.PARAM_SCHEMA[mode])
        assert set(p) == {x["key"] for x in registry.PARAM_SCHEMA[mode]}


# ----------------------------------------------------------------------------- --draft
@pytest.fixture
def clean_draft_env(monkeypatch):
    """apply_draft 直接写 os.environ（config 在 import 时才读）：用 setenv 登记，测试结束自动还原。"""
    for k in ("USE_FORMULA_FACTORS", "FORMULA_FACTOR_SUBSET", "FACTOR_PANEL_CACHE"):
        monkeypatch.setenv(k, "sentinel")
    yield


def test_draft_enables_only_that_formula_and_never_touches_the_shared_panel_cache(clean_draft_env):
    """真实风险：因子面板缓存的签名含公式库哈希与 FORMULA_FACTOR_SUBSET。若评估草稿时仍写缓存，
    生产夜间任务共用的缓存会被「只含这一个公式因子」的面板覆盖。所以必须关缓存。"""
    out = rp.apply_draft("factoreval", "my_draft", {"lookback_days": "730"})
    assert os.environ["USE_FORMULA_FACTORS"] == "1"
    assert os.environ["FORMULA_FACTOR_SUBSET"] == "my_draft"
    assert os.environ["FACTOR_PANEL_CACHE"] == "0"
    assert out == {"lookback_days": "730", "factor_subset": "my_draft"}


def test_draft_validation(clean_draft_env):
    with pytest.raises(rp.ParamError, match="只能用于"):
        rp.apply_draft("train", "x", {})
    for bad in ("1abc", "a-b", "a b", "", "x" * 65, "a;rm"):
        with pytest.raises(rp.ParamError, match="非法"):
            rp.apply_draft("factoreval", bad, {})
    with pytest.raises(rp.ParamError, match="不同因子"):
        rp.apply_draft("factoreval", "a", {"factor_subset": "b"})
    # 与 --set factor_subset 指向同一个因子是允许的
    assert rp.apply_draft("factoreval", "a", {"factor_subset": "a"})["factor_subset"] == "a"
    # 失败的校验不应已经改了环境
    assert os.environ["USE_FORMULA_FACTORS"] == "1"  # 上一个成功调用留下的；失败调用不再覆盖


def test_failed_draft_validation_leaves_env_untouched(clean_draft_env):
    with pytest.raises(rp.ParamError):
        rp.apply_draft("train", "x", {})
    assert os.environ["USE_FORMULA_FACTORS"] == "sentinel"
    assert os.environ["FACTOR_PANEL_CACHE"] == "sentinel"


def test_main_draft_flag_wires_through_before_registry_import(
    clean_draft_env, tmp_path, monkeypatch
):
    """--draft 必须在 import pipeline.registry（触发 config）之前生效；用假 registry 验证顺序。"""
    seen = {}

    def fake_run_steps(mode, label, steps, params):
        seen.update(env=dict(os.environ), params=params)
        return 0

    monkeypatch.setattr(rp, "run_steps", fake_run_steps)
    monkeypatch.setenv("RUN_REGISTRY_ROOT", str(tmp_path / "runs"))
    assert (
        rp.main(["--mode", "factoreval", "--draft", "my_draft", "--set", "lookback_days=730"]) == 0
    )
    assert seen["env"]["USE_FORMULA_FACTORS"] == "1" and seen["env"]["FACTOR_PANEL_CACHE"] == "0"
    assert seen["params"]["factor_subset"] == "my_draft" and seen["params"]["lookback_days"] == 730
    assert rp.main(["--mode", "train", "--draft", "x"]) == 2


# ----------------------------------------------------------------------------- 真实步骤集成
def test_real_factoreval_steps_run_through_the_wrapper_end_to_end(tmp_path, monkeypatch):
    """不用假步骤：把真实的 eval_factors / eval_report 经包装器跑一遍（合成面板，只写 tmp）。

    验证包装器与步骤函数的契约：ctx.session_dir=run.dir 时步骤的 artifacts/ 落在运行目录内、
    OUTPUT_DIR 重定向后 factor_eval.csv 与分层产物落在沙盒、报告落盘并可被前端读取。
    """
    import numpy as np
    import pandas as pd

    import config as C
    from pipeline import steps_factoreval as S
    from webapp2 import data
    from webapp2 import report_view as rv

    out = tmp_path / "outputs"
    monkeypatch.setattr(C, "OUTPUT_DIR", str(out))
    # 测试把 config.OUTPUT_DIR 指到沙盒后，run_registry.runs_root 跟随它（见其文档字符串）
    runs_dir = out / "runs"
    monkeypatch.setenv("RUN_ID", "real1")
    monkeypatch.setattr(data, "OUTPUTS", str(out))

    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    frames = []
    for d in dates:
        sig = rng.normal(0, 1, 30)
        frames.append(
            pd.DataFrame(
                {
                    "date": d,
                    "symbol": [f"S{i:03d}" for i in range(30)],
                    "fac_signal": sig,
                    "fac_noise": rng.normal(0, 1, 30),
                    C.TARGET: 0.02 * sig + rng.normal(0, 0.02, 30),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)

    def data_import(ctx):  # 真实的 data_import 要加载全市场面板；这里换成注入合成面板
        ctx.shared["panel"], ctx.shared["factors"] = panel, ["fac_signal", "fac_noise"]
        return S.report("data_import", "数据导入", summary="合成面板")

    steps = [
        ("data_import", "数据导入", data_import),
        ("eval_factors", "因子评估", S.eval_factors),
        ("eval_report", "评估报告", S.eval_report),
    ]
    params = rp.coerce_params(
        "factoreval",
        {"lookback_days": "730"},
        __import__("pipeline.registry", fromlist=["x"]).PARAM_SCHEMA["factoreval"],
    )
    assert rp.run_steps("factoreval", "因子评估", steps, params) == 0

    run_dir = runs_dir / "real1"
    meta = rr.read_run("real1", root=str(runs_dir))
    assert meta["status"] == "done" and [s["status"] for s in meta["steps"]] == ["done"] * 3
    assert (out / "factor_eval.csv").exists() and (out / "factor_layer_fac_signal.parquet").exists()
    assert (
        run_dir / "artifacts" / "eval_table.parquet"
    ).exists()  # 步骤的 artifacts/ 落在运行目录内
    assert (run_dir / "reports" / "01_eval_factors.json").exists()
    reps = rv.read_reports("real1", str(runs_dir))
    assert [r["step"] for r in reps] == ["data_import", "eval_factors", "eval_report"]
    assert any("准入通过" in json.dumps(r, ensure_ascii=False) for r in reps)
