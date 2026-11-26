"""模型页：指标口径、门控结论、对比、晋升/删除防呆、页面渲染与刷新门控。"""

from __future__ import annotations

import json
import time

import pytest

import run_registry as rr
from webapp2 import actions, data, model_logic as ml
from webapp2.pages import models

from tests.test_webapp2_pages import ids_of, text_of  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    monkeypatch.setattr("model_store.C.MODEL_DIR", str(tmp_path / "models"), raising=False)
    data.refresh()
    yield tmp_path
    data.refresh()


def metrics(sharpe, dd=-8.0, **kw):
    return {"sharpe": sharpe, "max_drawdown": dd, "alpha_ann": 0.05, "n_days": 72, **kw}


def put_model(sandbox, sub, mid, *, evaluation=None, promotion=None, recipe=None):
    # champion 直接放在 models/champion/，没有 id 子目录；staging/archive 才按 id 分目录
    d = sandbox / "models" / sub if sub == "champion" else sandbox / "models" / sub / mid
    d.mkdir(parents=True, exist_ok=True)
    man = {
        "model_id": mid,
        "train_end_date": "2026-09-03",
        "fingerprint": "fp",
        "config_snapshot": recipe or {"feat_cols": ["a", "b"], "n_seeds": 1, "fw": 20},
    }
    if promotion:
        man["promotion"] = promotion
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "ensemble_state.json").write_text("{}")
    if evaluation is not None:
        (d / "evaluation.json").write_text(json.dumps(evaluation))
    data.refresh()
    return d


def eval_(cand, champ, decision="rejected", reason="Sharpe 未超过 champion", **kw):
    return {
        "candidate": cand,
        "champion": champ,
        "decision": decision,
        "reason": reason,
        "eval_days": 120,
        "window": ["2026-07-21", "2026-11-05"],
        **kw,
    }


# ----------------------------------------------------------------------------- 指标口径
def test_champion_metrics_come_from_its_own_promotion_not_the_replaced_models_side():
    """真实踩坑：champion/evaluation.json 里的 "champion" 侧是被它取代的旧模型。
    旧前端把它当现任指标展示（Sharpe -0.80），而现任在晋升时是 +0.60。"""
    man = {
        "promotion": {
            "candidate_metrics": metrics(0.603),
            "champion_metrics": metrics(-0.798),
            "eval_window": ["2026-07-21", "2026-11-05"],
        }
    }
    ev = eval_(metrics(-1.47), metrics(-1.47))  # 入围时的评估，两侧相同且已被否决
    got, src = ml.champion_metrics(man, ev)
    assert got["sharpe"] == 0.603 and "晋升时门控重放" in src and "2026-07-21" in src


def test_champion_metrics_do_not_borrow_untrustworthy_numbers():
    man = {"promotion": {"reason": "bootstrap"}}
    got, src = ml.champion_metrics(man, eval_(metrics(-1.47), metrics(-1.47), decision="rejected"))
    assert got == {} and "无可用评估" in src
    # 评估结论是 promoted 且两侧不同 → 候选侧就是它入围时的表现
    ok = eval_(metrics(1.0), metrics(0.2), decision="promoted")
    got, src = ml.champion_metrics(man, ok)
    assert got["sharpe"] == 1.0 and "入围" in src
    # promoted 但两侧相同（F13 症状）→ 不可信，不采用
    assert ml.champion_metrics(man, eval_(metrics(1.0), metrics(1.0), decision="promoted"))[0] == {}


def test_identical_replay_detection():
    assert ml.identical_replay(eval_(metrics(-1.47), metrics(-1.47)))
    assert not ml.identical_replay(eval_(metrics(0.6), metrics(-0.8)))
    assert not ml.identical_replay({}) and not ml.identical_replay(None)
    # params 差异不算指标（两侧 params 本来就可以不同）
    a, b = metrics(1.0, params={"x": 1}), metrics(1.0, params={"x": 2})
    assert ml.identical_replay(eval_(a, b))


def test_staging_summary_fields():
    e = {"id": "c1", "evaluation": eval_(metrics(0.6, -6.7), metrics(-0.8, -8.8), "promoted", "过")}
    s = ml.staging_summary(e)
    assert s["eligible"] and s["d_sharpe"] == pytest.approx(1.4) and s["d_dd"] == pytest.approx(2.1)
    assert not s["suspect"] and s["eval_days"] == 120
    assert not ml.staging_summary({"id": "x", "evaluation": None})["has_eval"]
    assert ml.staging_summary({"id": "x", "evaluation": None})["d_sharpe"] is None


# ----------------------------------------------------------------------------- 对比
def test_metric_rows_direction_and_missing(sandbox):
    a = {"metrics": {"sharpe": 1.0, "max_drawdown": -10.0, "beta": 0.2}}
    b = {"metrics": {"sharpe": 1.5, "max_drawdown": -6.0, "beta": 0.1}}
    rows = {r["key"]: r for r in ml.metric_rows(a, b)}
    assert rows["sharpe"]["better"] == "b" and rows["sharpe"]["delta"] == pytest.approx(0.5)
    assert rows["max_drawdown"]["better"] == "b"  # -6 比 -10 更接近 0
    assert rows["beta"]["better"] is None  # 无方向
    assert "sortino" not in rows  # 两边都没有的指标不占行
    assert ml.metric_rows(None, None) == []


def test_recipe_and_feature_diff():
    a = {"feat_cols": ["x", "y"], "n_seeds": 1, "fw": 20}
    b = {"feat_cols": ["y", "z"], "n_seeds": 3, "fw": 20}
    d = {r["key"]: r for r in ml.recipe_diff(a, b)}
    assert "feat_cols" not in d and not d["n_seeds"]["same"] and d["fw"]["same"]
    assert ml.feat_diff(a, b) == {"n_a": 2, "n_b": 2, "only_a": ["x"], "only_b": ["z"]}


def test_model_record_and_options_cover_all_roles(sandbox):
    put_model(sandbox, "champion", "champ", promotion={"candidate_metrics": metrics(0.6)})
    put_model(sandbox, "staging", "cand", evaluation=eval_(metrics(0.1), metrics(0.6)))
    put_model(sandbox, "archive", "old")
    labels = [o["label"] for o in ml.model_options()]
    assert labels[0].startswith("Champion") and any("Staging · cand" in x for x in labels)
    assert any("Archive · old" in x for x in labels)
    assert ml.model_record("champion")["metrics"]["sharpe"] == 0.6
    assert ml.model_record("cand")["role"] == "staging"
    assert ml.model_record("archive:old")["role"] == "archive"
    assert ml.model_record("nope") is None and ml.model_record("archive:nope") is None


def test_staging_record_flags_untrustworthy_replay(sandbox):
    put_model(sandbox, "staging", "twin", evaluation=eval_(metrics(-1.47), metrics(-1.47)))
    assert "不可信" in ml.model_record("twin")["src"]


# ----------------------------------------------------------------------------- 晋升计划
def test_promote_plan_requires_reason_when_gate_failed_or_suspect(sandbox):
    put_model(
        sandbox, "staging", "good", evaluation=eval_(metrics(1), metrics(0), "promoted", "过")
    )
    put_model(sandbox, "staging", "bad", evaluation=eval_(metrics(0.1), metrics(0.5)))
    put_model(
        sandbox, "staging", "twin", evaluation=eval_(metrics(-1.47), metrics(-1.47), "promoted")
    )
    put_model(sandbox, "staging", "noeval")
    put_model(
        sandbox,
        "staging",
        "gap",
        evaluation=eval_(metrics(1), metrics(0), "promoted", champion_missing_cols=["f1", "f2"]),
    )
    assert not ml.promote_plan("good")["need_reason"]
    bad = ml.promote_plan("bad")
    assert bad["can"] and bad["need_reason"] and "未达门槛" in bad["warnings"][0]
    assert any("不可信" in w for w in ml.promote_plan("twin")["warnings"])
    assert any("没有门控评估" in w for w in ml.promote_plan("noeval")["warnings"])
    assert any("2 个特征列" in w for w in ml.promote_plan("gap")["warnings"])
    assert not ml.promote_plan("ghost")["can"]


# ----------------------------------------------------------------------------- 写操作防呆
def test_promote_gated_candidate_needs_no_reason_and_records_one(sandbox):
    put_model(sandbox, "champion", "old")
    put_model(
        sandbox, "staging", "good", evaluation=eval_(metrics(1), metrics(0), "promoted", "过")
    )
    ok, msg = actions.promote_staging("good", "")
    assert ok, msg
    assert data.champion_manifest()["model_id"] == "good"
    assert (sandbox / "models" / "archive" / "old").is_dir()


def test_promote_below_gate_is_refused_without_reason_and_audited_with_one(sandbox):
    put_model(sandbox, "champion", "old")
    put_model(sandbox, "staging", "bad", evaluation=eval_(metrics(0.1), metrics(0.5)))
    for weak in ("", "   ", "好", "abc"):
        ok, msg = actions.promote_staging("bad", weak)
        assert not ok and "必须填写理由" in msg
    assert data.champion_manifest()["model_id"] == "old"  # 被拒绝的晋升没有动 champion
    ok, msg = actions.promote_staging("bad", "近 3 窗口为负，需要换成更保守的配方")
    assert ok, msg
    log = (sandbox / "models" / "promotions.log").read_text()
    assert "PROMOTE bad" in log and "更保守的配方" in log  # 理由进了审计


def test_promote_and_delete_refused_while_nightly_is_running(sandbox):
    """夜间任务正在读 champion/staging：此时改动会让同一晚的预测与门控用到不同模型。"""
    put_model(sandbox, "champion", "old")
    put_model(
        sandbox, "staging", "good", evaluation=eval_(metrics(1), metrics(0), "promoted", "过")
    )
    live = rr.RunReporter("nightly", run_id="n_live", root=str(sandbox / "outputs" / "runs"))
    live.step("model")
    ok, msg = actions.promote_staging("good", "")
    assert not ok and "n_live" in msg
    ok, msg = actions.delete_staging("good")
    assert not ok and "n_live" in msg
    assert data.champion_manifest()["model_id"] == "old"
    assert (sandbox / "models" / "staging" / "good").is_dir()
    live.finish()
    assert actions.delete_staging("good")[0]


def test_promote_unknown_or_malformed_ids(sandbox):
    assert not actions.promote_staging("../evil", "xxxxxxx")[0]
    ok, msg = actions.promote_staging("ghost", "xxxxxxx")
    assert not ok and "不存在" in msg


# ----------------------------------------------------------------------------- 页面
def test_models_page_on_empty_disk(sandbox):
    t = text_of(models.layout())
    for needle in ("还没有 champion", "没有候选", "没有已退役", "还没有晋升/删除记录", "模型对比"):
        assert needle in t, needle


def test_models_page_renders_real_shaped_data(sandbox):
    put_model(
        sandbox,
        "champion",
        "champ",
        promotion={
            "reason": "manual：F13 修复后重评估通过",
            "decided_at": "2026-11-16T12:40:19",
            "candidate_metrics": metrics(0.603, -6.666),
        },
    )
    put_model(sandbox, "staging", "twin", evaluation=eval_(metrics(-1.47), metrics(-1.47)))
    put_model(sandbox, "archive", "old")
    t = text_of(models.layout())
    for needle in (
        "champ",
        "F13 修复后重评估通过",
        "晋升时门控重放",
        "+0.60",
        "twin",
        "两侧指标逐项相同",
        "old",
    ):
        assert needle in t, needle


def test_staging_rows_have_pattern_buttons(sandbox):
    put_model(sandbox, "staging", "c1", evaluation=eval_(metrics(1), metrics(0), "promoted"))
    found = set()

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            [walk(c) for c in n]
            return
        i = getattr(n, "id", None)
        if isinstance(i, dict):
            found.add((i["type"], i["id"]))
        walk(getattr(n, "children", None))

    walk(models.body())
    assert found == {("m-promote", "c1"), ("m-delete", "c1")}


def test_confirm_dialog_content(sandbox):
    put_model(sandbox, "staging", "bad", evaluation=eval_(metrics(0.1), metrics(0.5)))
    put_model(sandbox, "staging", "good", evaluation=eval_(metrics(1), metrics(0), "promoted"))
    p = models.open_confirm("promote", "bad")
    assert p["can"] and p["need_reason"] and "未达门槛" in " ".join(p["lines"])
    assert not models.open_confirm("promote", "good")["need_reason"]
    d = models.open_confirm("delete", "bad")
    assert d["can"] and "永久删除" in " ".join(d["lines"]) and not d["need_reason"]
    assert not models.open_confirm("delete", "ghost")["can"]
    assert not models.open_confirm("promote", "ghost")["can"]


def test_compare_view(sandbox):
    put_model(sandbox, "champion", "champ", promotion={"candidate_metrics": metrics(0.6, -6.7)})
    put_model(
        sandbox,
        "staging",
        "cand",
        evaluation=eval_(metrics(0.9, -5.0), metrics(0.6, -6.7)),
        recipe={"feat_cols": ["a", "c"], "n_seeds": 3, "fw": 20},
    )
    t = text_of(models.compare_view("champion", "cand"))
    for needle in ("Sharpe", "n_seeds", "仅 B 有 1", "指标来源", "晋升时门控重放"):
        assert needle in t, needle
    assert "选择两个不同的模型" in text_of(models.compare_view("champion", "champion"))
    assert "不存在" in text_of(models.compare_view("champion", "ghost"))


def test_static_area_outside_polled_body(sandbox):
    ids = ids_of(models.layout())
    assert {"m-tick", "m-sig", "m-body", "m-cmp-a", "m-cmp-b", "m-confirm-run", "m-reason"} <= ids
    assert not ({"m-cmp-a", "m-confirm-run", "m-reason"} & ids_of(models.body()))


# ----------------------------------------------------------------------------- 刷新门控
def test_refresh_skipped_when_models_unchanged_and_fires_on_change(sandbox):
    put_model(sandbox, "champion", "champ")
    new, sig = models.refresh_decision(None)
    assert new is not None
    assert models.refresh_decision(sig)[0] is None
    put_model(sandbox, "staging", "fresh")
    new, sig2 = models.refresh_decision(sig)
    assert new is not None and sig2 != sig
    # 门控把评估写进已有候选 → 也要刷新
    put_model(sandbox, "staging", "fresh", evaluation=eval_(metrics(1), metrics(0)))
    assert models.refresh_decision(sig2)[0] is not None


def test_refresh_fires_when_nightly_starts_or_stops(sandbox):
    _, sig = models.refresh_decision(None)
    live = rr.RunReporter("nightly", run_id="n9", root=str(sandbox / "outputs" / "runs"))
    live.step("x")
    new, sig2 = models.refresh_decision(sig)
    assert new is not None
    live.finish()
    assert models.refresh_decision(sig2)[0] is not None


def test_app_routes_models(sandbox):
    from webapp2 import app as appmod

    assert ("/models", "模型") in appmod.NAV
    assert "模型对比" in text_of(appmod.route("/models"))


def test_promote_refuses_when_model_dirs_disagree(sandbox, monkeypatch):
    """页面按 data.MODELS 展示，model_store 按 config.MODEL_DIR 晋升。两者不同就拒绝，
    否则页面上点的候选和实际被晋升的会是不同目录里的东西（沙盒浏览器测试里真实出现过）。"""
    put_model(sandbox, "champion", "old")
    put_model(
        sandbox, "staging", "good", evaluation=eval_(metrics(1), metrics(0), "promoted", "过")
    )
    monkeypatch.setattr("model_store.C.MODEL_DIR", str(sandbox / "elsewhere"), raising=False)
    ok, msg = actions.promote_staging("good", "")
    assert not ok and "模型目录不一致" in msg
    assert data.champion_manifest()["model_id"] == "old"
    assert (sandbox / "models" / "staging" / "good").is_dir()
    assert not (sandbox / "models" / "archive" / "old").exists()


# ----------------------------------------------------------------------------- champion 的留档副本
def test_champion_copy_in_staging_is_not_a_pending_candidate(sandbox):
    """真实场景：晋升是复制不是移动，现任 champion 的 model_id 在 staging/ 里还留着一份。
    它不该有「晋升」按钮、不该进对比下拉框、不该让引导条说「有候选待决策」。"""
    put_model(sandbox, "champion", "cur")
    put_model(sandbox, "staging", "cur", evaluation=eval_(metrics(-1.47), metrics(-1.47)))
    put_model(sandbox, "staging", "other", evaluation=eval_(metrics(0.1), metrics(0.6)))
    entries = data.staging_models()
    assert [e["id"] for e in ml.pending_candidates(entries)] == ["other"]
    assert ml.staging_summary(next(e for e in entries if e["id"] == "cur"))["is_champion"]
    values = [o["value"] for o in ml.model_options()]
    assert "cur" not in values and "other" in values and "champion" in values
    plan = ml.promote_plan("cur")
    assert not plan["can"] and "就是当前 champion" in plan["why"]


def test_champion_copy_matched_by_manifest_model_id_too(sandbox):
    """目录名带后缀（如 -2）但 manifest.model_id 与 champion 相同，同样视为副本。"""
    put_model(sandbox, "champion", "cur")
    d = put_model(sandbox, "staging", "cur-2")
    man = json.loads((d / "manifest.json").read_text())
    man["model_id"] = "cur"
    (d / "manifest.json").write_text(json.dumps(man))
    data.refresh()
    assert ml.pending_candidates(data.staging_models()) == []


def test_guidance_ignores_champion_copy(sandbox):
    put_model(sandbox, "champion", "cur")
    put_model(sandbox, "staging", "cur", evaluation=eval_(metrics(1), metrics(0), "promoted", "过"))
    assert "线上模型运行中" in ml.guidance(data.champion_manifest(), data.staging_models())
    put_model(
        sandbox, "staging", "real", evaluation=eval_(metrics(1), metrics(0), "promoted", "过")
    )
    assert "1 个候选已达晋升门槛" in ml.guidance(data.champion_manifest(), data.staging_models())


def test_page_shows_champion_copy_without_promote_button(sandbox):
    put_model(sandbox, "champion", "cur")
    put_model(sandbox, "staging", "cur", evaluation=eval_(metrics(1), metrics(0), "promoted"))
    put_model(sandbox, "staging", "real", evaluation=eval_(metrics(0.1), metrics(0.6)))
    body = models.body()
    t = text_of(body)
    assert "当前 champion" in t and "晋升后留档的副本" in t and "待决策 1 / 共 2" in t
    found = set()

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            [walk(c) for c in n]
            return
        i = getattr(n, "id", None)
        if isinstance(i, dict):
            found.add((i["type"], i["id"]))
        walk(getattr(n, "children", None))

    walk(body)
    assert ("m-promote", "cur") not in found and ("m-delete", "cur") in found
    assert ("m-promote", "real") in found


def test_promote_action_refuses_champion_copy(sandbox):
    put_model(sandbox, "champion", "cur")
    put_model(sandbox, "staging", "cur", evaluation=eval_(metrics(1), metrics(0), "promoted"))
    ok, msg = actions.promote_staging("cur", "")
    assert not ok and "就是当前 champion" in msg
    assert not (sandbox / "models" / "archive" / "cur").exists()  # 没有把自己归档掉


def test_compare_defaults_never_pair_champion_with_its_own_copy(sandbox):
    put_model(sandbox, "champion", "cur")
    put_model(sandbox, "staging", "cur")
    ids = ids_of(models.layout())
    assert "m-cmp-b" in ids
    out = text_of(models.layout())
    assert "选择两个不同的模型" in out  # 没有别的可对比对象时给空态而不是自己对自己


# ----------------------------------------------------------------------------- 发起训练 / 信号评估
def test_task_buttons_live_outside_the_polled_body(sandbox):
    ids = ids_of(models.layout())
    assert {
        "m-train",
        "m-eval",
        "m-eval-ref",
        "m-train-msg",
        "m-eval-msg",
        "m-train-status",
        "m-eval-status",
    } <= ids
    assert not ({"m-train", "m-eval", "m-eval-ref"} & ids_of(models.body()))
    text = text_of(models.layout())
    assert "训练区" in text and "评估区" in text


def test_train_and_modeleval_launch_through_the_run_registry_wrapper(sandbox, monkeypatch):
    captured = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            captured.append((cmd, kw.get("env", {})))

    monkeypatch.setattr(actions.subprocess, "Popen", FakePopen)
    ok, msg, rid = actions.launch_cli("train", [])
    assert ok, msg
    assert captured[-1][0][2:6] == ["scripts/run_pipeline.py", "--mode", "train"] or captured[-1][
        0
    ][1:5] == ["-u", "scripts/run_pipeline.py", "--mode", "train"]
    assert captured[-1][1]["RUN_ID"] == rid
    with rr.RunReporter("train", run_id=rid, root=str(sandbox / "outputs" / "runs")):
        pass  # 前一个结束后才允许下一个
    ok, msg, rid2 = actions.launch_cli("modeleval", ["--set", "model_ref=champion"])
    assert ok, msg
    assert "model_ref=champion" in captured[-1][0] and "modeleval" in captured[-1][0]


def test_modeleval_ref_is_restricted_by_the_safe_arg_whitelist(sandbox):
    assert not actions.launch_cli("modeleval", ["--set", "model_ref=champion; rm -rf /"])[0]
    assert not actions.launch_cli("modeleval", ["--set", "model_ref=$(whoami)"])[0]


def test_training_is_blocked_while_the_nightly_job_is_running(sandbox):
    """训练与夜间任务共用面板缓存且各占数 GB 内存。"""
    live = rr.RunReporter("nightly", run_id="n_live", root=str(sandbox / "outputs" / "runs"))
    live.step("model")
    ok, msg, rid = actions.launch_cli("train", [])
    assert not ok and "n_live" in msg and rid == "n_live"
    live.finish()


# ----------------------------------------------------------------------------- 训练实时进度
def test_task_status_shows_running_until_all_steps_finish(sandbox):
    root = str(sandbox / "outputs" / "runs")
    run = rr.RunReporter(
        "train",
        label="训练候选模型",
        steps=["data_import", "train_candidate", "gate_eval"],
        run_id="train_progress",
        root=root,
    )
    run.step("data_import", "数据导入")
    run.step_done("data_import", "5,726 只")
    run.step("train_candidate", "训练候选模型")
    run.step_done("train_candidate", "无新成熟标签，复用 champion 参数")
    run.step("gate_eval", "门控评估（→staging）")
    run.meta["started"] = time.time() - 125
    run._write(force=True)
    run.progress("候选 · 回放决策日", 3, 60)

    text = text_of(models._task_status())
    assert "正在运行 · 训练候选模型" in text
    assert "当前步骤：门控评估（→staging）" in text
    assert "任务步骤 2/3" in text
    assert "阶段进度 3/60（5%）" in text
    assert "已运行 2分" in text
    assert "候选 · 回放决策日" in text
    assert "任务完成" not in text and "最近一次训练" not in text

    run.finish("done", summary="候选已留档")
    done_text = text_of(models._task_status())
    assert "最近一次训练" in done_text and "完成" in done_text
    assert "任务步骤 3/3" in done_text and "总耗时" in done_text


def test_task_status_handles_missing_progress_and_stays_outside_body(sandbox):
    root = str(sandbox / "outputs" / "runs")
    run = rr.RunReporter("train", steps=["fit"], run_id="train_waiting", root=root)
    run.step("fit", "全量重建")
    text = text_of(models._task_status())
    assert "正在运行" in text and "等待进度上报" in text and "已运行" in text
    assert "任务完成" not in text
    run.finish()
    ids = ids_of(models.layout())
    assert "m-train-status" in ids and "m-eval-status" in ids
    assert "m-train-status" not in ids_of(models.body())


def test_task_status_marks_dead_run_interrupted_not_running(sandbox, monkeypatch):
    root = str(sandbox / "outputs" / "runs")
    run = rr.RunReporter("train", steps=["fit"], run_id="train_dead", root=root)
    run.step("fit", "全量重建")
    monkeypatch.setattr(rr, "_pid_alive", lambda _pid: False)
    text = text_of(models._task_status())
    assert "已中断" in text and "正在运行" not in text


def test_evaluation_status_is_separate_from_training(sandbox):
    root = str(sandbox / "outputs" / "runs")
    run = rr.RunReporter(
        "modeleval",
        label="信号评估",
        steps=["data_import", "signal_eval", "eval_report"],
        run_id="eval_progress",
        root=root,
    )
    run.step("data_import", "数据导入")
    run.step_done("data_import", "完成")
    run.step("signal_eval", "信号评估")
    run.progress("滚动预测中", 2, 5)
    text = text_of(models._task_status("modeleval"))
    assert "正在运行 · 信号评估" in text
    assert "任务步骤 1/3" in text and "阶段进度 2/5（40%）" in text
    assert "还没有训练运行记录" in text_of(models._task_status("train"))
    run.step_done("signal_eval", "PredIC +0.02")
    run.step("eval_report", "评估报告")
    run.step_done("eval_report", "完成")
    run.finish("done", summary="信号评估完成")
    done_text = text_of(models._task_status("modeleval"))
    assert "最近一次信号评估" in done_text and "完成" in done_text
