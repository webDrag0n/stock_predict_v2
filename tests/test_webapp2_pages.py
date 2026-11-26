"""webapp2 页面：用真实 RunReporter 写出各种状态，断言页面如实呈现（不启动服务器）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

import run_registry as rr
from webapp2 import app as appmod
from webapp2 import data
from webapp2.pages import runs


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    data.refresh()
    yield tmp_path
    data.refresh()


def text_of(node) -> str:
    """递归取出 dash 组件树里的全部文字（含 children 列表与嵌套）。"""
    if node is None:
        return ""
    if isinstance(node, (str, int, float)):
        return str(node)
    if isinstance(node, (list, tuple)):
        return " ".join(text_of(c) for c in node)
    return text_of(getattr(node, "children", None))


def ids_of(node) -> set[str]:
    out: set[str] = set()
    if node is None or isinstance(node, (str, int, float)):
        return out
    if isinstance(node, (list, tuple)):
        for c in node:
            out |= ids_of(c)
        return out
    i = getattr(node, "id", None)
    if isinstance(i, str):
        out.add(i)
    return out | ids_of(getattr(node, "children", None))


def runs_root(sandbox) -> str:
    return str(sandbox / "outputs" / "runs")


# ------------------------------------------------------------------------- 列表
def test_list_view_empty_state(sandbox):
    t = text_of(runs.list_view())
    assert "还没有运行记录" in t


def test_list_view_shows_done_failed_and_running(sandbox):
    root = runs_root(sandbox)
    with rr.RunReporter("nightly", label="夜间出单", run_id="n_done", root=root) as r:
        r.step("model", "模型")
        r.step_done("model", "frozen · 预测 5485 只")
    with pytest.raises(RuntimeError):
        with rr.RunReporter("walkforward", label="WF", run_id="w_fail", root=root) as r:
            r.step("walkforward")
            raise RuntimeError("OOM 了")
    live = rr.RunReporter("train", label="训练", run_id="t_live", root=root)
    live.step("fit", "训练候选")
    live.progress("窗口", 3, 9, "w3")

    t = text_of(runs.list_view())
    for needle in ("n_done", "w_fail", "t_live", "完成", "失败", "运行中", "OOM 了", "训练候选"):
        assert needle in t, needle
    live.finish()


def test_list_view_marks_killed_process_as_interrupted(sandbox):
    """被 kill -9 的运行没写终态：页面必须显示「已中断」，而不是永远转圈的「运行中」。"""
    root = runs_root(sandbox)
    code = (
        "import sys, time; sys.path.insert(0, 'src');"
        "import run_registry as rr;"
        f"r = rr.RunReporter('nightly', run_id='victim', root={root!r});"
        "r.step('x'); print('ready', flush=True); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(data.ROOT) if False else os.getcwd(),
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        assert "运行中" in text_of(runs.list_view())
        proc.kill()
        proc.wait(timeout=10)
        t = text_of(runs.list_view())
        assert "已中断" in t and "运行中" not in t
    finally:
        proc.kill()


# ------------------------------------------------------------------------- 详情
def test_detail_running_shows_progress_steps_and_stop_button(sandbox):
    root = runs_root(sandbox)
    r = rr.RunReporter(
        "walkforward", label="WF 增量", steps=["load", "wf"], run_id="wf1", root=root
    )
    r.step("load", "加载面板")
    r.step_done("load", "5,637 只")
    r.step("wf", "滚动 walk-forward")
    r.progress("窗口", 4, 9, "第 4 窗口", pred_ic=0.0512)
    r.note("warn", "候选被拒")
    r.artifact("walkforward", "outputs/walkforward/meta.json", kind="report")

    body = runs.detail_view("wf1")
    t = text_of(body)
    for needle in (
        "WF 增量",
        "当前进度",
        "窗口",
        "4/9",
        "44%",
        "加载面板",
        "5,637 只",
        "候选被拒",
        "walkforward",
        "缺失",
    ):  # 产物文件在沙盒里不存在 → 必须显示「缺失」
        assert needle in t, needle
    assert "stop-run" in ids_of(body)
    r.finish()
    assert "stop-run" not in ids_of(runs.detail_view("wf1"))


def test_detail_failed_shows_traceback_and_hides_stop(sandbox):
    root = runs_root(sandbox)
    with pytest.raises(ValueError):
        with rr.RunReporter("train", label="训练", steps=["fit"], run_id="f1", root=root) as r:
            r.step("fit")
            raise ValueError("特征列不一致")
    body = runs.detail_view("f1")
    t = text_of(body)
    assert "失败" in t and "ValueError" in t and "特征列不一致" in t and "堆栈" in t
    assert "stop-run" not in ids_of(body)


def test_detail_dead_run_explains_why(sandbox):
    root = runs_root(sandbox)
    rr.RunReporter("nightly", run_id="ghost", root=root).step("x")
    p = os.path.join(root, "ghost", "run.json")
    meta = json.load(open(p))
    meta["pid"] = 2**22 + 777  # 不存在的进程
    json.dump(meta, open(p, "w"))
    body = runs.detail_view("ghost")
    t = text_of(body)
    assert "进程已不存在" in t and "已中断" in t
    assert "stop-run" not in ids_of(body)


def test_detail_unknown_or_malicious_id_is_safe(sandbox):
    assert "找不到该运行" in text_of(runs.detail_view("nope"))
    assert "找不到该运行" in text_of(runs.detail_view("../../etc/passwd"))


def test_detail_renders_walkforward_result_kpis(sandbox):
    root = runs_root(sandbox)
    with rr.RunReporter("walkforward", run_id="wfr", root=root) as r:
        r.set_param(
            result_全特征={
                "sharpe": 1.11,
                "pred_ic": 0.0243,
                "nw_t": 1.05,
                "n_windows": 9,
                "sharpe_pos_ratio": 0.556,
            }
        )
    t = text_of(runs.detail_view("wfr"))
    assert "1.11" in t and "0.0243" in t and "56%" in t


def test_shell_keeps_polling_and_buttons_outside_refreshed_body(sandbox):
    """轮询组件与启动按钮在外壳/列表内，runs-body 是唯一被替换的容器。"""
    shell = runs.layout("/runs/xyz")
    ids = ids_of(shell)
    assert {"runs-tick", "runs-path", "runs-body"} <= ids
    assert "launch-nightly" in ids_of(runs.layout("/runs"))


# ------------------------------------------------------------------------- 外壳
def test_app_routes_and_status_chips_with_empty_disk(sandbox):
    assert "还没有运行记录" in text_of(appmod.route("/runs"))
    assert appmod.route("/") is not None and appmod.route("/anything") is not None
    chips = text_of(appmod.status_chips())
    assert "空闲" in chips and "无 champion" in chips and "未维护" in chips


def test_status_bar_reflects_active_run_and_links_to_it(sandbox):
    root = runs_root(sandbox)
    r = rr.RunReporter("nightly", label="夜间出单", run_id="live1", root=root)
    r.step("model", "模型")
    r.progress("训练", 1, 3)
    chips = appmod.status_chips()
    t = text_of(chips)
    assert "运行中" in t and "夜间出单" in t and "训练" in t
    assert any(getattr(c, "href", None) == "/runs/live1" for c in chips)
    r.finish()
    assert "空闲" in text_of(appmod.status_chips())


def test_status_bar_account_and_champion_signals(sandbox):
    out = sandbox / "outputs"
    (out / "account.csv").write_text("cash\n30000\n")
    old = time.time() - 10 * 86400
    os.utime(out / "account.csv", (old, old))
    (sandbox / "models" / "champion").mkdir(parents=True)
    (sandbox / "models" / "champion" / "manifest.json").write_text(
        json.dumps({"model_id": "20261109_4cf5b1f338d9"})
    )
    (out / "tracking").mkdir()
    (out / "tracking" / "tracking_report.json").write_text(
        json.dumps({"champion": {"age_days": 66}})
    )
    data.refresh()
    t = text_of(appmod.status_chips())
    assert "已过期" in t and "30,000" in t and "66天" in t
    chips = appmod.status_chips()
    assert any("bad" in (getattr(c, "className", "") or "") and "66" in text_of(c) for c in chips)


def test_nav_active_matching():
    assert appmod.nav_active("/", "/") and not appmod.nav_active("/runs", "/")
    assert appmod.nav_active("/runs/abc", "/runs") and not appmod.nav_active("/runsx", "/runs")


def test_callbacks_register_and_layout_ids_unique(sandbox):
    # 回调注册不抛、顶层布局 id 不重复
    ids = []

    def collect(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            [collect(c) for c in n]
            return
        if isinstance(getattr(n, "id", None), str):
            ids.append(n.id)
        collect(getattr(n, "children", None))

    collect(appmod.app.layout)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    assert len(appmod.app.callback_map) >= 5


def test_detail_running_without_progress_still_shows_steps_and_total_elapsed(sandbox):
    root = runs_root(sandbox)
    run = rr.RunReporter("train", steps=["load", "fit"], run_id="waiting", root=root)
    run.step("load", "数据导入")
    run.step_done("load", "完成")
    run.step("fit", "全量重建")
    run.meta["started"] = time.time() - 125
    run._write(force=True)
    text = text_of(runs.detail_view("waiting"))
    assert "当前进度" in text and "等待进度上报" in text
    assert "任务步骤 1/2" in text and "已运行 2分" in text
    run.finish()
