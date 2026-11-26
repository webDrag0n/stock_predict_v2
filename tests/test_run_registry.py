"""run 登记契约：CLI 与前端只通过 outputs/runs/<id>/ 下的文件对话。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import run_registry as rr


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_lifecycle_done_writes_snapshot_and_events(tmp_path):
    with rr.RunReporter("nightly", label="夜间出单", steps=["a", "b"], root=str(tmp_path)) as run:
        run.step("a", title="步骤 A")
        run.progress("训练", current=2, total=4, message="窗口 2/4")
        run.step("b")
        run.artifact("orders", "outputs/orders_x.csv")
        rid = run.run_id

    meta = rr.read_run(rid, root=str(tmp_path))
    assert meta["status"] == "done" and meta["exit_code"] == 0
    assert meta["liveness"] == "finished"
    assert [s["status"] for s in meta["steps"]] == ["done", "done"]
    assert meta["steps"][0]["title"] == "步骤 A"
    assert meta["progress"] is None
    assert meta["artifacts"][0]["name"] == "orders"
    types = [e["type"] for e in rr.read_events(rid, root=str(tmp_path))]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert {"step_start", "artifact"} <= set(types)


def test_exception_marks_failed_with_traceback_and_does_not_swallow(tmp_path):
    with pytest.raises(ValueError):
        with rr.RunReporter("train", steps=["fit"], root=str(tmp_path)) as run:
            run.step("fit")
            raise ValueError("boom")
    meta = rr.read_run(run.run_id, root=str(tmp_path))
    assert meta["status"] == "failed" and meta["exit_code"] == 1
    assert meta["failure"]["exc_type"] == "ValueError"
    assert "boom" in meta["failure"]["traceback"]
    assert meta["steps"][0]["status"] == "failed"


def test_system_exit_codes(tmp_path):
    with pytest.raises(SystemExit):
        with rr.RunReporter("nightly", root=str(tmp_path)) as run:
            raise SystemExit(3)
    meta = rr.read_run(run.run_id, root=str(tmp_path))
    assert meta["status"] == "failed" and meta["exit_code"] == 3

    with pytest.raises(SystemExit):
        with rr.RunReporter("nightly", root=str(tmp_path)) as run0:
            raise SystemExit(0)
    assert rr.read_run(run0.run_id, root=str(tmp_path))["status"] == "done"


def test_keyboard_interrupt_is_killed(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with rr.RunReporter("nightly", root=str(tmp_path)) as run:
            raise KeyboardInterrupt()
    assert rr.read_run(run.run_id, root=str(tmp_path))["status"] == "killed"


def test_numpy_and_nan_values_do_not_break_registry(tmp_path):
    with rr.RunReporter(
        "nightly", params={"x": np.float32(1.5), "y": float("nan")}, root=str(tmp_path)
    ) as run:
        run.progress("p", current=np.int64(3), total=np.int64(6), message="m", z=np.float64("inf"))
        run.set_param(arr=np.arange(3))
    meta = rr.read_run(run.run_id, root=str(tmp_path))
    assert meta["params"]["x"] == 1.5 and meta["params"]["y"] is None
    assert meta["params"]["arr"] == [0, 1, 2]


def test_progress_throttles_writes_but_phase_change_is_forced(tmp_path):
    with rr.RunReporter("nightly", root=str(tmp_path)) as run:
        run.progress("A", 1, 10)
        t0 = os.path.getmtime(os.path.join(run.dir, "run.json"))
        for i in range(2, 8):
            run.progress("A", i, 10)  # 同阶段高频帧：节流，磁盘快照不更新
        assert os.path.getmtime(os.path.join(run.dir, "run.json")) == t0
        run.progress("B", 1, 5)  # 阶段切换必须落盘
        assert _read_json(os.path.join(run.dir, "run.json"))["progress"]["phase"] == "B"


def test_progress_pct_and_eta(tmp_path):
    with rr.RunReporter("nightly", root=str(tmp_path)) as run:
        run.progress("训练", 5, 10)
        p = run.meta["progress"]
        assert p["pct"] == pytest.approx(0.5)
        run.progress("训练", 10, 0)  # total 未知
        assert run.meta["progress"]["pct"] == 0.0


def test_callback_is_compatible_with_engine_progress_signature(tmp_path):
    with rr.RunReporter("walkforward", root=str(tmp_path)) as run:
        cb = run.callback()
        cb(phase="窗口", current=1, total=9, message="w1", pred_ic=0.05)
        cb()  # 全默认参数也不得抛
        assert run.meta["progress"] is not None


def test_unwritable_root_degrades_to_noop(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    run = rr.RunReporter("nightly", root=str(blocker / "sub"))  # 不能在文件下建目录
    run.step("a")
    run.progress("p", 1, 2)
    run.finish()  # 全程不抛
    assert run.enabled is False


def test_run_id_validation_and_collision(tmp_path):
    with pytest.raises(ValueError):
        rr.RunReporter("nightly", run_id="../evil", root=str(tmp_path))
    a = rr.RunReporter("nightly", run_id="same", root=str(tmp_path))
    b = rr.RunReporter("nightly", run_id="same", root=str(tmp_path))
    assert a.run_id == "same" and b.run_id == "same_1"
    assert rr.read_run("../x", root=str(tmp_path)) is None
    assert rr.read_events("a/b", root=str(tmp_path)) == []


def test_liveness_dead_stale_alive(tmp_path):
    base = {"status": "running", "heartbeat": time.time(), "pid": os.getpid()}
    assert rr.liveness(base) == "alive"
    assert rr.liveness({**base, "heartbeat": time.time() - rr.STALE_AFTER_SEC - 5}) == "stale"
    assert rr.liveness({**base, "pid": 2**22 + 12345}) == "dead"
    assert rr.liveness({**base, "pid": None}) == "dead"
    assert rr.liveness({**base, "status": "done"}) == "finished"


def test_killed_process_is_reported_dead_without_writing(tmp_path):
    """真实场景：子进程被 kill -9，没机会写终态。读取侧应判 dead，且不改写磁盘。"""
    code = (
        "import sys, time; sys.path.insert(0, 'src');"
        "import run_registry as rr;"
        f"r = rr.RunReporter('nightly', run_id='victim', root={str(tmp_path)!r});"
        "r.step('x'); print('ready', flush=True); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        meta = rr.read_run("victim", root=str(tmp_path))
        assert meta["status"] == "running" and meta["liveness"] == "alive"
        proc.kill()
        proc.wait(timeout=10)
        before = os.path.getmtime(tmp_path / "victim" / "run.json")
        meta = rr.read_run("victim", root=str(tmp_path))
        assert meta["status"] == "running" and meta["liveness"] == "dead"
        assert os.path.getmtime(tmp_path / "victim" / "run.json") == before
        assert rr.active_runs(root=str(tmp_path)) == []
    finally:
        proc.kill()


def test_list_runs_filters_sorts_and_skips_corrupt(tmp_path):
    for i, kind in enumerate(["nightly", "train", "nightly"]):
        with rr.RunReporter(kind, run_id=f"r{i}", root=str(tmp_path)) as run:
            run.meta["created"] = 1000 + i
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "run.json").write_text("{not json")
    (tmp_path / "stray.txt").write_text("x")
    all_ids = [m["run_id"] for m in rr.list_runs(root=str(tmp_path))]
    assert all_ids == ["r2", "r1", "r0"]
    assert [m["run_id"] for m in rr.list_runs("nightly", root=str(tmp_path))] == ["r2", "r0"]
    assert len(rr.list_runs(limit=1, root=str(tmp_path))) == 1


def test_read_events_tail_and_corrupt_line(tmp_path):
    with rr.RunReporter("nightly", run_id="e", root=str(tmp_path)) as run:
        for i in range(30):
            run.note("info", f"m{i}")
    with open(tmp_path / "e" / "events.jsonl", "a", encoding="utf-8") as f:
        f.write("{broken\n")
    ev = rr.read_events("e", tail=5, root=str(tmp_path))
    assert len(ev) <= 5 and all("type" in e for e in ev)


def test_read_log_tail(tmp_path):
    with rr.RunReporter("nightly", run_id="l", root=str(tmp_path)):
        pass
    (tmp_path / "l" / "run.log").write_text("\n".join(f"line{i}" for i in range(500)))
    tail = rr.read_log_tail("l", n=3, root=str(tmp_path))
    assert tail.strip().splitlines() == ["line497", "line498", "line499"]
    assert rr.read_log_tail("nope", root=str(tmp_path)) == ""


def test_reporter_adopts_frontend_reserved_directory(tmp_path):
    """前端先建 outputs/runs/<rid>/（放 run.log）再把 RUN_ID 交给 CLI：CLI 应沿用该 id，
    否则前端跳转的 /runs/<rid> 永远找不到运行。已有 run.json 才算冲突并加序号。"""
    reserved = tmp_path / "walkforward_x"
    reserved.mkdir()
    (reserved / "run.log").write_text("starting\n")
    a = rr.RunReporter("walkforward", run_id="walkforward_x", root=str(tmp_path))
    assert a.run_id == "walkforward_x"
    assert (reserved / "run.json").exists() and (reserved / "run.log").exists()
    b = rr.RunReporter("walkforward", run_id="walkforward_x", root=str(tmp_path))
    assert b.run_id == "walkforward_x_1"


def test_runs_root_precedence(monkeypatch, tmp_path):
    """OUTPUT_DIR 被调用方改走 → 跟随它；仍是默认值时 RUN_REGISTRY_ROOT 才生效。"""
    import config as C

    monkeypatch.delenv("RUN_REGISTRY_ROOT", raising=False)
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path / "out"))
    assert rr.runs_root() == str(tmp_path / "out" / "runs")

    monkeypatch.setenv("RUN_REGISTRY_ROOT", str(tmp_path / "forced"))
    assert rr.runs_root() == str(tmp_path / "out" / "runs")  # 调用方已自行隔离，不被覆盖

    monkeypatch.setattr(C, "OUTPUT_DIR", os.path.join(rr.ROOT, "outputs"))
    assert rr.runs_root() == str(tmp_path / "forced")  # 默认 OUTPUT_DIR + 环境变量 → 环境变量


def test_suite_does_not_write_into_real_outputs_runs():
    """conftest 的 autouse fixture 必须把默认登记挡在真实 outputs/runs 之外。"""
    real = os.path.join(rr.ROOT, "outputs", "runs")
    assert os.path.abspath(rr.runs_root()) != os.path.abspath(real)
