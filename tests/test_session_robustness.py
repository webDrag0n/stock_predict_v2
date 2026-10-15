"""阶段 6 webapp/流水线健壮性测试：

· sid 白名单校验拒绝路径穿越（../）
· pipeline.context._jsonable 处理 pd.Timestamp / np.datetime64（含 NaT）
· reconcile_on_startup 把 pid 已死的 running 会话标记 failed

webapp.sessions 是纯 stdlib 模块（不依赖 dash），可在无 UI 环境下直接测。
"""
from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import pandas as pd
import pytest

from pipeline.context import _jsonable
from webapp import sessions


# --- sid 路径穿越 ------------------------------------------------------------
@pytest.mark.parametrize("bad", ["../etc", "..", "a/b", "a\\b", "", "a b", "a.b",
                                 "../../outputs", None, 123])
def test_valid_sid_rejects_traversal(bad):
    assert sessions.valid_sid(bad) is False


@pytest.mark.parametrize("good", ["backtest_20240101_120000", "nextday_x_1", "a-b_C"])
def test_valid_sid_accepts_whitelist(good):
    assert sessions.valid_sid(good) is True


def test_get_with_traversal_sid_returns_none():
    # 即使磁盘上存在对应文件，非法 sid 也不允许拼出 sessions 目录外的路径
    assert sessions.get("../../config") is None
    assert sessions.run_log("../x") == ""
    assert sessions.step_reports("../x") == []
    assert sessions.delete("../x") is False


# --- _jsonable 时间类型 --------------------------------------------------------
def test_jsonable_timestamp_and_datetime64():
    assert _jsonable(pd.Timestamp("2024-01-02")) == "2024-01-02T00:00:00"
    assert _jsonable(pd.Timestamp("2024-01-02 03:04:05")) == "2024-01-02T03:04:05"
    assert _jsonable(np.datetime64("2024-01-02")).startswith("2024-01-02")
    assert _jsonable(pd.NaT) is None
    assert _jsonable(np.datetime64("NaT")) is None
    # 嵌套结构里也要能转，且整体可 json 序列化
    out = _jsonable({"d": [pd.Timestamp("2024-01-02"), np.datetime64("NaT")]})
    json.dumps(out)
    assert out["d"][0] == "2024-01-02T00:00:00" and out["d"][1] is None


# --- 启动对账（僵尸 running -> failed）-----------------------------------------
def _dead_pid() -> int:
    """拿一个确定已退出的 pid：起个子进程立刻 wait，之后该 pid 探活必失败。"""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _mk_session(tmp_path, sid, meta: dict) -> None:
    sdir = tmp_path / sid
    sdir.mkdir(parents=True)
    with open(sdir / "session.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_reconcile_marks_dead_running_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", str(tmp_path))
    pid = _dead_pid()
    _mk_session(tmp_path, "backtest_zombie_1", {
        "id": "backtest_zombie_1", "mode": "backtest", "status": "running",
        "created": time.time() - 100, "started": time.time() - 90,
        "finished": 0.0, "current_step": 1, "pid": pid,
        "steps": [{"step": "data_import", "title": "数据导入", "status": "done"}],
    })
    n = sessions.reconcile_on_startup()
    assert n == 1
    meta = sessions.get("backtest_zombie_1")
    assert meta["status"] == "failed"
    assert meta["finished"] > 0
    assert meta["failure"]["exc_type"] == "ProcessDied"


def test_reconcile_keeps_alive_running(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", str(tmp_path))
    _mk_session(tmp_path, "backtest_alive_1", {
        "id": "backtest_alive_1", "mode": "backtest", "status": "running",
        "created": time.time(), "started": time.time(), "finished": 0.0,
        "current_step": 0, "pid": 1,      # init 进程必然存在
        "steps": [],
    })
    _mk_session(tmp_path, "backtest_done_1", {
        "id": "backtest_done_1", "mode": "backtest", "status": "done",
        "created": time.time(), "started": time.time(), "finished": time.time(),
        "current_step": 4, "steps": [],
    })
    assert sessions.reconcile_on_startup() == 0
    assert sessions.get("backtest_alive_1")["status"] == "running"
    assert sessions.get("backtest_done_1")["status"] == "done"
