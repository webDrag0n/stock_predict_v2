"""会话管理器 —— 创建 / 调度 / 查询 / 删除会话。

设计（detached-subprocess + 日志文件模式）：
  · 状态真相落盘在 outputs/sessions/<id>/session.json（进程重启后仍可读）。
  · 进程句柄(Popen)仅内存缓存，用于 poll 退出码与 kill。
  · tick() 是调度心跳：刷新运行中进程状态，并在 running < MAX_CONCURRENT 时
    启动下一个 queued 会话。前端 dcc.Interval 周期性调用（列表页与详情页都挂）。
  · MAX_CONCURRENT 模块级可变，默认 1。
  · webapp 启动时 reconcile_on_startup() 对账：磁盘上 status=="running" 但 pid
    已死的会话标记 failed（否则重启后僵尸 running 永久阻塞队列）。
  · sid 一律校验 ^[A-Za-z0-9_-]+$，防路径穿越。
  · tick()/_launch()/delete() 持同一把调度锁，防 Dash threaded 模式双重 Popen。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(ROOT, "outputs", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)
PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = "python"

MAX_CONCURRENT = 1                 # 同时运行的会话数（前端可改）
_PROCS: dict[str, subprocess.Popen] = {}   # session_id -> Popen（内存态）
_SCHED_LOCK = threading.RLock()    # 调度锁：tick/_launch/delete 互斥（RLock：tick 内调 _launch）

_SID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# delete() 时 SIGTERM 后等待进程退出的超时（秒），超时升级 SIGKILL
_TERM_WAIT_SEC = 5.0


def valid_sid(sid) -> bool:
    """会话 ID 白名单校验（防路径穿越）。"""
    return bool(isinstance(sid, str) and _SID_RE.fullmatch(sid))


def _meta_path(sid: str) -> str:
    if not valid_sid(sid):
        raise ValueError(f"非法会话 ID: {sid!r}")
    return os.path.join(SESSIONS_DIR, sid, "session.json")


def _read(sid: str) -> dict | None:
    if not valid_sid(sid):
        return None
    p = _meta_path(sid)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write(sid: str, meta: dict) -> None:
    p = _meta_path(sid)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


# --- create ----------------------------------------------------------------
def create(mode: str, label: str, params: dict) -> str:
    """创建一个 queued 会话，写好 session.json 骨架。返回 session_id。"""
    from pipeline import registry
    ts = time.strftime("%Y%m%d_%H%M%S")
    sid = f"{mode}_{ts}"
    # 同秒去重
    base = sid
    n = 1
    while os.path.exists(os.path.join(SESSIONS_DIR, sid)):
        sid = f"{base}_{n}"
        n += 1
    sdir = os.path.join(SESSIONS_DIR, sid)
    os.makedirs(sdir, exist_ok=True)
    steps = registry.STEPS[mode]
    meta = {
        "id": sid, "mode": mode,
        "label": label or registry.MODE_LABELS.get(mode, mode),
        "params": params,
        "status": "queued",
        "created": time.time(),
        "started": 0.0, "finished": 0.0,
        "current_step": -1,
        "steps": [{"step": s, "title": t, "status": "pending"}
                  for s, t, _ in steps],
    }
    _write(sid, meta)
    return sid


# --- scheduling ------------------------------------------------------------
def _refresh_running() -> None:
    """poll 内存中运行进程的退出码，回写终态（防止 run_session 崩溃未写状态）。"""
    for sid, proc in list(_PROCS.items()):
        rc = proc.poll()
        if rc is None:
            continue
        meta = _read(sid)
        if meta and meta.get("status") == "running":
            # run_session 正常会自己写 done/failed；这里兜底崩溃情况
            meta["status"] = "done" if rc == 0 else "failed"
            meta["finished"] = meta.get("finished") or time.time()
            _write(sid, meta)
        _PROCS.pop(sid, None)


def _launch(sid: str) -> None:
    with _SCHED_LOCK:
        sdir = os.path.join(SESSIONS_DIR, sid)
        log_path = os.path.join(sdir, "run.log")
        logf = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                [PY, "-u", "run_session.py", "--session-dir", sdir],
                cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True)
        finally:
            # 子进程已继承该 FD，父进程句柄必须关闭（否则每次启动泄漏一个 FD）
            logf.close()
        _PROCS[sid] = proc
        meta = _read(sid)
        if meta:
            meta["status"] = "running"
            meta["started"] = time.time()
            meta["pid"] = proc.pid
            _write(sid, meta)


def tick() -> None:
    """调度心跳：刷新运行态 + 按并发上限启动排队会话。"""
    with _SCHED_LOCK:
        _refresh_running()
        sessions = list_sessions()
        n_running = sum(1 for s in sessions if s["status"] == "running")
        if n_running >= MAX_CONCURRENT:
            return
        # 按创建时间升序启动 queued
        queued = sorted([s for s in sessions if s["status"] == "queued"],
                        key=lambda s: s["created"])
        for s in queued:
            if n_running >= MAX_CONCURRENT:
                break
            _launch(s["id"])
            n_running += 1


def reconcile_on_startup() -> int:
    """启动对账：磁盘上 status=='running' 但 pid 已死的会话标记 failed。

    webapp 重启后 _PROCS 为空，而磁盘上可能残留上次崩溃时的 running 会话；
    MAX_CONCURRENT=1 下这种僵尸 running 会永久阻塞队列。返回处理的会话数。
    """
    n = 0
    for name in os.listdir(SESSIONS_DIR):
        meta = _read(name)          # 非法目录名 _read 直接返回 None
        if not meta or meta.get("status") != "running":
            continue
        pid = meta.get("pid")
        alive = False
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 0)     # 探活：不发信号，只查存在性
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True        # 存在但属其他用户 —— 按活着处理
            except Exception:
                alive = False
        if alive:
            continue
        meta["status"] = "failed"
        meta["finished"] = meta.get("finished") or time.time()
        meta["failure"] = meta.get("failure") or {
            "step_index": meta.get("current_step", -1),
            "step_id": "", "step_title": "",
            "exc_type": "ProcessDied",
            "exc_msg": f"webapp 启动对账：运行进程已不存在 (pid={pid})，标记失败",
            "traceback": "",
            "progress_position": "", "progress_detail": "",
            "ts": time.time(),
        }
        _write(name, meta)
        n += 1
    return n


# --- query -----------------------------------------------------------------
def list_sessions() -> list[dict]:
    out = []
    for name in os.listdir(SESSIONS_DIR):
        meta = _read(name)
        if meta:
            out.append(meta)
    return sorted(out, key=lambda s: s.get("created", 0), reverse=True)


def get(sid: str) -> dict | None:
    return _read(sid)


def step_reports(sid: str) -> list[dict]:
    """读取该会话已落盘的全部 step report（按文件名顺序）。"""
    if not valid_sid(sid):
        return []
    sdir = os.path.join(SESSIONS_DIR, sid, "steps")
    if not os.path.isdir(sdir):
        return []
    reps = []
    for fn in sorted(os.listdir(sdir)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(sdir, fn), "r", encoding="utf-8") as f:
                    reps.append(json.load(f))
            except Exception:
                pass
    return reps


def run_log(sid: str, n: int = 120) -> str:
    if not valid_sid(sid):
        return ""
    p = os.path.join(SESSIONS_DIR, sid, "run.log")
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", errors="replace") as f:
            lines = f.readlines()
        lines = [ln for ln in lines if ln.strip() not in ("'close'",)]
        return "".join(lines[-n:])
    except Exception as e:
        return f"(无法读取日志: {e})"


def elapsed(meta: dict) -> float:
    if meta.get("status") in ("done", "failed", "killed"):
        return (meta.get("finished") or meta.get("started") or 0) - (meta.get("started") or meta.get("created", 0))
    if meta.get("status") == "running":
        return time.time() - (meta.get("started") or meta.get("created", 0))
    return 0.0


# --- delete ----------------------------------------------------------------
def delete(sid: str) -> bool:
    """删除会话：运行中先 SIGTERM（带超时等待）→ 超时升级 SIGKILL，再删目录。"""
    if not valid_sid(sid):
        return False
    with _SCHED_LOCK:
        proc = _PROCS.pop(sid, None)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=_TERM_WAIT_SEC)
            except subprocess.TimeoutExpired:
                # 子进程不响应 SIGTERM，升级 SIGKILL 强杀进程组
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=_TERM_WAIT_SEC)
                except Exception:
                    pass
            except Exception:
                pass
        sdir = os.path.join(SESSIONS_DIR, sid)
        if os.path.isdir(sdir):
            try:
                shutil.rmtree(sdir)
                return True
            except Exception:
                return False
        return False
