"""运行登记契约：CLI 独立运行，前端只读就能看到状态、进度、步骤、事件与结果。

每次运行（predict_tomorrow、walk-forward、train、daily_run.sh 的各阶段……）在
outputs/runs/<run_id>/ 下落三样东西，任何进程只靠读文件就能还原现场：

  run.json       当前快照：kind/label/status/步骤列表/当前进度/心跳/pid/产物清单。
                 原子写（临时文件 + os.replace），读到的永远是完整 JSON。
  events.jsonl   只追加的事件流，一行一个 JSON：阶段切换、告警、错误、产物登记。
  run.log        可选，原始 stdout/stderr（由调用方的 shell 重定向或 tee 产生）。

设计约束：
  · 本模块只依赖标准库，不 import dash / pipeline / config 的业务部分，CLI 与前端
    共用同一份读写代码，二者互不启动、互不依赖。
  · 写盘失败绝不影响主流程（登记是旁路）：所有对外方法吞掉 OSError 并 logger.warning。
  · 状态由「进程自己写」，不靠调度器代写；僵尸运行（进程已死但 status=running）由
    读取侧用 pid 探活 + 心跳超时判定为 stale，不改写磁盘，避免读侧与写侧竞争。
  · run_id 一律校验 ^[A-Za-z0-9_-]+$，防路径穿越。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_VERSION = 1

# 心跳写盘的最小间隔（秒）：进度回调可能每秒数十次，写盘必须节流
_MIN_WRITE_GAP = 1.0
# 读取侧判定「进程已死」的心跳超时（秒）：步骤内无进度回调的长任务也要定期写心跳
STALE_AFTER_SEC = 180.0

TERMINAL = ("done", "failed", "killed")

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def runs_root() -> str:
    """登记根目录，优先级：config.OUTPUT_DIR 被改动 > 环境变量 RUN_REGISTRY_ROOT > 默认。

    · 调用方（含测试）把 config.OUTPUT_DIR 改成别处，说明它已自行隔离输出目录，跟随它；
    · OUTPUT_DIR 仍是仓库默认值时，若设置了 RUN_REGISTRY_ROOT 就用它（测试 conftest
      借此把默认登记挡在真实 outputs/runs/ 之外，前端的运行历史不被测试污染）；
    · 都没有则是 <仓库>/outputs/runs。
    """
    default_base = os.path.join(ROOT, "outputs")
    try:
        import config as C

        base = C.OUTPUT_DIR
    except Exception:
        base = default_base
    override = os.environ.get("RUN_REGISTRY_ROOT", "").strip()
    if override and os.path.abspath(base) == os.path.abspath(default_base):
        return override
    return os.path.join(base, "runs")


def valid_run_id(run_id) -> bool:
    return bool(isinstance(run_id, str) and _RUN_ID_RE.fullmatch(run_id))


def _json_default(o: Any):
    """numpy / pandas 标量与时间戳转纯 Python；其余转字符串，保证 json.dump 不抛。"""
    try:
        import numpy as np
        import pandas as pd

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if v != v or v in (float("inf"), float("-inf")) else v
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, np.datetime64)):
            return str(pd.Timestamp(o).isoformat())
    except Exception:
        pass
    return str(o)


def _atomic_write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
    os.replace(tmp, path)


def _clean(obj):
    """递归把 NaN/inf 变 None（allow_nan=False 下 json.dump 会抛，而登记不得抛）。"""
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def new_run_id(kind: str) -> str:
    """kind_YYYYMMDD_HHMMSS，同秒冲突时由 RunReporter 加序号。"""
    return f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}"


class RunReporter:
    """一次运行的登记器。用法::

        with RunReporter("nightly", label="夜间出单", steps=["preflight", "predict", "orders"]) as run:
            run.step("preflight")
            ...
            run.progress("训练", current=3, total=10, message="窗口 3/10")
            run.artifact("orders", "outputs/orders_20261116.csv")
            run.note("warn", "候选被拒")
        # 正常退出 → done；抛异常 → failed 并记录 traceback；SystemExit(0) → done

    reporter 本身不抛异常；磁盘不可写时降级为空操作。
    """

    def __init__(
        self,
        kind: str,
        label: str = "",
        steps: list[str] | None = None,
        params: dict | None = None,
        run_id: str | None = None,
        parent: str | None = None,
        root: str | None = None,
    ):
        self.kind = kind
        self.root = root or runs_root()
        self.enabled = True
        base = run_id or new_run_id(kind)
        if not valid_run_id(base):
            raise ValueError(f"非法 run_id: {base!r}")
        rid, n = base, 1
        # 目录存在但没有 run.json = 前端预留的占位（launch_cli 先建目录放 run.log，再把
        # RUN_ID 交给 CLI），应沿用它；已有 run.json 才是真冲突，加序号。
        while os.path.exists(os.path.join(self.root, rid, "run.json")):
            rid, n = f"{base}_{n}", n + 1
        self.run_id = rid
        self.dir = os.path.join(self.root, rid)
        self._last_write = 0.0
        self._step_started: dict[str, float] = {}
        step_names = list(steps or [])
        self.meta: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "run_id": rid,
            "kind": kind,
            "label": label or kind,
            "status": "running",
            "pid": os.getpid(),
            "host": os.uname().nodename if hasattr(os, "uname") else "",
            "created": time.time(),
            "started": time.time(),
            "finished": None,
            "heartbeat": time.time(),
            "params": params or {},
            "parent": parent,
            "steps": [{"id": s, "title": s, "status": "pending"} for s in step_names],
            "current_step": None,
            "progress": None,
            "artifacts": [],
            "failure": None,
            "exit_code": None,
        }
        self._write(force=True)
        self.event("run_start", kind=kind, label=self.meta["label"])

    # ------------------------------------------------------------------ io
    def _write(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last_write < _MIN_WRITE_GAP:
            return
        self.meta["heartbeat"] = now
        try:
            _atomic_write_json(os.path.join(self.dir, "run.json"), _clean(self.meta))
            self._last_write = now
        except Exception as exc:  # 旁路：登记失败不得影响主流程
            self.enabled = False
            logger.warning("run 登记写盘失败，本次运行不再登记：%s: %s", type(exc).__name__, exc)

    def event(self, type_: str, **fields) -> None:
        """追加一条事件。events.jsonl 只增不改，前端按行 tail。"""
        if not self.enabled:
            return
        rec = {"ts": time.time(), "type": type_, **fields}
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(_clean(rec), ensure_ascii=False, default=_json_default) + "\n")
        except Exception as exc:
            self.enabled = False
            logger.warning("run 事件写盘失败：%s: %s", type(exc).__name__, exc)

    # --------------------------------------------------------------- steps
    def _find(self, step_id: str) -> dict | None:
        for s in self.meta["steps"]:
            if s["id"] == step_id:
                return s
        return None

    def step(self, step_id: str, title: str | None = None, summary: str = "") -> None:
        """进入一个步骤；上一个仍在 running 的步骤自动置 done。"""
        now = time.time()
        for s in self.meta["steps"]:
            if s["status"] == "running" and s["id"] != step_id:
                s["status"] = "done"
                s["elapsed"] = round(now - self._step_started.get(s["id"], now), 2)
        s = self._find(step_id)
        if s is None:
            s = {"id": step_id, "title": title or step_id, "status": "pending"}
            self.meta["steps"].append(s)
        if title:
            s["title"] = title
        s["status"] = "running"
        if summary:
            s["summary"] = summary
        self._step_started[step_id] = now
        self.meta["current_step"] = step_id
        self.meta["progress"] = None
        self.event("step_start", step=step_id, title=s["title"])
        self._write(force=True)

    def step_done(
        self, step_id: str | None = None, summary: str = "", status: str = "done"
    ) -> None:
        sid = step_id or self.meta.get("current_step")
        s = self._find(sid) if sid else None
        if s is None:
            return
        s["status"] = status
        s["elapsed"] = round(time.time() - self._step_started.get(sid, time.time()), 2)
        if summary:
            s["summary"] = summary
        self.event("step_done", step=sid, status=status, summary=summary, elapsed=s["elapsed"])
        self._write(force=True)

    def skip(self, step_id: str, reason: str = "") -> None:
        s = self._find(step_id)
        if s is None:
            s = {"id": step_id, "title": step_id}
            self.meta["steps"].append(s)
        s["status"] = "skipped"
        if reason:
            s["summary"] = reason
        self.event("step_skip", step=step_id, reason=reason)
        self._write(force=True)

    # ------------------------------------------------------------ progress
    def progress(
        self, phase: str, current: float = 0, total: float = 0, message: str = "", **extras
    ) -> None:
        """阶段内进度。签名与 live._emit_fn / walkforward 的 progress_cb 兼容，可直接当回调。

        写盘节流到 _MIN_WRITE_GAP；phase 变化时强制写（保证阶段切换可见）。
        """
        total = float(total or 0)
        current = float(current or 0)
        prev = self.meta.get("progress") or {}
        now = time.time()
        started = prev.get("phase_started") if prev.get("phase") == phase else now
        elapsed = now - (started or now)
        pct = (current / total) if total > 0 else 0.0
        eta = (elapsed / current * (total - current)) if current > 0 and total > current else None
        self.meta["progress"] = {
            "phase": phase,
            "current": current,
            "total": total,
            "pct": max(0.0, min(1.0, pct)),
            "message": message,
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(eta, 1) if eta is not None else None,
            "phase_started": started,
            "extras": extras,
            "ts": now,
        }
        self._write(force=(prev.get("phase") != phase))

    def callback(self):
        """返回可直接传给引擎的 progress_cb(phase=, current=, total=, message=, **extras)。"""

        def _cb(phase="", current=0, total=0, message="", **extras):
            try:
                self.progress(phase, current, total, message, **extras)
            except Exception:
                pass

        return _cb

    def heartbeat(self) -> None:
        """长时间无进度回调的步骤里手动调用，防止被读取侧误判 stale。"""
        self._write()

    # ------------------------------------------------------------ outputs
    def artifact(self, name: str, path: str, kind: str = "file", **info) -> None:
        """登记产物。path 存相对 ROOT 的路径（可移植），不存在也登记（前端显示缺失）。"""
        rel = os.path.relpath(path, ROOT) if os.path.isabs(path) else path
        self.meta["artifacts"].append({"name": name, "path": rel, "kind": kind, **info})
        self.event("artifact", name=name, path=rel, kind=kind)
        self._write(force=True)

    def note(self, level: str, message: str, **fields) -> None:
        """level ∈ info|warn|error。进入事件流，前端在「事件」里显示。"""
        self.event("note", level=level, message=message, **fields)

    def set_param(self, **kw) -> None:
        self.meta["params"].update(kw)
        self._write(force=True)

    # ----------------------------------------------------------- lifecycle
    def finish(self, status: str = "done", exit_code: int | None = None, summary: str = "") -> None:
        if self.meta["status"] in TERMINAL:
            return
        now = time.time()
        for s in self.meta["steps"]:
            if s["status"] == "running":
                s["status"] = "done" if status == "done" else "failed"
                s["elapsed"] = round(now - self._step_started.get(s["id"], now), 2)
        self.meta["status"] = status
        self.meta["finished"] = now
        self.meta["exit_code"] = exit_code
        self.meta["progress"] = None
        if summary:
            self.meta["summary"] = summary
        self.event("run_end", status=status, exit_code=exit_code, summary=summary)
        self._write(force=True)

    def fail(self, exc: BaseException, exit_code: int = 1) -> None:
        cur = self.meta.get("current_step")
        s = self._find(cur) if cur else None
        if s is not None and s["status"] == "running":
            s["status"] = "failed"
            s["elapsed"] = round(time.time() - self._step_started.get(cur, time.time()), 2)
        prog = self.meta.get("progress") or {}
        self.meta["failure"] = {
            "step": cur,
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc)[:500],
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
                -4000:
            ],
            "phase": prog.get("phase", ""),
            "ts": time.time(),
        }
        self.event("error", step=cur, exc_type=type(exc).__name__, exc_msg=str(exc)[:300])
        self.finish(
            "failed", exit_code=exit_code, summary=f"{type(exc).__name__}: {str(exc)[:120]}"
        )

    def __enter__(self) -> "RunReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.finish("done", exit_code=0)
        elif isinstance(exc, SystemExit):
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code in (None, 0) else 1)
            if code == 0:
                self.finish("done", exit_code=0)
            else:
                self.meta["failure"] = {
                    "step": self.meta.get("current_step"),
                    "exc_type": "SystemExit",
                    "exc_msg": f"退出码 {code}",
                    "traceback": "",
                    "phase": (self.meta.get("progress") or {}).get("phase", ""),
                    "ts": time.time(),
                }
                self.finish("failed", exit_code=code, summary=f"退出码 {code}")
        elif isinstance(exc, KeyboardInterrupt):
            self.finish("killed", exit_code=130, summary="被中断")
        else:
            self.fail(exc)
        return False  # 不吞异常


# ---------------------------------------------------------------------------
# 读取侧：前端与任何第三方只用这几个函数，不需要 RunReporter
# ---------------------------------------------------------------------------
def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def read_run(run_id: str, root: str | None = None) -> dict | None:
    """读取 run.json 并附加派生字段 liveness / age_sec。非法 id、缺失、损坏都返回 None。"""
    if not valid_run_id(run_id):
        return None
    p = os.path.join(root or runs_root(), run_id, "run.json")
    try:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    meta["liveness"] = liveness(meta)
    return meta


def liveness(meta: dict, now: float | None = None) -> str:
    """running 的运行是否真的活着：alive | stale | dead | finished。

    dead  = 声称 running 但 pid 已不存在（崩溃/被 kill -9，没来得及写终态）
    stale = pid 还在但心跳超过 STALE_AFTER_SEC（疑似卡死）
    """
    if meta.get("status") in TERMINAL:
        return "finished"
    now = now or time.time()
    if not _pid_alive(meta.get("pid")):
        return "dead"
    if now - float(meta.get("heartbeat") or 0) > STALE_AFTER_SEC:
        return "stale"
    return "alive"


def list_runs(kind: str | None = None, limit: int = 100, root: str | None = None) -> list[dict]:
    """最近的运行，按 created 倒序。只读 run.json，不读事件。"""
    base = root or runs_root()
    if not os.path.isdir(base):
        return []
    out = []
    for name in os.listdir(base):
        meta = read_run(name, root=base)
        if meta and (kind is None or meta.get("kind") == kind):
            out.append(meta)
    out.sort(key=lambda m: m.get("created") or 0, reverse=True)
    return out[:limit]


def read_events(run_id: str, tail: int = 200, root: str | None = None) -> list[dict]:
    """事件流的最后 tail 条；损坏行跳过。"""
    if not valid_run_id(run_id):
        return []
    p = os.path.join(root or runs_root(), run_id, "events.jsonl")
    if not os.path.exists(p):
        return []
    rows: list[dict] = []
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f.readlines()[-max(tail, 1) :]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def read_log_tail(run_id: str, n: int = 150, root: str | None = None) -> str:
    if not valid_run_id(run_id):
        return ""
    p = os.path.join(root or runs_root(), run_id, "run.log")
    if not os.path.exists(p):
        return ""
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception as exc:
        return f"(无法读取日志: {exc})"


def active_runs(root: str | None = None) -> list[dict]:
    """正在运行且确实活着的（含 stale）。前端顶部状态条用。"""
    return [
        m
        for m in list_runs(root=root, limit=50)
        if m.get("status") == "running" and m.get("liveness") in ("alive", "stale")
    ]
