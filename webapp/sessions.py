"""会话管理器 —— 薄封装：调度核心已迁入 pipeline/scheduler.py（调度独立化，
供 webapp 与未来的 CLI/TUI 共用）。本模块仅 re-export，保证旧 import 路径
（webapp.sessions / from .. import sessions）不破。

注意：测试若需 monkeypatch SESSIONS_DIR 等模块态，应直接 patch
pipeline.scheduler（本模块的同名属性只是引用副本，互不影响）。
"""

from __future__ import annotations

from pipeline.scheduler import (  # noqa: F401
    MAX_CONCURRENT,
    PY,
    ROOT,
    SESSIONS_DIR,
    concurrency,
    create,
    delete,
    elapsed,
    get,
    list_sessions,
    reconcile_on_startup,
    run_log,
    step_reports,
    tick,
    valid_sid,
)
