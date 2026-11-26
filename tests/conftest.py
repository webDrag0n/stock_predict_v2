"""Shared pytest fixtures and path setup.

Ensures the project root is importable (config, src, predict_tomorrow, ...)
regardless of how pytest is invoked, replacing the per-file sys.path hacks.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_run_registry(tmp_path_factory, monkeypatch):
    """测试默认不得写真实的 outputs/runs/。

    predict_tomorrow.main / scripts.run_walkforward.main 现在会登记运行；很多既有测试
    不替换 OUTPUT_DIR，登记就会落进真实目录，污染前端的运行历史（实测出现过两条
    由测试产生的 failed 运行）。这里为每个测试设置私有的 RUN_REGISTRY_ROOT。
    测试若自己把 config.OUTPUT_DIR 改到临时目录，run_registry.runs_root 会跟随
    OUTPUT_DIR 而不是该环境变量，行为与不加此 fixture 时一致。
    """
    monkeypatch.setenv("RUN_REGISTRY_ROOT", str(tmp_path_factory.mktemp("runs_root")))
    yield
