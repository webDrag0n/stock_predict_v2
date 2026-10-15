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
