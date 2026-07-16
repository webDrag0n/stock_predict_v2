"""Launch the QuantLab web app.

Usage:
    .venv/bin/python run_webapp.py            # serve on http://127.0.0.1:8050
    HOST=0.0.0.0 PORT=8060 .venv/bin/python run_webapp.py

The app reads only on-disk caches (no training/network in callbacks); heavy jobs
are launched as background subprocesses from the Jobs page.
"""
from __future__ import annotations

import os
import sys

# ensure project root + src are importable as the CLI scripts expect
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from webapp.app import app


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    debug = os.environ.get("DEBUG", "0") == "1"
    print(f"QuantLab → http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
