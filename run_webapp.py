"""Launch the QuantLab web app.

Usage:
    .venv/bin/python run_webapp.py            # serve on http://127.0.0.1:8050
    .venv/bin/python run_webapp.py --debug    # Dash debug 模式 + 热重载（改代码自动刷新）
    HOST=0.0.0.0 PORT=8060 .venv/bin/python run_webapp.py

Environment:
    HOST / PORT     bind address (default 127.0.0.1:8050)
    DEBUG=1         等价于 --debug；默认关闭（生产式启动，改代码需重启进程）

The app reads only on-disk caches (no training/network in callbacks); heavy jobs
are launched as background subprocesses from the Jobs page.
"""

from __future__ import annotations

import argparse
import os
import sys

# ensure project root + src are importable as the CLI scripts expect
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the QuantLab web app.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dash debug 模式 + 热重载（等价于环境变量 DEBUG=1；默认关闭）",
    )
    args = parser.parse_args()

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8050"))
    debug = args.debug or os.environ.get("DEBUG", "0") == "1"
    if debug:
        # webapp.app 依此区分 reloader 父/子进程，避免启动对账执行两次
        os.environ["DEBUG"] = "1"

    from webapp.app import app

    print(f"QuantLab → http://{host}:{port}" + (" (debug + hot reload)" if debug else ""))
    app.run(host=host, port=port, debug=debug, dev_tools_hot_reload=debug)


if __name__ == "__main__":
    main()
