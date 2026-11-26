"""启动 QuantLab v2（只读文件 + 触发 CLI），默认 http://127.0.0.1:8060。

与旧的 run_webapp.py（8050）互不影响，可并行。

    uv run python run_webapp2.py
    HOST=0.0.0.0 PORT=8061 uv run python run_webapp2.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8060"))
    from webapp2.app import app

    print(f"QuantLab v2 → http://{host}:{port}")
    app.run(host=host, port=port, debug=os.environ.get("DEBUG", "0") == "1")


if __name__ == "__main__":
    main()
