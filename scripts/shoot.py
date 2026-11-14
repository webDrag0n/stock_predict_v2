"""截图工具：uv run --with playwright python scripts/shoot.py <out_prefix> [paths...]

用系统已装的 chromium headless shell 截各页面整页 PNG。
"""

import sys
from playwright.sync_api import sync_playwright

EXE = "/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome"
BASE = "http://127.0.0.1:8050"

prefix = sys.argv[1] if len(sys.argv) > 1 else "shot"
paths = sys.argv[2:] or ["/", "/models", "/factors", "/market"]

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1560, "height": 1000}, device_scale_factor=1.5)
    for path in paths:
        name = path.strip("/").replace("/", "_") or "home"
        pg.goto(BASE + path, wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(2500)
        out = f"/tmp/{prefix}_{name}.png"
        pg.screenshot(path=out, full_page=True)
        print(out)
    b.close()
