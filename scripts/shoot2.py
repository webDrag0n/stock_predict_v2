"""截图 webapp2 的若干页面：uv run --with playwright python scripts/shoot2.py <前缀> <端口> <路径...>"""

import sys

from playwright.sync_api import sync_playwright

EXE = "/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome"
prefix = sys.argv[1] if len(sys.argv) > 1 else "v2"
port = sys.argv[2] if len(sys.argv) > 2 else "8060"
paths = sys.argv[3:] or ["/", "/runs"]

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=EXE, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    for path in paths:
        name = path.strip("/").replace("/", "_") or "home"
        pg.goto(f"http://127.0.0.1:{port}{path}", wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(3500)
        out = f"/tmp/{prefix}_{name}.png"
        pg.screenshot(path=out, full_page=True)
        print(out)
    b.close()
