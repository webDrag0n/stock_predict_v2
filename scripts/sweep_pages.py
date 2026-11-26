"""浏览器级巡检：逐页打开，捕获 JS 异常、4xx/5xx 请求、Dash 错误横幅、页面内的 Traceback 文本。
用法：uv run --with playwright python scripts/sweep_pages.py [port]
"""

import sys

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8061"
PAGES = [
    "/",
    "/trade",
    "/models",
    "/backtest",
    "/factors",
    "/factors/new",
    "/factors/ep",
    "/factors/gtja_alpha001_formula",
    "/factors/ghost",
    "/market",
    "/market?symbol=300750.SZ",
    "/runs",
    "/runs/demo_live",
    "/runs/nope",
]

with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path="/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome",
        args=["--no-sandbox"],
    )
    bad = 0
    for path in PAGES:
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        errs, fails = [], []
        pg.on("pageerror", lambda e: errs.append(str(e)[:120]))
        pg.on(
            "response",
            lambda r: (
                fails.append(f"{r.status} {r.url[-60:]}")
                if r.status >= 400 and "favicon" not in r.url
                else None
            ),
        )
        pg.goto(f"http://127.0.0.1:{PORT}{path}", wait_until="networkidle", timeout=60000)
        pg.wait_for_timeout(2500)
        banner = pg.locator(".dash-fe-error-banner, .dash-debug-menu").count()
        text = pg.inner_text("#page-content")
        broke = any(k in text for k in ("Traceback", "Callback error", "Internal Server Error"))
        ok = not errs and not fails and not banner and not broke and len(text) > 40
        bad += not ok
        print(
            ("OK   " if ok else "FAIL ")
            + f"{path:34s} chars={len(text):5d}"
            + (f" JS={errs[:1]}" if errs else "")
            + (f" HTTP={fails[:2]}" if fails else "")
            + (" BROKEN-TEXT" if broke else "")
        )
        pg.close()
    b.close()
print("SWEEP", "CLEAN" if not bad else f"{bad} PAGE(S) WITH PROBLEMS")
sys.exit(1 if bad else 0)
