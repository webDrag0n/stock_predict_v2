"""浏览器级检查：展开「分析」后等待轮询刷新，面板是否仍然展开。
用法：uv run --with playwright python scripts/check_details_poll.py <port>
"""

import sys

from playwright.sync_api import sync_playwright

port = sys.argv[1] if len(sys.argv) > 1 else "8061"
with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path="/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome",
        args=["--no-sandbox"],
    )
    pg = b.new_page()
    pg.goto(f"http://127.0.0.1:{port}/trade", wait_until="networkidle")
    pg.wait_for_selector("details.why summary")
    pg.click("details.why summary")
    print("展开后 open =", pg.eval_on_selector("details.why", "e => e.open"))
    pg.wait_for_timeout(7000)  # 跨过两次 3 秒轮询
    print("等待 7 秒后 open =", pg.eval_on_selector("details.why", "e => e.open"))
    b.close()
