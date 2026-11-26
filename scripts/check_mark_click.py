"""浏览器级检查：点「拒绝」→ 文件落盘 → 页面显示已拒绝 → 再点撤销 → 文件恢复。
会写真实的 outputs/execution_state_<今天>.json，结束时无论成败都还原。
用法：uv run --with playwright python scripts/check_mark_click.py <port>
"""

import json
import os
import sys
from datetime import datetime

from playwright.sync_api import sync_playwright

port = sys.argv[1] if len(sys.argv) > 1 else "8061"
today = datetime.now().strftime("%Y%m%d")
state_path = os.path.join("outputs", f"execution_state_{today}.json")
existed = os.path.exists(state_path)
backup = open(state_path).read() if existed else None


def disk():
    try:
        return json.load(open(state_path))
    except OSError:
        return None


try:
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path="/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome",
            args=["--no-sandbox"],
        )
        pg = b.new_page()
        pg.goto(f"http://127.0.0.1:{port}/trade", wait_until="networkidle")
        pg.wait_for_selector("button:has-text('拒绝')")
        print("点击前 磁盘 =", disk())
        pg.click("button:has-text('拒绝') >> nth=0")
        pg.wait_for_timeout(1500)
        print("点击后 磁盘 =", disk())
        print("页面出现「已拒绝」=", pg.locator("text=✗ 已拒绝").count() > 0)
        pg.click("button:has-text('已拒绝') >> nth=0")
        pg.wait_for_timeout(1500)
        print("撤销后 磁盘 =", disk())
        print("页面「已拒绝」消失 =", pg.locator("text=✗ 已拒绝").count() == 0)
        b.close()
finally:
    if existed:
        open(state_path, "w").write(backup)
    elif os.path.exists(state_path):
        os.remove(state_path)
    print("已还原；文件存在 =", os.path.exists(state_path))
