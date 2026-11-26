"""浏览器级端到端：沙盒里走完 拒绝 → 确认成交 → 持仓写回 → 二次确认被拦 → 现金保存。
用法：uv run --with playwright python scripts/check_confirm_flow.py
全程只写临时目录。
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8071
box = tempfile.mkdtemp(prefix="v2sandbox_")
os.makedirs(f"{box}/outputs")
os.makedirs(f"{box}/data_cache/tushare")
pd.DataFrame(
    [
        ["000001.SZ", "buy", 1000, 2.0, 2.0, "再平衡:建仓", False, "2026-11-06"],
        ["600000.SH", "buy", 500, 4.0, 4.0, "再平衡:建仓", False, "2026-11-06"],
        ["600002.SH", "buy", 200, 5.0, 5.0, "涨停拦截:再平衡:建仓", True, "2026-11-06"],
    ],
    columns=[
        "symbol",
        "side",
        "shares",
        "limit_price",
        "ref_close",
        "reason",
        "blocked",
        "asof_date",
    ],
).to_csv(f"{box}/outputs/orders_20261106.csv", index=False)
pd.DataFrame({"symbol": ["000001.SZ"], "close": [2.0]}).to_parquet(
    f"{box}/data_cache/tushare/daily_20261106.parquet"
)

srv = subprocess.Popen(
    [sys.executable, "scripts/sandbox_webapp2.py", box, str(PORT)],
    cwd=ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


try:
    time.sleep(5)
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path="/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome",
            args=["--no-sandbox"],
        )
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.goto(f"http://127.0.0.1:{PORT}/trade", wait_until="networkidle")
        pg.wait_for_selector("button:has-text('拒绝')")

        # 1) 拒绝 600000.SH
        pg.click("div.orow:has-text('600000.SH') button:has-text('拒绝')")
        pg.wait_for_timeout(1500)
        state = json.load(open(f"{box}/outputs/execution_state_20261109.json"))
        check("拒绝写入执行状态文件", state == {"buy-600000.SH": "rejected"})

        # 2) 打开确认弹层：应在视口内可见，预览写明跳过 1 笔拒绝 + 1 笔拦截
        pg.click("#confirm-open")
        pg.wait_for_selector("#confirm-panel", state="visible")
        preview = pg.inner_text("#confirm-preview")
        check(
            "弹层预览：买 1，拒绝 1，拦截 1",
            "买 1" in preview and "拒绝 1" in preview and "拦截 1" in preview,
        )
        check("确认按钮在视口内", pg.locator("#confirm-run").is_visible())

        # 3) 确认写回
        pg.click("#confirm-run")
        pg.wait_for_timeout(1500)
        h = pd.read_csv(f"{box}/outputs/holdings.csv").set_index("symbol")
        check(
            "只写入未拒绝未拦截的单",
            list(h.index) == ["000001.SZ"] and h.loc["000001.SZ", "shares"] == 1000,
        )
        check("结果提示可见", "已写回持仓" in pg.inner_text("#confirm-msg"))

        # 4) 二次确认必须被拦下：按钮禁用，持仓不变
        pg.wait_for_timeout(3500)  # 等轮询把「已确认」状态刷出来
        check("确认后按钮变为禁用", pg.locator("#confirm-open").is_disabled())
        before = open(f"{box}/outputs/holdings.csv").read()
        pg.evaluate("document.getElementById('confirm-open').disabled = false")
        pg.click("#confirm-open")
        pg.wait_for_timeout(1000)
        check("强行再点也不会重复加仓", open(f"{box}/outputs/holdings.csv").read() == before)

        # 5) 现金
        pg.fill("#trade-cash", "31234.5")
        pg.click("#trade-cash-save")
        pg.wait_for_timeout(1000)
        acc = open(f"{box}/outputs/account.csv").read().splitlines()
        check("现金追加到 account.csv", acc[0] == "date,cash" and acc[-1].endswith(",31234.50"))
        b.close()
finally:
    srv.terminate()
    srv.wait(timeout=10)
    print("沙盒目录:", box)
print("ALL PASS" if ok else "HAS FAILURES")
sys.exit(0 if ok else 1)
