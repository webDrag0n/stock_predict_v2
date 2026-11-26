"""浏览器级端到端（沙盒）：公式编辑器 校验 → 拒绝危险公式 → 保存草稿 → 重复被拒。
用法：uv run --with playwright python scripts/check_factor_editor_flow.py
全程只写临时目录；不点「立即评估」（那会真的启动 CLI）。
"""

import os
import subprocess
import sys
import tempfile
import time

import yaml
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8078
box = tempfile.mkdtemp(prefix="v2factor_")
os.makedirs(f"{box}/outputs")
os.makedirs(f"{box}/factor_library")
open(f"{box}/factor_library/base.yaml", "w").write(
    yaml.safe_dump(
        {"factors": [{"id": "existing_one", "name": "x", "formula": "close", "enabled": True}]}
    )
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


def fill(pg, fid, name, formula):
    pg.fill("#fe-id", fid)
    pg.fill("#fe-name", name)
    pg.fill("#fe-formula", formula)


try:
    time.sleep(5)
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path="/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome",
            args=["--no-sandbox"],
        )
        pg = b.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(f"http://127.0.0.1:{PORT}/factors/new", wait_until="networkidle")

        fill(pg, "evil", "危险", "__import__('os').system('echo hi')")
        pg.click("#fe-validate")
        pg.wait_for_timeout(1000)
        check("危险公式：校验未通过", "校验未通过" in pg.inner_text("#fe-result"))
        pg.click("#fe-save")
        pg.wait_for_timeout(1000)
        check(
            "危险公式：保存被拒且没写文件", not os.path.exists(f"{box}/factor_library/custom.yaml")
        )

        fill(pg, "my_mom5", "五日动量", "close / ts_delay(close, 5) - 1")
        pg.click("#fe-validate")
        pg.wait_for_timeout(1000)
        check(
            "合法公式：校验通过并列出依赖",
            "校验通过" in pg.inner_text("#fe-result") and "close" in pg.inner_text("#fe-result"),
        )

        pg.click("#fe-save")
        pg.wait_for_timeout(1500)
        saved = yaml.safe_load(open(f"{box}/factor_library/custom.yaml"))["factors"]
        check(
            "保存：写入 custom.yaml 且 enabled=false",
            [f["id"] for f in saved] == ["my_mom5"] and saved[0]["enabled"] is False,
        )
        check("保存后提示面板缓存会重建", "全量重建" in pg.inner_text("#fe-result"))
        check("保存后出现「立即评估」按钮", pg.locator("#fe-eval-now").count() == 1)
        check(
            "其他库文件未被改动",
            yaml.safe_load(open(f"{box}/factor_library/base.yaml"))["factors"][0]["id"]
            == "existing_one",
        )

        # 重复：同名再存一次、与其他库文件里的 id 重名，都必须被拒
        pg.click("#fe-save")
        pg.wait_for_timeout(1200)
        check("重复 id：被拒", "已存在" in pg.inner_text("#fe-result"))
        fill(pg, "existing_one", "撞名", "close")
        pg.click("#fe-save")
        pg.wait_for_timeout(1200)
        check("与其他库文件撞名：被拒", "已存在" in pg.inner_text("#fe-result"))
        ids = [
            f["id"] for f in yaml.safe_load(open(f"{box}/factor_library/custom.yaml"))["factors"]
        ]
        check("被拒的保存没有污染文件", ids == ["my_mom5"])

        # 总览能看到刚存的草稿
        pg.goto(f"http://127.0.0.1:{PORT}/factors", wait_until="networkidle")
        check("总览主表出现草稿", "my_mom5" in pg.inner_text("#fw-table"))
        b.close()
finally:
    srv.terminate()
    srv.wait(timeout=10)
    print("沙盒目录:", box)
print("ALL PASS" if ok else "HAS FAILURES")
sys.exit(0 if ok else 1)
