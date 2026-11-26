"""浏览器级端到端（沙盒）：模型页的 删除 / 未达门槛晋升需理由 / 达门槛晋升。
用法：uv run --with playwright python scripts/check_models_flow.py
全程只写临时目录。
"""

import json
import os
import subprocess
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8076
box = tempfile.mkdtemp(prefix="v2models_")
os.makedirs(f"{box}/outputs")


def put(sub, mid, evaluation=None, promotion=None):
    d = f"{box}/models/{sub}" if sub == "champion" else f"{box}/models/{sub}/{mid}"
    os.makedirs(d)
    man = {"model_id": mid, "train_end_date": "2026-09-03", "config_snapshot": {"feat_cols": ["a"]}}
    if promotion:
        man["promotion"] = promotion
    json.dump(man, open(f"{d}/manifest.json", "w"))
    open(f"{d}/ensemble_state.json", "w").write("{}")
    if evaluation:
        json.dump(evaluation, open(f"{d}/evaluation.json", "w"))


m = lambda s: {"sharpe": s, "max_drawdown": -8.0, "n_days": 72}  # noqa: E731
put("champion", "champ", promotion={"candidate_metrics": m(0.6), "reason": "初始"})
put(
    "staging",
    "s_bad",
    {
        "candidate": m(0.1),
        "champion": m(0.6),
        "decision": "rejected",
        "reason": "Sharpe 未超过 champion",
    },
)
put(
    "staging",
    "s_good",
    {"candidate": m(1.0), "champion": m(0.6), "decision": "promoted", "reason": "通过"},
)
put(
    "staging",
    "s_del",
    {"candidate": m(0.2), "champion": m(0.6), "decision": "rejected", "reason": "x"},
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


def champion_id():
    return json.load(open(f"{box}/models/champion/manifest.json"))["model_id"]


try:
    time.sleep(5)
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path="/home/webdrag0n/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome",
            args=["--no-sandbox"],
        )
        pg = b.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(f"http://127.0.0.1:{PORT}/models", wait_until="networkidle")
        pg.wait_for_selector("button:has-text('晋升')")

        # 1) 删除：取消不删，确认才删
        row = "div:has(> div > div > code:text-is('s_del'))"
        pg.click(f"{row} button:has-text('删除') >> nth=0")
        pg.wait_for_selector("#m-confirm-panel", state="visible")
        pg.click("#m-confirm-cancel")
        pg.wait_for_timeout(800)
        check("取消删除：目录仍在", os.path.isdir(f"{box}/models/staging/s_del"))
        pg.click(f"{row} button:has-text('删除') >> nth=0")
        pg.wait_for_selector("#m-confirm-panel", state="visible")
        pg.click("#m-confirm-run")
        pg.wait_for_timeout(1500)
        check("确认删除：目录消失", not os.path.exists(f"{box}/models/staging/s_del"))
        check("结果提示可见", "已删除" in pg.inner_text("#m-toast"))

        # 2) 未达门槛晋升：不填理由被拒；champion 不变
        row_bad = "div:has(> div > div > code:text-is('s_bad'))"
        pg.click(f"{row_bad} button:has-text('晋升') >> nth=0")
        pg.wait_for_selector("#m-confirm-panel", state="visible")
        check("未达门槛：出现理由输入框", pg.locator("#m-reason").is_visible())
        check("弹层写明未达门槛", "未达门槛" in pg.inner_text("#m-confirm-body"))
        pg.click("#m-confirm-run")
        pg.wait_for_timeout(1500)
        check("无理由晋升被拒，champion 不变", champion_id() == "champ")
        check("提示需要理由", "必须填写理由" in pg.inner_text("#m-toast"))

        # 3) 填理由后晋升成功，旧 champion 进 archive，审计里有理由
        pg.click(f"{row_bad} button:has-text('晋升') >> nth=0")
        pg.wait_for_selector("#m-confirm-panel", state="visible")
        pg.fill("#m-reason", "近三窗口为负，换成更保守的配方")
        pg.click("#m-confirm-run")
        pg.wait_for_timeout(1500)
        check("填理由后晋升成功", champion_id() == "s_bad")
        check("旧 champion 已归档", os.path.isdir(f"{box}/models/archive/champ"))
        log = open(f"{box}/models/promotions.log").read()
        check("审计记录含理由", "PROMOTE s_bad" in log and "更保守的配方" in log)

        # 4) 达门槛的候选：弹层不要求理由
        pg.wait_for_timeout(3500)  # 等轮询刷新
        row_good = "div:has(> div > div > code:text-is('s_good'))"
        pg.click(f"{row_good} button:has-text('晋升') >> nth=0")
        pg.wait_for_selector("#m-confirm-panel", state="visible")
        check("达门槛：不要求理由", not pg.locator("#m-reason").is_visible())
        pg.click("#m-confirm-cancel")
        b.close()
finally:
    srv.terminate()
    srv.wait(timeout=10)
    print("沙盒目录:", box)
print("ALL PASS" if ok else "HAS FAILURES")
sys.exit(0 if ok else 1)
