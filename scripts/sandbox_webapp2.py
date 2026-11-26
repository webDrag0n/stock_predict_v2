"""在临时目录里启动 webapp2：用于浏览器级测试写操作，绝不触碰真实 outputs/models。

用法：uv run python scripts/sandbox_webapp2.py <沙盒目录> <端口>
沙盒目录里放 outputs/、models/、data_cache/ 即可，缺什么页面显示什么空态。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

sandbox, port = os.path.abspath(sys.argv[1]), int(sys.argv[2])
os.environ["RUN_REGISTRY_ROOT"] = os.path.join(sandbox, "outputs", "runs")
os.environ["TRADING_SETTINGS_FILE"] = os.path.join(sandbox, "outputs", "trading_settings.json")

import config as C  # noqa: E402

# model_store.promote 按 config.MODEL_DIR 解析路径：不指到沙盒，晋升会作用于真实 models/
C.MODEL_DIR = os.path.join(sandbox, "models")

from webapp2 import actions, data  # noqa: E402

data.ROOT = sandbox
data.OUTPUTS = os.path.join(sandbox, "outputs")
data.MODELS = os.path.join(sandbox, "models")
data.FACTOR_LIBRARY = os.path.join(sandbox, "factor_library")
data.CACHE = os.path.join(sandbox, "data_cache")
data.refresh()
# 沙盒里「今天」固定为订单执行日，便于复现
actions.today_str = lambda: os.environ.get("SANDBOX_TODAY", "20261109")

from webapp2.app import app  # noqa: E402

app.run(host="127.0.0.1", port=port, debug=False)
