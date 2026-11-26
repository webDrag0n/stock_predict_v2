#!/usr/bin/env bash
# ============================================================================
# start.sh —— 一键启动 QuantLab Web 面板（webapp2；CLI 不依赖它）
#
# 用法：
#   ./start.sh            # 启动面板 → http://127.0.0.1:8060（Ctrl-C 停止）
#   ./start.sh --debug    # Dash debug 模式 + 热重载（改代码自动刷新）
#   PORT=8061 ./start.sh  # 换端口；HOST=0.0.0.0 可对局域网开放
#
# 行为：
#   1. 自动同步依赖（uv sync --extra ui，已装则秒过）；
#   2. 端口已被占用时直接提示地址并退出，不重复起服务；
#   3. 前台运行，日志直接打在终端。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8060}"

# 端口已占用 → 认为服务已在运行，直接提示
if curl -s -o /dev/null --max-time 2 "http://${HOST}:${PORT}/"; then
    echo "QuantLab 已在运行 → http://${HOST}:${PORT}"
    exit 0
fi

echo "== 同步依赖（uv sync --extra ui）=="
uv sync --extra ui --quiet

echo "== 启动 QuantLab → http://${HOST}:${PORT}（Ctrl-C 停止）=="
# run_webapp2.py 用环境变量 DEBUG=1 开热重载；--debug 参数在这里翻译成它
DEBUG_FLAG=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--debug" ]; then DEBUG_FLAG=1; else ARGS+=("$a"); fi
done
exec env HOST="$HOST" PORT="$PORT" DEBUG="$DEBUG_FLAG" uv run python run_webapp2.py "${ARGS[@]}"
