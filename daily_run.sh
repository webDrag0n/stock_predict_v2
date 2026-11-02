#!/usr/bin/env bash
# ============================================================================
# daily_run.sh —— 每个交易日收盘后的实盘例行（3 万本金配置）
#
# 用法：
#   ./daily_run.sh           # 出单模式：交易日门控 → 更新数据 → 生成明日订单
#   ./daily_run.sh commit    # 确认今日订单已成交后：目标持仓写回 holdings.csv
#
# 建议 cron（周一至周五 18:10；tushare 日线约 17:00 就绪，hk_hold 晚间披露）：
#   10 18 * * 1-5  cd /home/webdrag0n/stock_predict_v2 && ./daily_run.sh
#
# 数据更新（日线/估值/北向增量）在面板加载时自动完成；自适应重训与
# champion/staging 门控由 predict_tomorrow 内部处理（默认每晚先评估
# 是否触发重训，触发才训练，否则冻结 champion 直接出单）。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- 3 万本金参数 -----------------------------------------------------------
# 最低佣金 5 元/笔是小账户的主要成本约束：单笔成交额 ≥5000 元才能把单边
# 佣金压到 ≤0.1%。3 万 / 6 仓 ≈ 5000 元/仓，正好卡在阈值上。
CAPITAL=30000
MAX_POS=6          # 持仓数：再多 → 单笔金额跌破最低佣金甜点区
MAX_WEIGHT=0.20    # 6 仓等权 ≈16.7%，上限留一点头部空间
MIN_NOTIONAL=2000  # 调仓差单下限：低于 2000 元的碎单佣金占比 ≥0.25%，不值得
# 风控五层（vol 目标/回撤刹车/IC 降仓/IV 降仓）保持默认全开：风控对照实验
# （outputs/ablations/walkforward_riskoff）证明 Sharpe +2.48 中约 +0.98
# 来自风控层——小本金更亏不起回撤。
# 若你的券商免五（无 5 元最低佣金），可把 MAX_POS 提到 10、
# MIN_NOTIONAL 降到 1000，分散度更好。

MODE="${1:-orders}"
TODAY="$(date +%Y%m%d)"
LOG_DIR="outputs/job_logs"
LOG="$LOG_DIR/daily_${TODAY}.log"
mkdir -p "$LOG_DIR"

if [ "$MODE" = "commit" ]; then
    echo "== $(date '+%F %T') 成交确认：写回 holdings.csv ==" | tee -a "$LOG"
    uv run python predict_tomorrow.py --commit 2>&1 | tee -a "$LOG"
    echo "提示：如现金有变动（分红/出入金），请同步维护 outputs/account.csv 的 cash 列。"
    exit 0
fi

if [ "$MODE" != "orders" ]; then
    echo "未知模式: $MODE（可用: orders | commit）" >&2
    exit 64
fi

# --- 0) 交易日门控：非交易日 / 日历故障都不出单 -----------------------------
# 日历故障时宁可跳过（陈旧数据出单比漏一单更危险）。
if ! uv run python - >>"$LOG" 2>&1 <<'PY'
import datetime
import sys

sys.path.insert(0, "src")
from data.tushare import TushareConnector

today = datetime.date.today().isoformat()
try:
    ok = TushareConnector().is_trading_day(today)
except Exception as e:
    print(f"交易日历检查失败（{e}）——为安全起见今晚跳过，避免用陈旧数据出单")
    sys.exit(2)
if not ok:
    print(f"{today} 非交易日，跳过")
    sys.exit(1)
print(f"{today} 是交易日，继续")
PY
then
    echo "== $(date '+%F %T') 非交易日或日历不可用，跳过（详见 $LOG）"
    exit 0
fi

# --- 1) 生成明日订单 ---------------------------------------------------------
echo "== $(date '+%F %T') 生成明日订单（日志 $LOG）=="
uv run python predict_tomorrow.py \
    --capital "$CAPITAL" \
    --max-positions "$MAX_POS" \
    --max-weight "$MAX_WEIGHT" \
    --min-notional "$MIN_NOTIONAL" \
    2>&1 | tee -a "$LOG"

echo
echo "== 完成。明日开盘前查看 outputs/ 下最新的 orders_*.csv 执行订单；"
echo "   成交后运行 ./daily_run.sh commit 写回持仓。 =="
