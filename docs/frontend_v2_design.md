# 前端 v2 设计（CLI 独立运行，前端可见全部状态）

状态：已完成。旧前端 `webapp/`、会话机制（`run_session.py`、`pipeline/scheduler.py`、`pipeline/stages.py`）与只测它们的旧测试已删除。

## 1. 原则

1. **CLI 不依赖前端**：`predict_tomorrow.py`、`scripts/run_walkforward.py`、`daily_run.sh` 在没有任何 webapp 进程时照常运行。
2. **前端不持有业务逻辑**：前端只做两件事，读 `outputs/` 与 `models/` 下的文件，或调用与 CLI 相同的函数/入口。
3. **同一份落盘契约**：CLI 与前端用 `src/run_registry.py` 定义的 `outputs/runs/<run_id>/` 对话，二者互不启动、互不 import 对方。
4. **写操作要么是纯函数，要么是子进程 CLI**：这样 CLI 与前端的行为逐位一致，也可以单测。

## 2. 落盘契约（已实现，`src/run_registry.py`）

```
outputs/runs/<run_id>/
  run.json       快照：kind / label / status / steps[] / progress / artifacts[] / failure / heartbeat / pid
  events.jsonl   只追加事件：run_start / step_start / step_done / artifact / note / error / run_end
  run.log        可选：原始 stdout（由调用方重定向）
```

- `status ∈ running | done | failed | killed`。
- `liveness` 由读取侧派生，不写盘：`alive`（pid 在且心跳新）/ `stale`（pid 在但心跳超 180 秒）/ `dead`（声称 running 但 pid 已不存在，如 kill -9）/ `finished`。
- 写盘失败只降级，不影响主流程；`RUN_REGISTRY=0` 可整体关闭。

已接入：`predict_tomorrow.py`（kind=`nightly|tracking|commit|promote`）、`scripts/run_walkforward.py`（kind=`walkforward`）。

## 3. 前端写操作清单（旧功能 → 新前端的实现方式）

| 旧功能 | 新前端的做法 | 与 CLI 的关系 |
|---|---|---|
| 生成次日订单 | 子进程 `predict_tomorrow.py ...` | 就是 CLI 本身 |
| 成交确认 | 调用 `execution.save_holdings` / `save_holdings_from_fills` | 与 `--commit` 共用同一函数 |
| 现金维护 | 调用 `dataio.append_account_cash`（迁入 `src/account_io.py`） | CLI 体检读同一文件 |
| 执行勾选 / 拒绝 | 写 `outputs/execution_state_<日>.json` | `--commit` 已读取 |
| 晋升候选 | 调用 `model_store.promote` | 与 `--promote` 共用 |
| 因子草稿校验与保存 | 迁入 `src/factor_draft.py`（纯函数） | 前端与脚本共用 |
| 发起 walk-forward / 训练 / 因子评估 | 子进程 CLI | 就是 CLI 本身 |
| 终止运行 | 向 `run.json.pid` 发 SIGTERM | 读侧用 `liveness` 确认 |

## 4. 页面（共 7 个）

1. **总览**：今日状态灯、最新订单、champion 与年龄、风控五层、告警、最近运行。
2. **运行**：全部运行列表；单次运行详情（步骤、实时进度、事件、日志尾部、产物、失败堆栈）。
3. **交易**：订单执行、成交确认、持仓与账户、现金维护。
4. **模型**：champion / staging / archive、门控评估、晋升、配方指纹与特征差异。
5. **回测**：walk-forward 结果、窗口表、多口径对比、成本敏感性、试验与 deflated Sharpe、留出段。
6. **因子**：因子库、评估结果、详情、公式编辑器。
7. **行情**：保留旧行情浏览器的只读功能。

## 5. 技术

继续用 Dash + Plotly（沿用 `webapp/theme.py` 的调色板，改为冷灰浅色）。页面一律是 `layout()` 加 `register(app)`，数据一律经 `webapp2/data.py` 读取，页面里不直接读文件。

## 6. 分阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| 1 | 落盘契约 + CLI 接入 + 读取层 + 外壳 + 总览 + 运行 | 不启动 webapp 跑 CLI，前端能显示进度；测试全绿 |
| 2 | 交易（含成交确认、现金维护、执行勾选）+ 模型（含晋升） | 写操作与 CLI 共用函数，有测试 |
| 3 | 回测 + 因子（含编辑器）+ 行情 | 覆盖旧页面全部功能 |
| 4 | 对照旧前端逐项核对，通过后删除 `webapp/` 与旧测试 | 已完成 |
