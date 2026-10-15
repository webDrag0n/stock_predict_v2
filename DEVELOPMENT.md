# DEVELOPMENT.md — stock-predict-v2 开发指南

本文档面向开发者与研究者：如何搭建环境、跑测试、复现结果、调试问题、扩展功能。

---

## 1. 环境搭建

### 1.1 依赖管理：uv（推荐）

项目使用 `uv` 作为 Python 包管理器。如未安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后创建虚拟环境并安装依赖：

```bash
uv venv
uv pip install -e .
# 若需要前端 + 完整模型库
uv pip install -e ".[full,ui,dev]"
```

所有后续命令默认在虚拟环境中执行：

```bash
uv run python ...
# 或
source .venv/bin/activate
python ...
```

### 1.2 凭据

Tushare token 放在 `config_credentials.toml`：

```toml
[tushare]
token = "your_token_here"
```

或用环境变量：

```bash
export TUSHARE_TOKEN="your_token_here"
```

 free-tier token 有接口权限与限速限制，全市场模式基本可行，但基本面/财务/期权链可能受限。详见记忆 `stock-predict-v2-tushare-token-limits`。

### 1.3 验证安装

```bash
uv run python -c "import config; print(config.TUSHARE_TOKEN[:4])"
uv run pytest tests/test_backtest_synthetic.py -v
```

---

## 2. 核心开发工作流

### 2.1 跑一个最小回测

用合成数据验证引擎正确性：

```bash
uv run python tests/test_backtest_synthetic.py
```

这会跑一个 80 天、2 只股票的微观面板，验证现金守恒、T+1 锁、订单生成。

### 2.2 跑全市场 walk-forward

```bash
FORWARD_WINDOW=20 TRAIN_WINDOW=500 MODEL_OBJECTIVE=ret_weighted \
UNIVERSE_MODE=all USE_IMPACT_COST=1 \
uv run python scripts/run_walkforward.py
```

首次会从 tushare 拉取全市场日线并缓存，耗时较长（取决于 token 限速）。产物在 `outputs/`。

### 2.3 跑前端会话

```bash
uv run python run_webapp.py
```

默认监听 `8050`。创建 backtest/nextday 会话后，session 在独立子进程运行；完成后刷新查看步骤报告。

### 2.4 生成次日订单

```bash
uv run python predict_tomorrow.py
```

读取 `holdings.csv`（格式见 `src/trading.py:parse_initial_holdings`），输出 `outputs/orders_*.csv`。

---

## 3. 测试

### 3.1 运行全部测试

```bash
uv run pytest tests/ -v
```

### 3.2 关键测试说明

| 测试 | 作用 |
|---|---|
| `test_backtest_synthetic.py` | 微观合成数据验证 backtest_daily 引擎正确性 |
| `test_backtest_periodic.py` | 周期调仓引擎：成本公式与 trade_costs 一致、年化指标口径 |
| `test_backtest_regression.py` | 真实数据冒烟+守恒断言（非数值基准比对，缓存缺失时 skip） |
| `test_nextday_cli_consistency.py` | 次日订单路径与回测路径一致性 |
| `test_sim2real_fixes.py` | 实盘化修正（复权、成本、T+1 等）验证 |

新增代码应优先补充到 `tests/`。涉及交易的逻辑必须有合成数据覆盖；涉及模型训练的逻辑用固定 seed 与小型面板保证可复现。

### 3.3 手动验证新改动

任何可能影响回测数字的改动，都应该跑一个最小全市场/或小样本 walk-forward 对比前后差异：

```bash
# 改动前保存基准
uv run python scripts/run_walkforward.py --out outputs/before.json
# 改动后
uv run python scripts/run_walkforward.py --out outputs/after.json
# 对比 Sharpe / alpha / IC
```

---

## 4. 调试与排错

### 4.1 缓存问题

如果修改了因子工程、目标计算、截面归一化等，但结果没变，很可能是缓存命中：

```bash
rm -rf data_cache/dataset_*.parquet
```

`load_data` 的缓存签名包含主要因子开关，但不包含所有实现细节。调试新逻辑时养成先清缓存的习惯。

### 4.2 数据拉取失败

- tushare free-tier 对 `daily_basic` 限速 1 次/分钟，全市场模式会自动跳过基本面。
- 若某天全市场拉取中断，可删除对应日缓存或清空 `data_cache/tushare/` 重跑。
- 用 `OPTION_SOURCE=akshare` 作为 IV 数据回退。

### 4.3 内存爆炸

全市场 ~5700 只 × 500 日 × 150 因子 ≈ 大几百 MB。常见原因：

- 忘记 `reduce_mem_usage`
- `groupby().apply(..., include_groups=False)` 返回高维对象
- 把原始数组直接塞进 JSON（见 `pipeline/context.py:fig_hist` 的教训）

使用 `float32` 而非 `float64`，必要时对 panel 做列裁剪。

### 4.4 回测结果“太好”

按 checklist 逐项排查：

1. 是否用了 `close` 而非 `adj_close` 算目标？
2. 是否 `EMBARGO_DAYS < FORWARD_WINDOW`？
3. 是否引入了"每天赚 pos×N日收益"的重叠口径？（旧的每日重叠引擎
   `run_backtest` 与 `PERIODIC_REBALANCE` 开关已删除，周期调仓是唯一口径）
4. 是否在小股票池 (49/300) 上跑？
5. 是否在测试集上选策略/超参？
6. 是否把未来信息（如次日 open）泄露到特征里？

### 4.5 前端 session 卡死

- 查看 `outputs/sessions/<id>/session.json` 的 `status` 和 `current_progress`。
- 子进程 stdout/stderr 通常重定向到 `outputs/job_logs/`。
- 如果 session 状态为 `failed`，`steps/` 目录下的失败步骤 JSON 包含 traceback。

---

## 5. 代码质量

### 5.1 格式化

```bash
uv run black src/ tests/ pipeline/ webapp/ scripts/ --line-length 100
uv run ruff check src/ tests/ pipeline/ webapp/ scripts/ --fix
```

### 5.2 类型与文档

- 新函数加类型注解。
- 涉及防泄露、成本、撮合规则的地方必须写清楚注释。
- 复杂算法引用来源（如 GTJA191 的 DolphinDB 参考链接）。

### 5.3 提交前检查清单

```bash
uv run pytest tests/
uv run black --check src/ tests/ pipeline/ webapp/ scripts/
uv run ruff check src/ tests/ pipeline/ webapp/ scripts/
```

---

## 6. 扩展功能

### 6.1 添加新因子

见 CLAUDE.md §7.2。完成后应：

1. 在 `tests/` 加一个最小面板验证该因子可计算且不产生 NaN/inf。
2. 跑 `scripts/eval_factors.py` 看单因子 IC/ICIR。
3. 跑一次 walk-forward 看端到端影响。

### 6.2 添加新模型目标

见 CLAUDE.md §7.3。必须同时更新：

- `src/walkforward.py` 的 `_train_predict`
- `src/ensemble.py` 的 `predict`（如果是新 leg）
- `config.py` 的 `MODEL_OBJECTIVE` 注释
- 相关诊断脚本（如 `scripts/diagnostics/diag_linear_vs_gbm.py`）

### 6.3 添加新策略

在 `src/strategy.py` 实现新策略函数，加入 `ALL_STRATEGIES`。若可实盘，加入 `TRADEABLE`。A 股散户只能 long-only，任何 short/market-neutral 策略必须标注为 research-only。

### 6.4 添加前端页面

1. 在 `webapp/pages/` 创建新模块，暴露 `layout` 与可选 `register(app)`。
2. 在 `webapp/app.py` 的 `PAGES` 和 `NAV_ITEMS` 注册。
3. 如需新 session 模式，在 `pipeline/registry.py` 添加 `STEPS` + `PARAM_SCHEMA`。

---

## 7. 性能优化

### 7.1 每日重放回测很慢？

- 默认 `n_seeds=1` 已足够，增量集成内部多 seed 收益微弱。
- `REBUILD_DAYS` 保持 60，不要每天全量重建。
- 使用 `score_cache` 参数：一次训练，多次重放不同策略/择时参数。
- 限制回测区间（`start_date`, `end_date`）。

### 7.2 数据加载很慢？

- 首次构建缓存后，后续读取 parquet 很快。
- 实盘/次日任务用 `live_panel_start_date()` 计算最小所需窗口，避免拉全量 5 年。

---

## 8. 协作规范

- 不要提交 `config_credentials.toml`、模型文件、大数据缓存。
- `.gitignore` 应已包含 `data_cache/`, `models/`, `outputs/`, `.venv/`。
- 提交信息用中文或英文均可，但同一 PR 保持一致。
- 涉及回测数字的改动，PR 描述中应包含前后对比。

---

## 9. 参考命令速查

```bash
# 安装
uv venv && uv pip install -e ".[full,ui,dev]"

# 测试
uv run pytest tests/ -v

# 格式化
uv run black src/ tests/ pipeline/ webapp/ scripts/ --line-length 100
uv run ruff check src/ tests/ pipeline/ webapp/ scripts/ --fix

# 最小回测
uv run python tests/test_backtest_synthetic.py

# 全市场 WF
FORWARD_WINDOW=20 TRAIN_WINDOW=500 MODEL_OBJECTIVE=ret_weighted \
UNIVERSE_MODE=all uv run python scripts/run_walkforward.py

# 前端
uv run python run_webapp.py

# 次日订单
uv run python predict_tomorrow.py

# 清缓存
rm -rf data_cache/dataset_*.parquet
```
