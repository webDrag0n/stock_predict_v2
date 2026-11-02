# A股预测、策略与回测系统

[English README](README.md)

源自 Jane Street 笔记本（`jane-street-baseline-lgb-xgb-and-catboost.ipynb`）的建模思路，改造为面向**中国A股**、使用 **tushare** 真实数据的完整端到端框架：笔记本只训练模型并打分 weighted R²，本系统补齐了缺失的两块——**交易策略**与**防泄露的滚动 walk-forward 回测**——并串成一条以**最大化实际收益**为目标的可复现流水线。

```
tushare行情 (全市场 ~5700 只 ex-ST)  →  技术+基本面因子  →  预测 (远期收益)  →  策略 (仓位)  →  回测 (收益、风险)
```

## 功能特性

- **多模型增量集成**（`src/ensemble.py`）：gbm_l2 / gbm_ret_weighted / ic_linear 三腿，每日续训（LightGBM `init_model`）+ 滚动 OOS IC 定权（EMA 平滑）。
- **滚动 walk-forward 引擎**（`src/walkforward.py`）：embargo 隔离的样本外验证，与生产集成共用同一套 GBM 超参与截面归一化（唯一真相源 `walkforward.GBM_PARAMS`）。
- **交易策略**（`src/strategy.py`）：`long_topk`（A股默认、仅做多、散户唯一可实盘）、`tanh`、`cross_sectional`（市场中性，仅研究对照）。
- **回测引擎（唯一）**：每日重放+次日撮合（`src/backtest_daily.py` + `src/trading.py`）——周期调仓向量化引擎已删除。A股真实成本（佣金+印花税）、涨跌停、T+1、整手约束、参与率上限；walk-forward 的 OOS 预测以带指纹的 score_cache 喂入同一引擎逐笔撮合。
- **五层风控叠乘链**（`src/risk.py`）：回测与实盘共用唯一实现。
- **因子准入门槛**（`src/factor_eval.py`）：因子进模型前先过 IC / RankIC / ICIR / t 值 / 分层单调性检验；`factor_library/` 提供 YAML 公式因子库。
- **基本面因子**：`ep`/`bp`/`sp`/`ln_mv`（akshare 东财估值 `stock_value_em`，免费无限速，`merge_asof(backward)` point-in-time 对齐，杜绝前瞻）。全市场模式（~5700 只）同样可用：逐股缓存于 `data_cache/akshare_value/val_<code>.parquet`，首次回填全市场约 45~60 分钟（分批 + 限流 + 失败容忍，日志每 500 只报进度与 ETA），之后每日只增量重抓有缺口的股票；次新股等无数据股票自动跳过。`USE_AK_VALUE=0` 可单独关闭。财务因子（ROE/YoY 等，`USE_FINANCIAL_FACTORS`）走 tushare `daily_basic`/`fina_indicator`，限速过严，全市场模式下仍自动禁用。
- **Web 前端**（`run_webapp.py`，Dash）：会话式流水线 + 逐步骤报告——研究类会话（`walkforward` / `train` / `modeleval` / `factoreval`）在模型实验室「新建任务」面板发起，生产类 `nextday` 次日出单在交易台 A 卡「生成次日订单」发起；运行中任务显示进度条。
- **实盘工具**（`predict_tomorrow.py`）：次日可执行订单、模型门控（`models/staging/` 晋升）、自适应重训触发、回测 vs 实盘偏差跟踪。

## 快速开始

需要 [uv](https://github.com/astral-sh/uv) 与 Python ≥ 3.10。

```bash
# 1. 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖（自动创建 .venv；依赖声明在 pyproject.toml，由 uv 锁定）
uv sync --extra dev          # 核心 + pytest
# 可选 extra：--extra attn（注意力模型 leg，torch）
#             --extra full（XGBoost / CatBoost）
#             --extra ui（webapp）

# 3. 配置 tushare token（见下），然后运行
uv run python run_webapp.py                  # 前端会话（推荐）
uv run python run_webapp.py --debug          # 开发模式：热重载，改代码自动生效
                                             #（默认模式改代码后需重启进程）
uv run python scripts/run_walkforward.py     # 滚动 walk-forward 样本外回测
uv run python predict_tomorrow.py            # 生成次日可执行订单
uv run pytest                                # 测试套件
```

### 配置 tushare token

数据来自 [tushare pro](https://tushare.pro)（注册积分制）。把 token 填入 `config_credentials.toml`（已在 `.gitignore` 中，不会提交）：

```toml
[tushare]
token = "你的token"
```

或用环境变量 `TUSHARE_TOKEN`（优先级更高）。

默认 universe 为**全市场**（约 5700 只 ex-ST 股票，按日拉取）。首次运行按日下载并缓存到 `data_cache/tushare/`，之后秒级加载。

### 数据源配置（env 覆盖）

所有参数集中在 `config.py`，环境变量覆盖对所有入口生效：

```bash
START_DATE=2020-01-01 END_DATE=2024-12-31 FORWARD_WINDOW=5 \
  uv run python scripts/run_walkforward.py
```

### 预测目标与策略对齐（重要）

`responder_6` = 未来 `FORWARD_WINDOW` 日收益（裁剪到 ±0.5）。`MARKET_NEUTRAL_TARGET`（默认开启）与 `INDUSTRY_NEUTRAL`（默认开启）决定模型预测什么：

- **`INDUSTRY_NEUTRAL=1`（默认）**：预测**行业内超额收益**——远期收益减当日**同行业**均值（剔除 A 股强板块轮动成分）。当日有效样本数 < `INDUSTRY_MIN_COUNT`（默认 5）的行业回退到全市场均值，防小行业均值噪声。行业标签来自 tushare `stock_basic`（静态快照近似，月度缓存，见 `TushareConnector.get_industry_map` 注释）。`INDUSTRY_NEUTRAL=1` 隐含截面中性——即使 `MARKET_NEUTRAL_TARGET=0` 也照常去行业均值。
- **`MARKET_NEUTRAL_TARGET=1` 且 `INDUSTRY_NEUTRAL=0`**：预测**横截面超额收益**（远期收益减去当日全体均值），与 `long_topk` / `cross_sectional` 等排序类策略对齐——模型学"哪些股票跑赢/跑输大盘"。
- **两者均关闭**：预测**原始远期收益**（含市场 beta），适合 `tanh` 方向性策略。

### 北向持股因子

`USE_NORTHBOUND`（默认开启）并入 tushare `hk_hold` 的陆股通个股持股（北向 = 沪股通 `exchange='SH'` + 深股通 `exchange='SZ'` 两路，按日分路缓存 `data_cache/tushare/hk_hold_{SH,SZ}_YYYYMMDD.parquet`；⚠️ `hk_hold` 不传 exchange 返回的是**南向**港股数据，已实测确认，必须显式指定），派生 `north_ratio` / `north_ratio_chg5` / `north_vol_chg5` / `north_covered` 四个股票级因子。**防泄露**：T 日持股数据在 T 日晚间/次晨才披露，特征按 symbol 滞后一日（shift(1)）后使用，休市错开日前向填充上限 `NORTHBOUND_MAX_FILL_DAYS`（默认 10）。非陆股通标的填 0 且 `north_covered=0`。注意当前 token 档 `hk_hold` 双重限速（1 次/小时 + 5 次/天，已实测）：每日增量（每天 SH+SZ 共 2 次调用，跨两次运行自动补齐）可行，历史逐日回填会熔断（见 `fetch_hk_hold` 注释）。

## 常用命令

```bash
# 滚动 walk-forward 样本外回测（拼接 OOS 净值 + 各窗口指标分布）
uv run python scripts/run_walkforward.py

# 每窗口在训练段内做因子准入筛选（隔离，无泄露）并对比
uv run python scripts/run_walkforward.py --compare

# 推荐配置：300 只市值筛选横截面 + 20 日窗口（信号更稳健）
USE_SCREENING=1 UNIVERSE_SIZE=300 FORWARD_WINDOW=20 \
  uv run python scripts/run_walkforward.py --compare

# 生成次日订单 / 确认成交后写回持仓
uv run python predict_tomorrow.py --capital 1000000
uv run python predict_tomorrow.py --commit

# 每日例行封装（推荐）：交易日门控 + 数据更新 + 出单，小本金参数已内置
./daily_run.sh           # 每个交易日收盘后运行（默认按 3 万本金价差约束配置）
./daily_run.sh commit    # 确认成交后写回持仓
# cron 示例：10 18 * * 1-5  cd <项目目录> && ./daily_run.sh

# 模型门控：新配方模型先入 models/staging/ 双回放对比，Sharpe 更高且回撤不劣才晋升
uv run python predict_tomorrow.py --no-gate          # 跳过门控（旧行为）
uv run python predict_tomorrow.py --promote <id>     # 人工晋升 staging 候选

# 自适应重训与偏差跟踪：默认指标触发才重训（IC 退化/收益落后/超龄）
uv run python predict_tomorrow.py --force-retrain    # 强制今晚重训
uv run python predict_tomorrow.py --tracking-report  # 回测 vs 实盘偏差报告

# 单因子评估 / 多 seed 稳健性复核（前端 factoreval 会话是 eval_factors 的等价入口）
uv run python scripts/eval_factors.py
uv run python scripts/research/multiseed_robust.py
```

运行后生成 `outputs/`：`orders_*.csv`（次日订单）、`holdings.csv` / `account.csv`（实盘持仓/现金）。walk-forward 产物统一为 canonical 目录 `outputs/walkforward/`（CLI 与前端会话共用契约）：`oos_curve.parquet`（连续 OOS 净值序列）、`window_metrics.parquet`（各窗口 OOS 指标）、`oos_bench.parquet`（等权基准日收益）、`meta.json`（汇总指标 + 配置快照 + 生成时间）；前端模型实验室页直接读取该目录展示「最近一次 CLI Walk-Forward 结果」。所有仪表盘图表已迁移到前端会话，不再输出 PNG。

## 防泄露设计（本框架的核心约定）

单次划分回测极易因泄露与窗口运气产生虚高数字。本框架修复了三个前瞻泄露源：

- **embargo**：训练段与样本外段之间剔除 `EMBARGO_DAYS`（默认 = `FORWARD_WINDOW`）天，杜绝标签跨界重叠（config import 期硬断言 + 引擎层防御性断言）。
- **因子选择隔离**：每个滚动窗口只用其训练段选因子（`select_passing_factors`），绝不碰测试段。
- **窗口内 early-stopping 验证集 embargo**：GBM 早停用的训练段尾部验证集前再剔除 embargo 天，防止验证标签偷看训练尾部。

**防数据窥探约定**：策略与超参固定在 `config.py`，所有结论只看滚动 walk-forward 的拼接样本外曲线；调参实验只允许对照历史 OOS 做研究性参考，不进生产默认值而不标注。**重点看「为正窗口占比」**——衡量信号能否跨时间稳健复现，比单一 Sharpe 可信得多。

> **实测结论（记住这两行）：**
> - 49 只蓝筹：所有配置均为负 OOS Sharpe，3~4/11 窗口为正 ≈ 随机——信号不足以战胜成本。
> - **300 只市值筛选 + 20 日窗口**：OOS Sharpe **+2.36**、8/11 窗口为正、Calmar +1.04——跨 2.6 年防泄露后稳健，是可信的真实水平。根因是横截面太窄：49 只高相关蓝筹撑不起稳定排序；扩到 300 只后价值因子（bp/sp/ep）信号站住了。
> - **反直觉**：宽横截面下全特征(+2.36) > 窗口内硬门槛选因子(+1.32)。因子准入门槛适合诊断/理解信号，不适合当硬性特征裁剪器——让 GBM 自己加权全部因子更稳健。

## 仓库结构

```
config.py                  # 所有参数 + 凭证读取（env 可覆盖）
config_credentials.toml    # tushare token（不提交）
predict_tomorrow.py        # CLI：生成次日订单
run_session.py             # 会话执行器（由 webapp 子进程调用）
run_webapp.py              # Dash 前端入口

src/                       # 核心库（顶层模块，src/ 加入 sys.path 后 import）
  data/                    # 数据连接器（tushare/akshare）、因子工程、面板加载
    factors/               # 公式因子引擎（parser/registry/evaluator）
  walkforward.py           # 滚动 walk-forward 引擎 + GBM_PARAMS
  ensemble.py              # 多模型增量集成（每日续训 + 滚动 IC 定权）
  strategy.py              # 预测 → 持仓
  trading.py               # 每日决策 + 成交模拟（回测/实盘共用）
  risk.py                  # 五层风控叠乘链（唯一实现）
  backtest_daily.py        # 每日重放回测引擎（唯一回测口径）
  factor_eval.py           # 单因子评估
  measure_alpha.py         # Newey-West alpha 显著性检验
  live.py / live_state.py / live_tracking.py   # 实盘推理、状态持久化、偏差跟踪
  model_store.py / model_gate.py               # 模型工件存取 + 晋升门控

pipeline/                  # 会话式流水线框架（registry/context/steps）
webapp/                    # Dash 多页面前端
scripts/                   # 研究脚本（活/归档清单见 scripts/README.md）
factor_library/            # YAML 公式因子库
tests/                     # pytest 测试套件
docs/system_architecture.html  # 架构图（浏览器打开）
```

新增数据源只需在 `src/data/` 下实现 `DataConnector` 的 3 个方法，并在 `__init__.py` 的 `_CONNECTORS` 注册。

## 文档索引

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | English README |
| [DEVELOPMENT.md](DEVELOPMENT.md) | 开发者指南：测试、调试、约定、扩展点 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南 |
| [CLAUDE.md](CLAUDE.md) | 完整研究结论与方法论铁律（AI 助手上下文，亦可人读） |
| [factor_library/README.md](factor_library/README.md) | 公式因子语言 |
| [scripts/README.md](scripts/README.md) | 活脚本 vs 归档脚本清单 |
| [docs/system_architecture.html](docs/system_architecture.html) | 架构图 |

## 如何继续优化

按收益贡献从高到低：

1. **更强的特征（最高优先级）**：裸技术指标对A股远期收益的预测力很弱。基本面（PE/PB/ROE）与北向持股（`USE_NORTHBOUND`，个股级陆股通持股）已接入；可继续补充主力净流入等资金流、行业/概念哑变量。
2. **行业中性**：A股板块轮动强。行业中性目标已实现（`INDUSTRY_NEUTRAL=1`，标签减当日同行业均值，小行业回退全市场均值）；可进一步在选股时按行业分组约束。
3. **预测窗口与调仓频率**：`FORWARD_WINDOW` 增大（如10/20日）通常信噪比更高、换手更低；同时降低交易成本拖累。
4. **标签处理**：对 `responder_6` 做 winsorize / 分位数归一，减少极端值对回归的干扰。
5. **模型**：装上 XGBoost/CatBoost 自动进入集成；或加入 rank-based 目标（LambdaRank），直接优化排序而非回归。
6. **风险约束**：单票权重上限、涨跌停日不可调仓（已有 `limit_flag` 特征）。

## 注意事项

- 本系统仅供研究和回测，**不构成投资建议**；历史回测不保证未来收益。
- universe 已修复存活偏差，但 tushare `index_weight` 历史深度有限，极早期可能回退到当前成分。
- A股 **T+1** 已在撮合层强制（当日买入次日才能卖），见 `src/trading.py`。
- A股成本模型：买入佣金 `COMMISSION_RATE`（0.02%）；卖出佣金 + 印花税 `STAMP_DUTY_RATE`（0.1%），按每只股票逐日仓位变化方向计费。
- 原 Kaggle 笔记本保持不变，本系统独立运行。

## 许可证

[MIT](LICENSE)——详见 LICENSE 文件。发布前请将其中的版权人占位替换为你的名字。
