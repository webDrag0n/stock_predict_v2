# A股预测、策略与回测系统

源自 Jane Street 笔记本（`jane-street-baseline-lgb-xgb-and-catboost.ipynb`）的建模思路，改造为面向**中国A股**、使用 **tushare** 数据的完整端到端系统，目标是**最大化实际收益**。

```
tushare行情 (全市场 ~5700 只 ex-ST)  →  技术+A股因子  →  预测 (responder_6)  →  策略 (仓位)  →  回测 (收益、风险)
```

## 环境配置（uv）

本项目使用 [uv](https://github.com/astral-sh/uv) 管理 Python 环境。

```bash
# 1. 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 在项目目录创建虚拟环境并安装依赖（依赖声明在 pyproject.toml，由 uv 管理）
cd ~/stock_predict_v2
uv sync --extra dev          # 创建 .venv 并安装核心依赖 + pytest

# 3. 配置 tushare token（见下）后运行
.venv/bin/python run_webapp.py                 # 前端会话（推荐）
# 或命令行：
.venv/bin/python scripts/run_walkforward.py    # 滚动 walk-forward 样本外回测
.venv/bin/python predict_tomorrow.py           # 生成次日可执行订单
```

> 阶段 5 架构收敛已删除旧端到端入口 `scripts/run_pipeline.py` 与其模型实现
> `src/model.py`（无 embargo 切分的驻留式 CV，存在泄露与虚高）；当前入口就是
> 上面三个（详见 CLAUDE.md 第 5 节）。

可选 extra：`--extra attn`（注意力模型 leg，torch）、`--extra full`（XGBoost / CatBoost）、
`--extra ui`（webapp）。

## 配置 tushare token

数据来自 [tushare pro](https://tushare.pro)（注册积分制）。把 token 填入 `config_credentials.toml`（已在 `.gitignore` 中，不会提交）：

```toml
[tushare]
token = "你的token"
```

或用环境变量 `TUSHARE_TOKEN`（优先级更高）。

## 快速开始

```bash
.venv/bin/python run_webapp.py                      # 前端：创建 backtest / nextday 会话（推荐）
.venv/bin/python scripts/run_walkforward.py         # 滚动窗口 OOS 回测 + 仪表盘
.venv/bin/python predict_tomorrow.py --capital 1000000   # 生成次日订单
```

默认 universe 为**全市场**（约 5700 只 ex-ST 股票，按日拉取）。首次运行按日下载并缓存到 `data_cache/tushare/`，之后秒级加载。

### 数据源配置

```bash
# 自定义日期范围和预测窗口（env 覆盖对所有入口生效）
START_DATE=2020-01-01 END_DATE=2024-12-31 FORWARD_WINDOW=5 \
  .venv/bin/python scripts/run_walkforward.py
```

### 预测目标与策略对齐（重要）

`MARKET_NEUTRAL_TARGET`（默认开启）决定模型预测什么：

- **开启**（默认）：预测**横截面超额收益**（远期收益减去当日全体均值）。与 `long_topk` / `cross_sectional` 等排序类策略对齐——模型学"哪些股票跑赢/跑输大盘"。
- **关闭**：预测**原始远期收益**（含市场beta）。适合 `tanh` 方向性策略。

## 核心功能

### 1. 预测模型
- **多模型增量集成**（`src/ensemble.py`）：gbm_l2 / gbm_ret_weighted / ic_linear 三腿，
  每日续训（LightGBM init_model）+ 滚动 OOS IC 定权（EMA 平滑）
- **滚动 walk-forward 引擎**（`src/walkforward.py`）：embargo 隔离的样本外验证，
  与集成共用同一套 GBM 超参与截面归一化

### 2. 交易策略
- **long_topk**（A股默认，可实盘）：每个交易日做多预测排名前 `TOPK_RATIO`（默认20%）的股票，等权，仓位和归一为1。**仅做多**，符合A股散户约束。
- **tanh**：平滑方向性仓位 = tanh(scale × prediction)（含做空，仅研究/期货）
- **cross_sectional**（**研究对照，非实盘**）：市场中性多空。A股散户无法做空个股，仅用于诊断信号质量。

### 3. 策略选择方法（避免数据窥探）

旧的"训练/验证/测试三段切分 + 验证集选策略"管线（run_pipeline）已删除。
当前的防窥探约定：**策略与超参固定在 config.py，所有结论只看滚动
walk-forward 的拼接样本外曲线**（`scripts/run_walkforward.py`），调参实验
只允许对照历史 OOS 做研究性参考，不进生产默认值而不标注。

### 4. 回测引擎
- **严格时序**：测试集模型从未见过
- **A股真实成本**：买入佣金 0.02%；卖出佣金 0.02% + 印花税 0.1%
- **性能指标**：总收益、夏普、索提诺、卡玛、最大回撤、胜率、加权R²
- **数据集质量检查**：标签统计、横截面覆盖、前瞻偏差嗅探

### 5. 单因子评估框架（因子准入门槛）

因子进模型前先过单因子检验——这是持续因子优化的"度量衡"。

```bash
.venv/bin/python scripts/eval_factors.py            # 评估全部命名因子
.venv/bin/python scripts/eval_factors.py --layers 10
```

- 指标：IC、RankIC、ICIR（含年化）、IC 的 t 统计量、分层收益单调性、Top-Bottom 多空、换手
- 准入门槛（已按小横截面调校，主用统计显著性）：`|IC_t|≥2` 且 `|RankICIR|≥0.2` 且 `|RankIC|≥0.02` 且 `单调性≥0.5`
- 输出 `outputs/factor_eval.csv`（按 |IC_t| 排序）；因子分层累计净值图已迁移到前端回测会话的因子分析步骤
- 核心实现 `src/factor_eval.py`；命名因子面板由 `data.load_factor_panel()` 提供（保留因子原名，区别于喂模型的 `feature_NN`）

### 6. 基本面因子（估值类）

引入估值/规模因子，与量价因子相关性低、提升组合 ICIR。

- 因子：`ep`(1/PE)、`bp`(1/PB)、`sp`(1/PS)、`ln_mv`(对数市值) 及其截面分位
- 数据源：**akshare `stock_value_em`**（东财个股每日估值，免费无限速、日频原生）。tushare 的 `daily_basic` 限速过严、`fina_indicator`(ROE/增速)需高积分，故估值因子走 akshare。
- **point-in-time 对齐**：`merge_asof(backward)` 并入日频面板，杜绝前瞻
- 开关 `USE_FUNDAMENTALS=1`（默认开）；逐股票缓存于 `data_cache/akshare_value/`

### 7. 滚动走动回测（防泄露样本外验证 · 最重要）

单次划分回测极易因泄露与窗口运气产生虚高数字。本框架用滚动 walk-forward 给出可信结论。

```bash
# 默认（精选50只蓝筹，快）
.venv/bin/python scripts/run_walkforward.py --compare

# 300只市值筛选横截面（推荐，信号更稳健）—— 须先确认 stock_basic / pro_bar 可用
USE_SCREENING=1 UNIVERSE_SIZE=300 FORWARD_WINDOW=20 .venv/bin/python scripts/run_walkforward.py --compare
```

修复了三个前瞻泄露源：
- **embargo**：训练段与样本外段之间剔除 `EMBARGO_DAYS`(默认=FORWARD_WINDOW)天，杜绝标签跨界重叠（config import 期硬断言 + 引擎层防御性断言）
- **因子选择隔离**：每个滚动窗口只用其训练段选因子（`select_passing_factors`），绝不碰测试段
- **窗口内 early-stopping 验证集 embargo**：GBM 早停用的训练段尾部验证集前再剔除 embargo 天，防止验证标签偷看训练尾部

输出 `outputs/walkforward_windows_*.csv`（各窗口指标分布）和 `walkforward_oos_curve.parquet`（连续 OOS 净值序列）；多面板仪表盘已迁移到前端「滚动Walk-Forward」会话（净值 vs 基准 / 累计 alpha / 逐窗口 Sharpe / 回撤）。**重点看「为正窗口占比」**——衡量信号能否跨时间稳健复现，比单一 Sharpe 可信得多。

> **实测结论（记住这两行）：**
> - 49 只蓝筹：所有配置均为负 OOS Sharpe，3~4/11 窗口为正 ≈ 随机——信号不足以战胜成本。
> - **300 只市值筛选 + 20 日窗口**：OOS Sharpe **+2.36**、8/11 窗口为正、Calmar +1.04——跨 2.6 年防泄露后稳健，是可信的真实水平。根因就是横截面太窄：49 只高相关蓝筹撑不起稳定排序；扩到 300 只后价值因子（bp/sp/ep）信号站住了。
> - **反直觉**：宽横截面下全特征(+2.36) > 窗口内硬门槛选因子(+1.32)。因子准入门槛适合诊断/理解信号，不适合当硬性特征裁剪器——让 GBM 自己加权全部因子更稳健。


## 输出结果

运行后生成 `outputs/` 目录：

- **因子分层净值 / walk-forward 多面板 / 多 seed 稳健性仪表盘** — 已迁移到前端（回测会话因子分析步骤、「滚动Walk-Forward」「多Seed稳健性」会话）
- **factor_eval.csv** — 单因子评估表
- **orders_*.csv** — 次日可执行订单（predict_tomorrow.py）

## 主要文件

```
config.py                  # 所有参数 + 凭证读取
config_credentials.toml    # tushare token（不提交）
run_webapp.py              # 启动前端（推荐入口）
run_session.py             # 会话执行器
predict_tomorrow.py        # 生成次日订单

src/
  data/                    # 数据连接器与因子工程
  ensemble.py              # 多模型增量集成（每日续训 + 滚动 IC 定权）
  walkforward.py           # 滚动回测引擎
  strategy.py              # 预测 → 持仓
  trading.py               # 每日决策 + 成交模拟
  risk.py                  # 五层风控叠乘链（回测/实盘共用唯一实现）
  backtest.py              # 周期调仓回测引擎（run_backtest_periodic）
  backtest_daily.py        # 每日重放回测引擎
  factor_eval.py           # 单因子评估
  measure_alpha.py         # Alpha 显著性检验

scripts/                   # 研究脚本（活/归档清单见 scripts/README.md）
  eval_factors.py          # 单因子评估 CLI
  run_walkforward.py       # walk-forward CLI
  diagnose_factors.py      # 因子诊断
  diagnostics/             # 诊断脚本
  optimize/                # 参数优化（optimize_daily_params.py）
  research/                # 研究复现（multiseed_robust.py）
  archive/                 # 一次性实验归档（不再维护）

pipeline/                  # 会话式流水线
webapp/                    # Dash 前端
tests/                     # 单元/集成测试
```

新增数据源只需在 `src/data/` 下实现 `DataConnector` 的 3 个方法，并在 `__init__.py` 的 `_CONNECTORS` 注册。

## 常用命令

```bash
# 滚动 walk-forward 样本外回测（拼接 OOS 净值 + 各窗口指标分布）
.venv/bin/python scripts/run_walkforward.py

# 每窗口在训练段内做因子准入筛选（隔离，无泄露）并对比
.venv/bin/python scripts/run_walkforward.py --compare

# 生成次日订单 / 确认成交后写回持仓
.venv/bin/python predict_tomorrow.py --capital 1000000
.venv/bin/python predict_tomorrow.py --commit

# 模型门控：新配方模型先入 models/staging/ 双回放对比，Sharpe 更高且回撤不劣才晋升
.venv/bin/python predict_tomorrow.py --no-gate          # 跳过门控（旧行为）
.venv/bin/python predict_tomorrow.py --promote <id>     # 人工晋升 staging 候选

# 自适应重训与偏差跟踪：默认指标触发才重训（IC 退化/收益落后/超龄）
.venv/bin/python predict_tomorrow.py --force-retrain    # 强制今晚重训
.venv/bin/python predict_tomorrow.py --tracking-report  # 回测 vs 实盘偏差报告

# 单因子评估 / 多 seed 稳健性复核
.venv/bin/python scripts/eval_factors.py
.venv/bin/python scripts/research/multiseed_robust.py

# 切换预测窗口等参数（env 覆盖对所有入口生效）
FORWARD_WINDOW=10 .venv/bin/python scripts/run_walkforward.py
```

## 输出结果

运行后生成 `outputs/`：
- **walkforward_windows_\*.csv** — 各滚动窗口的 OOS 指标
- **walkforward_oos_curve.parquet** — 连续 OOS 净值序列（前端「滚动Walk-Forward」会话的数据源）
- **净值/多面板/多 seed 仪表盘图** — 已迁移到前端会话，不再输出 PNG
- **orders_\*.csv / holdings.csv / account.csv** — 次日订单与实盘持仓/现金状态

## 关于"是否只用一个策略"

**结论：最终只用一个策略。** 对 A 股散户而言很简单：`long_topk` 是唯一实际
可交易的策略（仅做多），`cross_sectional` 中性多空保留为研究对照（不可实盘）。
策略与超参固定在 `config.py`，结论一律看滚动 walk-forward 的拼接样本外曲线，
避免"在测试集上挑最优"的数据窥探偏差。

## 如何继续优化

按收益贡献从高到低：

1. **更强的特征（最高优先级）**：裸技术指标对A股远期收益的预测力很弱。优先加入基本面（PE/PB/ROE，tushare `daily_basic`/`fina_indicator`）、资金流（北向、主力净流入）、行业/概念哑变量做行业中性化。
2. **行业中性**：A股板块轮动强。把目标改为"行业内超额收益"、或在选股时按行业分组，能显著降低风险。
3. **预测窗口与调仓频率**：`FORWARD_WINDOW` 增大（如10/20日）通常信噪比更高、换手更低；同时降低交易成本拖累。
4. **标签处理**：对 `responder_6` 做 winsorize / 分位数归一，减少极端值对回归的干扰。
5. **模型**：装上 XGBoost/CatBoost 自动进入集成；或加入 rank-based 目标（LambdaRank），直接优化排序而非回归。
6. **风险约束**：单票权重上限、涨跌停日不可调仓（已有 `limit_flag` 特征，可在回测里强制 T 日涨停不买、跌停不卖）。

## 注意事项

- 本系统仅供研究和回测，不构成投资建议；历史回测不保证未来收益。
- universe 已修复存活偏差，但 tushare `index_weight` 历史深度有限，极早期可能回退到当前成分。
- A股 **T+1** 已在撮合层强制（当日买入次日才能卖），见 `src/trading.py`。
- 原 Kaggle 笔记本保持不变，本系统独立运行。

## 技术细节

### 预测目标
`responder_6` = 未来 `FORWARD_WINDOW` 日收益（裁剪到 ±0.5）。`MARKET_NEUTRAL_TARGET=1` 时再减去当日全体均值，变为横截面超额收益。

### 时间切分
滚动 walk-forward：每步用前 `TRAIN_WINDOW` 天训练、剔除 `EMBARGO_DAYS` 天 embargo、
预测其后 `TEST_WINDOW` 天（纯样本外），各窗口 OOS 拼接成连续净值曲线。

### long_topk 策略
每个交易日：按预测值取横截面前 `TOPK_RATIO`，等权买入，权重和归一为 `GROSS_LEVERAGE`。仓位恒非负（long-only）。

### A股成本模型
买入：佣金 `COMMISSION_RATE`（0.02%）。卖出：佣金 + 印花税 `STAMP_DUTY_RATE`（0.1%）。按每只股票逐日的仓位变化方向计费。
