# A股预测、策略与回测系统

源自 Jane Street 笔记本（`jane-street-baseline-lgb-xgb-and-catboost.ipynb`）的建模思路，改造为面向**中国A股**、使用 **tushare** 数据的完整端到端系统，目标是**最大化实际收益**。

```
tushare行情 (沪深300成分股)  →  技术+A股因子  →  预测 (responder_6)  →  策略 (仓位)  →  回测 (收益、风险)
```

## 环境配置（uv）

本项目使用 [uv](https://github.com/astral-sh/uv) 管理 Python 环境。

```bash
# 1. 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 在项目目录创建虚拟环境并安装依赖
cd ~/stock_predict_v2
uv venv                          # 创建 .venv
uv pip install -r pyproject.toml # 安装依赖

# 3. 配置 tushare token（见下）后运行
.venv/bin/python run_pipeline.py
```

可选的额外集成模型（XGBoost / CatBoost）：`uv pip install -e ".[full]"`

## 配置 tushare token

数据来自 [tushare pro](https://tushare.pro)（注册积分制）。把 token 填入 `config_credentials.toml`（已在 `.gitignore` 中，不会提交）：

```toml
[tushare]
token = "你的token"
```

或用环境变量 `TUSHARE_TOKEN`（优先级更高）。

## 快速开始

```bash
.venv/bin/python run_pipeline.py            # 自动选策略：验证集选，测试集报
.venv/bin/python run_pipeline.py --compare  # 额外在测试集展示所有策略（仅研究对比）
.venv/bin/python run_pipeline.py --strategy long_topk  # 指定策略，跳过自动选择
```

默认 universe 为**沪深300成分股**（约5年历史）。首次运行按股票下载并缓存到 `data_cache/tushare/`，之后秒级加载。

### 数据源配置

```bash
# 切换指数 universe（如中证500）
DEFAULT_INDEX=000905.SH .venv/bin/python run_pipeline.py

# 自定义日期范围和预测窗口
START_DATE=2020-01-01 END_DATE=2024-12-31 FORWARD_WINDOW=5 \
  .venv/bin/python run_pipeline.py
```

成分股 universe 默认按**回测期间各月成分股的并集**取得，**已修复存活偏差**（被调出指数的股票仍纳入）。

### 预测目标与策略对齐（重要）

`MARKET_NEUTRAL_TARGET`（默认开启）决定模型预测什么：

- **开启**（默认）：预测**横截面超额收益**（远期收益减去当日全体均值）。与 `long_topk` / `cross_sectional` 等排序类策略对齐——模型学"哪些股票跑赢/跑输大盘"。
- **关闭**：预测**原始远期收益**（含市场beta）。适合 `tanh` 方向性策略。

## 核心功能

### 1. 预测模型
- **集成模型**：LightGBM（必备）+ XGBoost + CatBoost（装了就自动用）
- **时间切分交叉验证**：按日期分折，避免日内泄露
- **评估指标**：加权R²

### 2. 交易策略
- **long_topk**（A股默认，可实盘）：每个交易日做多预测排名前 `TOPK_RATIO`（默认20%）的股票，等权，仓位和归一为1。**仅做多**，符合A股散户约束。
- **tanh**：平滑方向性仓位 = tanh(scale × prediction)（含做空，仅研究/期货）
- **cross_sectional**（**研究对照，非实盘**）：市场中性多空。A股散户无法做空个股，仅用于诊断信号质量。

### 3. 策略选择方法（避免数据窥探）
三段时序切分：

```
日期轴: ──── 训练 ──── | ── 验证 ── | ── 测试 ──
        拟合模型        选策略        唯一汇报点
```

- 在**验证集**比较各策略，选 Sharpe 最高者
- 仅对选中策略在**测试集**做最终样本外汇报
- **在测试集上挑最优=数据窥探偏差**，会系统性高估表现，本系统刻意避免

### 4. 回测引擎
- **严格时序**：测试集模型从未见过
- **A股真实成本**：买入佣金 0.02%；卖出佣金 0.02% + 印花税 0.1%
- **性能指标**：总收益、夏普、索提诺、卡玛、最大回撤、胜率、加权R²
- **数据集质量检查**：标签统计、横截面覆盖、前瞻偏差嗅探

### 5. 单因子评估框架（因子准入门槛）

因子进模型前先过单因子检验——这是持续因子优化的"度量衡"。

```bash
.venv/bin/python eval_factors.py            # 评估全部命名因子
.venv/bin/python eval_factors.py --plot     # 额外输出分层累计净值曲线
.venv/bin/python eval_factors.py --layers 10
```

- 指标：IC、RankIC、ICIR（含年化）、IC 的 t 统计量、分层收益单调性、Top-Bottom 多空、换手
- 准入门槛（已按小横截面调校，主用统计显著性）：`|IC_t|≥2` 且 `|RankICIR|≥0.2` 且 `|RankIC|≥0.02` 且 `单调性≥0.5`
- 输出 `outputs/factor_eval.csv`（按 |IC_t| 排序）和 `outputs/factor_layers.png`
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
.venv/bin/python run_walkforward.py --compare

# 300只市值筛选横截面（推荐，信号更稳健）—— 须先确认 stock_basic / pro_bar 可用
USE_SCREENING=1 UNIVERSE_SIZE=300 FORWARD_WINDOW=20 .venv/bin/python run_walkforward.py --compare
```

修复了三个前瞻泄露源：
- **embargo**：训练段与样本外段之间剔除 `EMBARGO_DAYS`(默认=FORWARD_WINDOW)天，杜绝标签跨界重叠
- **因子选择隔离**：每个滚动窗口只用其训练段选因子（`select_passing_factors`），绝不碰测试段
- **连续块 purged CV**：折划分从 `date%N_FOLD` 交错改为连续块 + 块间 embargo

输出 `outputs/walkforward_*.png`（连续 OOS 净值）和 `walkforward_windows_*.csv`（各窗口指标分布）。**重点看「为正窗口占比」**——衡量信号能否跨时间稳健复现，比单一 Sharpe 可信得多。

> **实测结论（记住这两行）：**
> - 49 只蓝筹：所有配置均为负 OOS Sharpe，3~4/11 窗口为正 ≈ 随机——信号不足以战胜成本。
> - **300 只市值筛选 + 20 日窗口**：OOS Sharpe **+2.36**、8/11 窗口为正、Calmar +1.04——跨 2.6 年防泄露后稳健，是可信的真实水平。根因就是横截面太窄：49 只高相关蓝筹撑不起稳定排序；扩到 300 只后价值因子（bp/sp/ep）信号站住了。
> - **反直觉**：宽横截面下全特征(+2.36) > 窗口内硬门槛选因子(+1.32)。因子准入门槛适合诊断/理解信号，不适合当硬性特征裁剪器——让 GBM 自己加权全部因子更稳健。


## 输出结果

运行后生成 `outputs/` 目录：

- **equity_<strategy>.png** — 权益曲线 + 回撤图
- **factor_eval.csv / factor_layers.png** — 单因子评估表与分层净值
- **predictions.csv** — 每行的预测值、仓位、实际PnL
- **report.json** — 数据来源、参数、所有策略的完整指标

## 主要文件

```
config.py                  # 所有参数 + 凭证读取
config_credentials.toml    # tushare token（不提交）
src/data/
  base.py                  # DataConnector 抽象基类（扩展新数据源的接口）
  tushare.py               # TushareConnector：成分股 + qfq日线 + daily_basic
  __init__.py              # 特征工程 + 目标 + Jane Street格式 + load_data/split_dates/质量检查
src/features.py            # 特征矩阵提取、NaN填充
src/model.py               # 集成模型（按日期 purged CV）
src/strategy.py            # 预测转仓位（long_topk / tanh / cross_sectional）
src/backtest.py            # 回测引擎 + A股成本 + 指标
run_pipeline.py            # CLI 主程序（三段切分 + 自动选策略）
```

新增数据源只需在 `src/data/` 下实现 `DataConnector` 的 3 个方法，并在 `__init__.py` 的 `_CONNECTORS` 注册。

## 常用命令

```bash
# 自动选策略（验证集选，测试集报）
.venv/bin/python run_pipeline.py

# 指定策略
.venv/bin/python run_pipeline.py --strategy long_topk

# 自动选择时只考虑A股可交易（long-only）策略
.venv/bin/python run_pipeline.py --tradeable-only

# 额外在测试集对比所有策略（仅研究）
.venv/bin/python run_pipeline.py --compare

# 复用已训练模型
.venv/bin/python run_pipeline.py --no-train

# 切换 universe / 预测窗口
DEFAULT_INDEX=000905.SH FORWARD_WINDOW=10 .venv/bin/python run_pipeline.py
```

## 输出结果

运行后生成 `outputs/`：
- **equity_<strategy>.png** — 选中策略的净值曲线 + 回撤
- **predictions.csv** — 测试集每行的预测、仓位、净PnL
- **report.json** — 数据源/参数/成本、验证集选择过程、测试集最终指标

## 关于"是否只用一个策略"

**结论：最终只用一个策略，但必须在验证集选、测试集报。**

在测试集上挑表现最好的策略是数据窥探偏差，会系统性高估真实表现。本系统用三段切分严格区分"选策略"和"报表现"。对 A 股散户而言更简单：`long_topk` 是唯一实际可交易的策略，可直接 `--strategy long_topk` 指定，从根本上无需在测试集挑选。`cross_sectional` 中性多空保留为研究对照（不可实盘）。

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
- A股 **T+1**（当日买入次日才能卖）尚未在回测中强制，当前以日频持仓近似；做日内或高频前需补充该约束。
- 原 Kaggle 笔记本保持不变，本系统独立运行。

## 技术细节

### 预测目标
`responder_6` = 未来 `FORWARD_WINDOW` 日收益（裁剪到 ±0.5）。`MARKET_NEUTRAL_TARGET=1` 时再减去当日全体均值，变为横截面超额收益。

### 时间切分
模型训练内部按 `date_index % N_FOLD != fold` 分折，确保同一日期所有行同折，避免日内泄露。整体三段（训练/验证/测试）按时序前后切分。

### long_topk 策略
每个交易日：按预测值取横截面前 `TOPK_RATIO`，等权买入，权重和归一为 `GROSS_LEVERAGE`。仓位恒非负（long-only）。

### A股成本模型
买入：佣金 `COMMISSION_RATE`（0.02%）。卖出：佣金 + 印花税 `STAMP_DUTY_RATE`（0.1%）。按每只股票逐日的仓位变化方向计费。
