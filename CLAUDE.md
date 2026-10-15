# stock-predict-v2

A-share 量化研究系统：数据 → 因子 → 模型 → 策略 → 回测/实盘，端到端可复现。

本项目从 Jane Street Kaggle notebook 的预测任务出发，补全了缺失的**策略**与**回测**环节，并针对中国 A 股市场做了大量改造。系统围绕一个核心目标：**在严格防泄露的前提下，验证并最大化样本外的已实现收益**。

---

## 1. 项目定位与研究结论（必读）

这不是一个“开箱即用的赚钱机器”。经过多轮端到端验证，当前系统的主结论是：

- **稳健 alpha 极其微弱**：在防泄露、 realistic cost、全市场 (~5700 只) 条件下，ret_weighted 目标 + 全量因子库的长期样本外 alpha 约 **+13.8%/年，NW-t ≈ 2.80**（3 个独立 seed 稳健显著）。但这是经过大量试错后才收敛到的结果。
- **常见捷径都会虚高**：小股票池 (49/300 只)、未复权价格、每日重叠调仓、偷看测试集选策略等都会让 Sharpe/alpha 虚高数倍，必须逐项修正。
- **信号站得住，但容量有限**：alpha 主要来源于小盘错误定价，容量天花板约 **¥1500 万**；流动性倾斜会摧毁 alpha。
- **默认引擎是每日重放回测**：`src/backtest_daily.py` 逐日重放每日决策 + 次日撮合，与实盘 `predict_tomorrow.py` / `pipeline` 次日交易共用同一套 `src/trading.py` 决策代码，保证回测=实盘同分布。（旧的事件驱动引擎 `src/backtest_event.py` 含一日前视，已随阶段 5 删除；**基于它得出的历史结论一律作废**。）

做任何改动前，先确认跑的是**目标模型**且**缓存没被覆盖**。这是项目最重要的方法论铁律。

---

## 2. 高层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  数据层 (src/data)                                                   │
│  tushare / akshare → OHLCV → 因子工程(technical/orthogonal/α158/GTJA191) │
│  → 基本面/财务（月度快照，merge_asof 防泄露）→ 期权 IV 宏观特征        │
│  → Jane Street 行格式 或 命名因子面板                                 │
├──────────────────────────────────────────────────────────────────────┤
│  模型层 (src/walkforward.py, src/ensemble.py, src/model_attn*.py)         │
│  滚动 / 增量集成；目标：l2 / ret_weighted / lambdarank / ic_linear / fusion │
│  截面 MAD-winsorize + z-score；多 seed bagging；滚动 IC 定权           │
├──────────────────────────────────────────────────────────────────────┤
│  策略层 (src/strategy.py, src/trading.py)                             │
│  long_topk（A股默认可交易）/ cross_sectional（仅研究）                 │
│  每日决策：ML 选池 + 布林带/RSI 择时 + 逐笔订单（可关择时）             │
├──────────────────────────────────────────────────────────────────────┤
│  回测层 (src/backtest_daily.py, src/backtest.py)                      │
│  每日重放：增量续训 → 预测 → decide_orders → simulate_fills(次日行情)   │
│  旧 walk-forward 引擎保留用于对比与单元测试                            │
├──────────────────────────────────────────────────────────────────────┤
│  实盘层 (predict_tomorrow.py, src/live.py)                            │
│  复用同一套 daily engine 代码生成次日可执行订单                        │
├──────────────────────────────────────────────────────────────────────┤
│  交互层 (webapp/, pipeline/, run_session.py)                          │
│  Dash 前端；session = 独立子进程流水线；运行中不可查看，完成后可视化      │
└──────────────────────────────────────────────────────────────────────┘
```

核心数据流：

```
load_factor_panel(keep_trade_cols=True)
    ↓
IncrementalEnsemble 逐日 update/predict/combine
    ↓
trading.decide_orders(state, today, score)  # ML+择时
    ↓
trading.simulate_fills(state, orders, next_bars)  # 次日撮合
    ↓
组合净值 / 逐笔成交 / 绩效指标
```

---

## 3. 目录结构

```
config.py                  # 所有路径、超参、开关的单一真源
pyproject.toml             # uv/pip 依赖

src/
  data/                    # 数据连接器与因子工程
    __init__.py            # load_data / load_factor_panel / list_factor_columns
    base.py                # DataConnector 接口
    tushare.py             # Tushare 连接器（按日拉取全市场、基本面、行业）
    option_features.py     # 期权 IV / QVIX 宏观特征
    factors/               # 统一因子库
      technical.py         # 基础 OHLCV + A股特征（涨跌停、换手率、量比等）
      orthogonal.py        # 异象因子（Amihud、MAX、特质波动、残差动量等）
      alpha158.py          # Qlib Alpha158 家族
      gtja191.py           # 国泰君安 191 精选子集
  model_attn.py            # 截面自注意力模型（可选 leg）
  model_attn_seq.py        # 序列注意力模型
  features.py              # feature_00..NN 矩阵提取（Jane Street 旧格式专用）
  walkforward.py           # 滚动回测引擎（WF）+ 目标/模型分发 + 截面归一化
                           #   + GBM 超参唯一真相源（GBM_PARAMS，ensemble 经映射共用）
  ensemble.py              # IncrementalEnsemble（每日续训+滚动 IC 定权+EMA 平滑）
  strategy.py              # 预测 → 持仓（long_topk / tanh / cross_sectional）
  trading.py               # 每日决策核心 + 逐笔订单 + 成交模拟 + 持仓状态
  risk.py                  # 五层风控叠乘链唯一实现（compute_risk_scale，回测/实盘共用）
  backtest.py              # 周期调仓回测引擎（run_backtest_periodic，唯一口径）
  backtest_daily.py        # 新版每日重放回测引擎
  live.py                  # 实盘推理：ensemble_predict_latest / make_orders_from_state
  live_state.py            # 实盘风控状态（波动/回撤/IC + 持久化预测历史）
  factor_eval.py           # 单因子 IC/ICIR/分层评估 + _windows / _daily_rank_ic 共享函数
  multiseed_eval.py        # 多 seed 回测纯计算（seed_backtest，前端 robustness 会话用）
  measure_alpha.py         # Newey-West alpha/beta 显著性检验（_newey_west_alpha_t）

scripts/                   # 研究脚本（活脚本清单见 scripts/README.md）
  run_walkforward.py       # 旧 walk-forward CLI
  eval_factors.py          # 单因子评估 CLI
  diagnose_factors.py      # 因子层诊断三件套（CLI）
  measure_alpha.py         # Alpha/beta 测量（CLI）
  diagnostics/             # 诊断脚本（diag_decile / diag_linear_vs_gbm / diag_option_iv）
  optimize/                # optimize_daily_params.py（日频参数调优）
  demo/                    # 合成数据端到端冒烟演示
  research/                # multiseed_robust.py（多 seed 稳健性复核）
  archive/                 # 一次性实验脚本归档（ablation/optimize/research，不再维护）

pipeline/                  # 会话式流水线
  registry.py              # mode → 步骤序列 + 参数 schema
  param_env.py             # 参数 → 环境变量映射（轻量，先于 config import 使用）
  context.py               # StepContext + report 构造器
  steps_backtest.py        # 回测模式 4 步实现
  steps_nextday.py         # 次日交易模式 3 步实现
  steps_walkforward.py     # 滚动 Walk-Forward 模式（替代原 walkforward PNG 仪表盘）
  steps_robustness.py      # 多 Seed 稳健性模式（替代原 dashboard_multiseed PNG）

webapp/                    # Dash 前端
  app.py                   # 应用入口与路由（注册：会话/会话详情/行情浏览器/因子实验室）
  sessions.py              # session 创建/调度/删除（detached 子进程 + 日志）
  dataio.py                # 轻量缓存加载（面板/净值/因子元数据）
  pages/                   # sessions、session_detail、market_explorer、factor_lab
  render.py                # 通用 step report 渲染器

run_session.py             # 独立子进程执行一个 session
run_webapp.py              # 启动前端
predict_tomorrow.py        # 生成次日订单 CLI

tests/                     # 单元/集成测试
outputs/                   # 报告、模型、session 产物
models/                    # 模型工件：champion/（生产）+ staging/（候选）+ archive/（退役），见 5.2a
data_cache/                # 数据与中间面板缓存
```

---

## 4. 关键设计约定

### 4.1 配置单一真源：`config.py`

所有可调参数都在 `config.py` 中，并通过 `os.environ` 允许外部覆盖。修改行为优先用环境变量，不要硬编码。

关键开关：

| 环境变量 | 含义 | 默认值 |
|---|---|---|
| `FORWARD_WINDOW` | 前瞻收益窗口（交易日） | 20 |
| `TRAIN_WINDOW` | 滚动训练长度 | 500 |
| `MODEL_OBJECTIVE` | 模型目标 | `ret_weighted` |
| `UNIVERSE_MODE` | 股票池模式 | `all`（全市场） |
| `TOPK_RATIO` | long_topk 持仓比例 | 0.15 |
| `MAX_POSITIONS` | 最大持仓只数 | 30 |
| `REBALANCE_EVERY` | 调仓频率（日） | 10 |
| `ORDER_PRICE_MODE` | 成交模式 | `next_open` |
| `USE_BB_TIMING` | 布林/RSI 择时 | 0（默认关） |
| `USE_IMPACT_COST` | 市场冲击成本 | 1 |
| `MIN_COMMISSION` | 每笔最低佣金（元） | 5.0 |
| `MAX_PARTICIPATION` | 单笔成交占次日成交额上限 | 0.10 |
| `USE_OPTION_FEATURES` | 期权 IV 宏观特征 | 1 |
| `USE_DYNAMIC_TOPK` | 基于滚动 IC 的动态集中度 | 0 |
| `USE_BETA_CONSTRAINT` | 组合 beta 约束（替换高 beta 股） | 0 |
| `MAX_PORTFOLIO_BETA` | 最大允许组合 beta | 1.10 |
| `MULTI_HORIZON_WINDOWS` | 多 horizon 目标窗口（逗号分隔） | 5,10,60 |
| `ENSEMBLE_MODELS` | 增量集成 legs | `gbm_l2,ic_linear` |
| `SEED_COMBINE` | 多 seed 融合方式（walkforward 直接读 config，无兜底） | `mean` |
| `N_SEEDS` | walkforward 窗口内 bagging seed 数（增量集成为 `ENSEMBLE_N_SEEDS`，默认 1） | 3 |
| `IC_WEIGHT_EMA_HALFLIFE` | 滚动 IC 定权 EMA 平滑半衰期（交易日，0=不平滑） | 5 |
| `INCR_TREES` | 增量续训树数 = INCR_TREES × 新成熟天数（总树数上限 FULL_TREES×2，超限提前重建） | 60 |

已删除的开关：`PERIODIC_REBALANCE`（每日重叠引擎 `run_backtest` 已删除，周期调仓是唯一口径，无回落开关）。

### 4.2 数据格式

- **命名因子面板** (`load_factor_panel`)：长格式，列包括 `date`, `symbol`, `<命名因子...>`, `responder_6`，可选 `close/open/high/low/bb_*` 等 meta 列。
- **Jane Street 行格式** (`load_data`)：行是 `(date_id, time_id, symbol_id)`，特征重命名为 `feature_00..feature_NN`。旧入口；`src/model.py`（唯一消费方）已删除，目前仅兼容用途。缓存落盘时附带 `*.features.json` sidecar（因子名 → 列号映射），防止历史缓存列含义错位。

模型实际训练使用**命名因子面板**。

### 4.3 防泄露规则（不可妥协）

1. **标签成熟原则**：训练/定权只能使用 `date <= asof - FORWARD_WINDOW` 的行。`responder_6` 是 `close.shift(-FORWARD_WINDOW)` 的远期收益，在 `date + FORWARD_WINDOW` 之前未知。
2. **embargo**：训练段与测试段之间至少间隔 `EMBARGO_DAYS`（默认等于 `FORWARD_WINDOW`）。
3. **截面归一化按日**：`_cross_sectional_norm` 每天只用当天截面做 MAD-winsorize + z-score，不能跨日。
4. **基本面/财务月度快照**：使用 `pd.merge_asof(..., direction='backward')`，确保每个交易日只看到已公告的最新快照。
5. **实盘决策→执行 1 日滞后**：`decide_orders` 用当日收盘特征；`simulate_fills` 用次日 OHLC。

违反以上任意一条都会让结果不可信。

### 4.4 模型目标

- `l2`：直接回归远期收益，预测在收益空间。
- `ret_weighted`：**默认**。L2 回归但样本权重 ∝ 当日远期收益排名，把拟合重点放到 long_topk 能捕获的头部。
- `lambdarank`：学习排序，直接优化截面排序。
- `ic_linear`：IC 加权线性因子 blend，输出排序分。
- `fusion`：GBM 与 ic_linear 的 rank 平均。

只有 `l2` / `ret_weighted` + `SEED_COMBINE=mean` 时预测才在收益空间，价格 RMSE 才有意义。

新增多 horizon 目标：`multi_ret_weighted`（每 horizon 一个 ret_weighted GBM，rank 融合）、`multi_fusion`（每 horizon GBM+ic_linear 再跨 horizon 融合）。通过 `MULTI_HORIZON_WINDOWS` 与 `MULTI_HORIZON_WEIGHTS` 配置。

### 4.5 成本模型

成本公式唯一实现于 `trading.trade_costs`（每日重放引擎、周期调仓引擎、实盘事前估算共用）：

- 佣金双边：`max(notional × COMMISSION_RATE, MIN_COMMISSION)`（`COMMISSION_RATE = 0.0002`，每笔最低 `MIN_COMMISSION = 5` 元）
- 印花税卖出单边：`STAMP_DUTY_RATE = 0.001`
- 可选平方根市场冲击：`IMPACT_K * sqrt(notional / ADV) * notional`
- 参与率上限：单笔成交 ≤ `MAX_PARTICIPATION`(0.10) × 次日成交额（冲击只计价不限量，此上限防止大单不切实际地全额成交；0 = 不限量）
- A 股规则：涨跌停拦截、T+1 锁仓（按当日新增股数粒度，老仓可卖）、整手（100 股）

---

## 5. 主要入口与使用方式

### 5.1 前端会话（推荐）

```bash
uv run python run_webapp.py
```

浏览器打开后创建 backtest 或 nextday 会话。会话在独立子进程中运行，完成后查看每一步的可视化。

### 5.2 次日订单 CLI

```bash
uv run python predict_tomorrow.py
```

读取当前持仓 `holdings.csv`，生成 `outputs/orders_*.csv`。

### 5.2a 模型门控与自适应重训（champion/staging + 触发式重训 + 偏差跟踪）

**默认不再每晚重训。** 每晚先算跟踪指标，`src/retrain_trigger.py` 做 OR 判定，
任一触发才训练候选：

| 条件 | 阈值（config，均可 env 覆盖） |
| --- | --- |
| 活体 IC 退化 | 成熟样本 ≥ `RETRAIN_IC_MIN_SAMPLES`(10) 且尾部 `RETRAIN_IC_WINDOW`(20) 日 OOS rank-IC 均值 < `RETRAIN_IC_MIN`(0) |
| 收益落后（需 `account.csv` 真实净值） | 尾部 `RETRAIN_UNDERPERF_DAYS`(20) 实盘收益 < 等权基准 − `RETRAIN_UNDERPERF_MARGIN`(0.03) |
| 模型超龄（硬上限） | `train_end_date` 距今 > `MAX_MODEL_AGE_DAYS`(60) 个交易日 |

- **未触发** → 加载 champion 冻结工件直接预测（`live.champion_predict_latest`），
  IC 定权状态每日照常刷新并回存（`train_end_date` 不变）；
- **触发** → 训练候选 → `src/model_gate.py` 门控：配方指纹与 champion 相同 →
  原位刷新工件；不同 → 候选落盘 `models/staging/<id>/`，双方各跑一遍最近
  `PROMOTE_EVAL_DAYS`（默认 120）个交易日的完整每日重放，**候选 Sharpe 更高
  且最大回撤不劣于 champion − `PROMOTE_DD_TOLERANCE`**（默认 2pp）才晋升；
  被拒则用冻结 champion 出信号（订单永远出自 champion）。

**回测 vs 实盘偏差跟踪**（`src/live_tracking.py`）：每晚追加一行到
`outputs/tracking/live_daily.parquet`，重生成 `tracking_report.json` ——
晋升时回测预期 × 实盘实际（活体 IC + 真实净值表现）× 偏差 × 告警
（IC 退化 / 收益落后 / 方向背离 / 临近超龄）。无 `account.csv` 时净值维度
降级为 None 并告警提示，IC 维度照常。

```bash
uv run python predict_tomorrow.py --force-retrain      # 无视指标强制重训
uv run python predict_tomorrow.py --tracking-report    # 打印偏差报告后退出
uv run python predict_tomorrow.py --no-gate            # 跳过门控（每晚重训旧行为）
uv run python predict_tomorrow.py --promote <id>       # 人工晋升 staging 候选
USE_ADAPTIVE_RETRAIN=0 uv run python predict_tomorrow.py  # 关自适应，每晚重训
USE_MODEL_GATE=0 uv run python predict_tomorrow.py        # 整体关门控
```

门控指标与 webapp 回测报告同源：`src/perf_metrics.py::compute_performance`；
活体 IC 与 IC 降仓层同源：`live_state.collect_matured_ics`（纯函数）。
`USE_ADAPTIVE_RETRAIN=0` 时退回"每晚重训 + 门控"（即初版门控行为）。

### 5.3 旧 walk-forward CLI

```bash
uv run python scripts/run_walkforward.py
```

用于跑滚动窗口研究，产出 OOS dashboard。

### 5.4 单元测试

```bash
uv run pytest tests/ -v
```

### 5.5 优化实验

```bash
# 真实数据：扫描 REBALANCE_EVERY × TOPK × MAX_POSITIONS × MIN_NOTIONAL × ORDER_PRICE
# 一次训练，复用 score_cache，避免每个组合都重训模型
uv run python scripts/optimize/optimize_daily_params.py

# 合成数据演示（验证框架本身）
uv run python scripts/demo/demo_optimize_daily_synthetic.py
uv run python scripts/demo/demo_liquidity_ablation_synthetic.py
```

更早的消融/参数扫描脚本（attn、multi-horizon、风控消融等）已归档到
`scripts/archive/`（一次性实验，不再维护；清单见 `scripts/README.md`）。

---

## 6. 常见陷阱

1. **缓存污染**：`data_cache/dataset_*.parquet` 的签名包含因子开关，但如果改的是签名没覆盖到的参数（如截面归一化实现本身），会读到旧缓存。调试新逻辑时主动删除缓存。
2. **未复权价格**：必须用 `adj_close` 计算远期收益，否则除权除息日会注入虚假收益。
3. **小股票池虚高**：49/300 只池子破坏截面相关性，结果不稳定。研究用全市场 `UNIVERSE_MODE=all`。
4. **每日重叠调仓虚高**：旧的每日重叠引擎 `run_backtest`（Sharpe/PnL 双重虚高）与 `PERIODIC_REBALANCE` 开关已一并删除，周期调仓 `run_backtest_periodic` 是唯一回测口径——不要再引入"每天赚 pos×N日收益"的写法。
5. **训练窗口过大导致 OOS 过短**：`TRAIN_WINDOW=750` 把 OOS 压缩到 7 个窗口，NW-t 会掉到 1.74。保持 500。
6. **BB/RSI 择时默认应关闭**：历史归因证明它只稀释信号，不增加价值。
7. **Regime 空仓默认应关闭**：滞后趋势规则在 V 型反弹里追涨杀跌。
8. **公司行动不入账（known limitation）**：回测只做务实级除权处理——估值按 `pre_close` 链条调整（`backtest_daily._price_map` 维护每股累计 `val_factor`，单次调整幅度超 [1/3, 3] 判定为脏数据跳过并计数），消除除权日净值假性盈亏；但现金分红/送股不入账（无全量公司行动引擎），长期回测会轻微低估真实收益。成交价始终用原始 OHLC，不受估值因子影响。
9. **score_cache 必须带指纹复用**：复用缓存分跳过训练时，传 `{"scores": ..., "fingerprint": meta["score_fingerprint"]}` 包装；指纹（特征列+ensemble 配置+前瞻窗）不匹配会被引擎拒绝，无指纹的旧 pkl 只警告放行——特征或模型改过后切勿复用旧缓存。
10. **冻结期回测≠实盘分布**：自适应重训（5.2a）下实盘在触发间隔内用冻结 champion，而回测引擎每日增量重训——两者分布不同，回测净值会系统性偏优。`outputs/tracking/tracking_report.json` 的 divergence/方向背离告警就是度量这个偏差的；看到持续背离先怀疑分布漂移，不要直接信回测 Sharpe。

---

## 7. 扩展系统

### 7.1 添加新数据源

实现 `src/data/base.py` 的 `DataConnector`，在 `src/data/__init__.py` 的 `_CONNECTORS` 注册。

### 7.2 添加新因子家族

1. 在 `src/data/factors/` 新建模块，实现 `compute_<family>_features(df)`。
2. 在 `src/data/factors/__init__.py` 的 `FACTORY` 注册。
3. 在 `config.py` 添加 `USE_<FAMILY>_FACTORS` 开关。
4. 更新 `load_data` 的缓存签名以包含新开关。

### 7.3 添加新模型目标

在 `src/walkforward.py` 的 `_train_predict` 分发；如果是每日增量集成 leg，在 `src/ensemble.py` 的 `predict` 中处理。

### 7.4 添加流水线模式/步骤

修改 `pipeline/registry.py` 的 `STEPS` 与 `PARAM_SCHEMA`，新增步骤函数在 `pipeline/steps_*.py` 中实现。

---

## 8. 命名与代码风格

- Python 3.10+，类型注解 encouraged。
- line-length 100（black/ruff）。
- 模块级常量 `UPPER_CASE`，函数/变量 `snake_case`。
- 配置变量优先从 `config as C` 读取。
- 所有可能昂贵的操作（API 调用、模型训练）显式打印进度。
- 涉及“是否含未来信息”的操作必须加注释说明防泄露机制。

---

## 9. 相关记忆

项目记忆在 `~/.claude/projects/-home-webdrag0n-stock-predict-v2/memory/`。与本项目高度相关的几条：

- `stock-predict-v2-architecture`：流水线连接方式。
- `stock-predict-v2-walkforward`：泄露修复与关键结论。
- `stock-predict-v2-layer-attribution`：问题在预测层 + 每日调仓成本。
- `stock-predict-v2-combiner-bottleneck`：组合方式瓶颈，ic_linear 提升显著。
- `stock-predict-v2-phase2-alpha-beta`：Phase2 回测优化终局。
- `stock-predict-v2-daily-trading-engine`：每日策略 + 逐笔模拟重构。
- `stock-predict-v2-whole-market`：全市场结果与纠错教训。
- `stock-predict-v2-session-pipeline`：前端会话架构。

阅读或修改相关模块前，先回顾对应记忆。
