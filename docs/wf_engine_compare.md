# Walk-forward 引擎结果对比

## 研究口径

- windowed：`outputs/walkforward/meta.json`，生成于 2026-11-04。
- incremental：`outputs/experiments/wf_engine_compare/incremental/walkforward/meta.json`，生成于 2026-11-15，退出码 0。
- 两者均为 9 个窗口、540 个 OOS 交易日、初始资金 100 万元、默认研究持仓口径。
- 注意：两次运行的数据快照不同。windowed 的 OOS 为 2024-03-25～2026-08-11，incremental 为 2024-04-03～2026-08-20，起止均相差 7 个交易日。因此下表用于记录当前结果，不应把全部指标差异都解释为引擎本身的因果影响。

| 指标 | windowed | incremental |
|---|---:|---:|
| Sharpe（净） | 2.5606 | 1.1100 |
| 年化收益 | 36.17% | 20.07% |
| 总净 PnL | 112.46% | 48.52% |
| 最大回撤 | -9.27% | -17.68% |
| 总成本 / 本金 | 12.79% | 15.80% |
| 平均仓位 | 82.69% | 55.93% |
| PredIC | 0.10242 | 0.02434 |
| PredICIR | 0.85408 | 0.21614 |
| Sharpe 为正窗口 | 7 / 9 | 8 / 9 |
| PredIC 为正窗口 | 9 / 9 | 6 / 9 |
| Newey–West t | 未记录 | 1.05345 |
| 年化 alpha | 未记录 | 12.91% |

## 结论

线上同款 `IncrementalEnsemble` 已有完整逐日增量 OOS 结果，但其 Sharpe、回撤和 PredIC 均弱于既有 windowed 快照，不能把 windowed 的头条指标直接视为线上预期。后续涉及 3 万元口径或成本敏感性的对比，应固定同一数据快照与 OOS 区间。

运行日志：`outputs/experiments/wf_engine_compare_incremental.log`。数据准备阶段有 159 只股票估值获取失败，北向持股与部分 yfinance 宏观数据触发频率限制；完整警告见日志。


## 3 万口径

运行参数：`engine=incremental`、`capital=30000`、`max_positions=6`、`max_weight=0.2`、`min_notional=2000`、`allowed_boards=["main"]`、`skip_unaffordable=True`。结果位于 `outputs/experiments/wf_engine_compare/small30k/walkforward/meta.json`，生成于 2026-11-15，退出码 0。

本次与上方 incremental 使用相同数据快照和 OOS 区间（2024-04-03～2026-08-20），PredIC 完全相同（0.024339），可直接比较交易口径差异。

| 指标 | 100 万研究口径 incremental | 3 万口径 incremental |
|---|---:|---:|
| Sharpe（净） | 1.1100 | 0.8879 |
| 年化收益 | 20.07% | 14.34% |
| 总净 PnL | 48.52% | 32.28% |
| 最大回撤 | -17.68% | -22.07% |
| 总成本 / 本金 | 15.80% | 10.44% |
| 佣金+印花税 / 本金 | 3.77% | 7.16% |
| 冲击 / 本金 | 12.03% | 3.27% |
| 平均仓位 | 55.93% | 82.04% |
| Sharpe 为正窗口 | 8 / 9 | 9 / 9 |
| Newey–West t | 1.05345 | 0.79500 |
| 年化 alpha | 12.91% | 9.16% |

小账户的总冲击成本占比下降，但最低佣金使佣金与印花税占比几乎翻倍；净 Sharpe 和最大回撤均弱于 100 万研究口径。该结果不改变研究默认参数，生产夜间入口 `daily_run.sh` 已使用本节的主板过滤与补位规则。

运行日志：`outputs/experiments/wf_engine_compare_small30k.log`。
