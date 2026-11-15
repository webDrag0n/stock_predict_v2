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
