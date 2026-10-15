# scripts/ 目录说明

## 活脚本（有被引用的入口 / 库）

| 脚本 | 用途 | 引用方 |
|------|------|--------|
| `run_walkforward.py` | walk-forward 回测 CLI | DEVELOPMENT.md、CLAUDE.md |
| `eval_factors.py` | 单因子评估 CLI | SKILLS.md、DEVELOPMENT.md、README_CN.md |
| `diagnose_factors.py` | 因子诊断三件套 CLI | webapp/pages/factor_lab.py |
| `measure_alpha.py` | Newey-West alpha（被当库 import） | pipeline/steps_backtest.py |
| `diagnostics/` | diag_decile / diag_linear_vs_gbm / diag_option_iv（诊断工具箱） | DEVELOPMENT.md |
| `demo/` | demo_optimize_daily_synthetic / demo_liquidity_ablation_synthetic（合成数据端到端冒烟） | 修复方案全局验证清单、CLAUDE.md |
| `optimize/optimize_daily_params.py` | 日频参数调优 | CLAUDE.md |
| `research/multiseed_robust.py` | 多 seed 稳健性 | CLAUDE.md |

## 已删除（阶段 5 架构收敛）

- `plot_multiseed_dashboard.py` — 多 seed PNG 仪表盘；计算逻辑上移为
  `src/multiseed_eval.py::seed_backtest`，展示由前端「多Seed稳健性」会话接管。
- `run_pipeline.py` — 旧全流程 CLI，随其唯一模型实现 `src/model.py`（无 embargo
  切分 + 驻留式 CV）一并删除；README/README_CN 中的相关命令已同步移除。
- `archive/optimize/event_sweep.py` 依赖的 `src/backtest_event.py`（一日前视
  引擎）已删除——归档脚本不维护，保持原样仅供参考。

## 归档（`archive/`）：一次性实验脚本，不再维护

- `archive/ablation/` — 2022–2024 消融实验（attn / multi-horizon / 风控）全套。
- `archive/optimize/` — 一次性参数扫描：`event_sweep.py`（依赖已删除的
  `src/backtest_event.py`，**不可运行**，仅历史参考）、`ensemble_opt.py`、
  `optimize_accuracy.py`、`optimize_topk.py`、`optimize_topk_params.py`、
  `rebalance_sweep.py`。
- `archive/research/` — `realistic_cost.py`、`reproduce_ensemble.py`、
  `reexplore_corrected.py`、`simulate_short.py`。

归档脚本按原目录结构保留，仅作历史参考：其中的 sys.path 相对层级与 import
（如 `from measure_alpha import ...`）按旧位置书写，直接运行需要自行修正路径。
