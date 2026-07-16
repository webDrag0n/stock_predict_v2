# SKILLS.md — 本项目可用的 Claude Code Skills

本文件说明在 `stock-predict-v2` 中使用 Claude Code Skills 的约定。

---

## 1. 常用 Skill 速查

| Skill | 何时调用 | 典型用法 |
|---|---|---|
| `verify` | 修改了有运行时表面的代码（模型、回测、交易、数据加载） | `/verify` 或 `Skill({"skill":"verify"})` |
| `run` | 用户要求启动/截图 app | `/run` 启动 webapp |
| `code-review` | 提交前或用户要求 review 当前 diff | `/code-review --comment` / `/code-review --fix` |
| `simplify` | 代码能跑但希望清理/简化 | `/simplify` |
| `dataviz` | 任何新增/修改图表、KPI 卡片、dashboard | `/dataviz` |
| `update-config` | 用户要求“以后自动…”或添加权限 | `/update-config` |

---

## 2. verify — 验证改动是否真生效

### 2.1 触发时机

只要改动了以下任一来源文件，就应该跑 `verify`：

- `src/data/` 数据加载或因子工程
- `src/walkforward.py`, `src/ensemble.py`, `src/model*.py`
- `src/strategy.py`, `src/trading.py`, `src/backtest*.py`
- `src/live.py`, `predict_tomorrow.py`
- `pipeline/steps_*.py`

### 2.2 验证策略

`verify` 应驱动受影响的端到端流程，而不仅仅是跑测试：

- **改因子/数据加载**：跑 `tests/test_backtest_synthetic.py` + 清缓存后跑一次最小 walk-forward，检查 `responder_6` 统计量与 IC。
- **改模型/目标**：固定 seed 跑一个 2~3 窗口的 walk-forward，对比预测 IC 与 OOS Sharpe。
- **改交易/回测**：跑 `tests/test_backtest_synthetic.py` 和 `tests/test_sim2real_fixes.py`，检查现金守恒与 T+1。
- **改前端/流水线**：创建并运行一个 backtest session，确认步骤产物正确渲染。

### 2.3 成功标准

- 无异常、无断言失败。
- 关键指标方向与预期一致（例如修复 bug 后 alpha 下降而非上升）。
- 现金守恒在 `STRICT_CHECK=True` 下通过。

---

## 3. run — 启动应用

### 3.1 启动前端

```bash
uv run python run_webapp.py
```

默认端口 `8050`。可打开浏览器访问 `http://localhost:8050`。

### 3.2 启动次日订单 CLI

这不是长期服务，但用户说“跑一下实盘命令”时：

```bash
uv run python predict_tomorrow.py
```

### 3.3 注意

- 不要在前端正在运行 session 时重启 webapp，会丢失子进程跟踪。
- 若端口被占用，检查是否有残留的 `python run_webapp.py` 进程。

---

## 4. code-review / simplify — 代码审查与清理

### 4.1 code-review

在以下场景调用：

- 用户说“review 一下这段改动”。
- 准备提交前。
- 修改了复杂的多文件 PR。

关注重点（按优先级）：

1. **泄露风险**：是否用了未来信息？训练/测试边界是否正确？`merge_asof` 方向是否 backward？
2. **正确性**：现金守恒、T+1、涨跌停、整手、成本计算。
3. **性能**：是否遗漏 `reduce_mem_usage`、是否把大数组塞进 JSON。
4. **可维护性**：是否重复实现已有函数、配置是否硬编码。

### 4.2 simplify

用于清理已通过验证的代码：

- 删除重复逻辑（如多个地方手写整手取整）。
- 把 magic number 抽到 `config.py`。
- 简化 pandas groupby apply（pandas 3.x 有 `include_groups=False` 要求）。

**注意**：`simplify` 不 hunt bugs，复杂逻辑先用 `code-review`。

---

## 5. dataviz — 图表与可视化

### 5.1 触发条件

只要新增或修改任何图表，包括：

- `pipeline/context.py` 的 `fig_*` 构造器
- `src/walkforward.py:plot_dashboard`
- `webapp/render.py` 对 figure spec 的渲染
- 任何诊断脚本中的 `matplotlib`/`plotly` 图

### 5.2 设计约定

- 跟随项目已有的 light-theme 风格（`webapp/theme.py` 的 `PALETTE`）。
- 折线图：策略用 `#1a7f37`（绿），基准用 `#ff7f0e`（橙），回撤用 `#d62728`（红），alpha 用 `#7b3fb5`（紫）。
- 柱状图：`diverging=True` 时正负自动着色。
- 直方图：服务端预分箱（参考 `fig_hist`），不要把原始数组塞进 JSON。
- 图表标题用中文（面向国内用户）。

### 5.3 验证

修改后启动 webapp，创建一个 backtest session，检查各步骤图表正常显示。

---

## 6. update-config — 自动化与权限

### 6.1 常见需求

用户说以下句式时调用：

- “以后自动允许 uv 命令”
- “运行 python 时不要老问我”
- “把 TUSHARE_TOKEN 设为环境变量”

### 6.2 推荐配置

项目级 `.claude/settings.json` 可添加：

```json
{
  "permissions": {
    "allow": [
      "Bash(uv *)",
      "Bash(.venv/bin/python *)",
      "Bash(python run_*.py)",
      "Bash(rm -rf data_cache/dataset_*.parquet)"
    ]
  }
}
```

**不要把 `TUSHARE_TOKEN` 明文写进 settings.json**，继续用 `config_credentials.toml` 或环境变量。

---

## 7. 何时不使用 Skill

不要为以下情况调用 Skill：

- 纯信息查询（如“某个文件在哪”）——直接搜索。
- 简单文档修改（如改 README 一句话）——直接 Edit。
- 已经明确知道是一次性探索（如临时画张图看分布）——可以调用 `dataviz` 也可以直接写代码，视复杂度而定。

---

## 8. Skill 组合流程示例

### 场景：修复一个回测现金守恒 bug

1. 定位问题并修复。
2. `/code-review` 检查泄露与正确性。
3. `/verify` 跑 `test_backtest_synthetic.py` 与一次最小 walk-forward。
4. 若涉及图表（如新增现金曲线），`/dataviz`。
5. `/simplify` 清理重复代码。

### 场景：新增一个因子家族

1. 实现因子模块。
2. `/verify` 跑合成面板 + `scripts/eval_factors.py`。
3. `/code-review` 检查性能与缓存签名。
4. 若在前端展示 IC 图，调用 `/dataviz`。

---

## 9. 自定义 Skill 提示

当前项目没有自定义 Skill。如需添加（例如 `/backtest` 一键跑验证），可在 `.claude/skills/` 下创建 skill 文件，并在本文件记录。
