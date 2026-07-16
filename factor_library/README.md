# 因子公式语言 — 快速参考

## 什么是公式因子？

在 `factor_library/` 目录下写 YAML 文件，定义因子的 ID、名称、公式字符串和标签。
系统自动：校验、提取依赖、排序执行、集成到因子面板、缓存结果。
**不需要写 Python 代码，不需要改 `config.py`。**

## YAML 格式

```yaml
factors:
  - id: my_factor_id           # 唯一列名，必填
    name: "人类可读名称"         # 必填
    description: "详细说明"      # 可选
    category: momentum          # 可选，用于分组/筛选
    source: "论文标题或URL"       # 可选
    formula: "adj_close / ts_delay(adj_close, 5) - 1"  # 必填
    tags: [momentum, price]     # 可选
    expected_sign: 1            # 可选，+1 或 -1
```

## 可用列名

### 原始 OHLCV（总是可用）
`open` `high` `low` `close` `volume` `amount` `pct_chg`

### 技术因子产出（总可用）
`adj_close` `return_1d` `return_5d` `return_10d` `return_20d`
`log_return` `momentum_5` `momentum_10` `momentum_20`
`volatility_5` `volatility_10` `volatility_20`
`ma_5` `ma_10` `ma_20` `ma_60`
`price_to_ma_5` `price_to_ma_10` `price_to_ma_20` `price_to_ma_60`
`rsi_14` `macd` `macd_signal` `macd_diff`
`bb_position_20` `stoch_k` `stoch_d`
`hl_pct` `hl_ma_5`
`return_1d_rank` `volume_rank` `volatility_20_rank`
`turnover_rate` `turnover_ma_5` `turnover_ma_20` `turnover_ratio`
`volume_ratio` `limit_flag` `near_limit`

### 正交因子
`amihud_20` `max_ret_20` `idio_vol_20` `resid_mom_20` `ret_skew_20` `high_52w_prox`

### Alpha158
`a158_kmid` `a158_klen` `a158_kup` `a158_klow` `a158_ksft` `a158_ksft2`
`a158_rsqr{5,20,60}` `a158_resi{5,20,60}` `a158_qtlu{5,20,60}` `a158_qtld{5,20,60}`
`a158_rsv{5,20,60}` `a158_imax{5,20,60}` `a158_imin{5,20,60}` `a158_imxd{5,20,60}`
`a158_corr{5,20,60}` `a158_cord{5,20,60}` `a158_cntp{5,20,60}` `a158_cntn{5,20,60}`
`a158_cntd{5,20,60}` `a158_sump{5,20,60}` `a158_sumd{5,20,60}` `a158_wvma{5,20,60}`

### GTJA191（如启用）
`gtja_alpha001` ~ `gtja_alpha102`

### 基本面（如启用）
`ep` `bp` `sp` `ln_mv` `roe` `gross_margin` `revenue_yoy` `profit_yoy`

## DSL 原语（来自 `_base.py`）

| 函数 | 签名 | 说明 |
|---|---|---|
| `ts_delay(s, n)` | `(Series, int) -> Series` | 滞后 n 天 |
| `ts_delta(s, n)` | `(Series, int) -> Series` | s - delay(s, n) |
| `ts_sum(s, n)` | `(Series, int) -> Series` | n 天滚动和 |
| `ts_mean(s, n)` | `(Series, int) -> Series` | n 天滚动均值 |
| `ts_std(s, n)` | `(Series, int) -> Series` | n 天滚动标准差 |
| `ts_max(s, n)` | `(Series, int) -> Series` | n 天滚动最大值 |
| `ts_min(s, n)` | `(Series, int) -> Series` | n 天滚动最小值 |
| `ts_skew(s, n)` | `(Series, int) -> Series` | n 天滚动偏度 |
| `ts_corr(a, b, n)` | `(Series, Series, int) -> Series` | n 天滚动 Pearson 相关 |
| `ts_rank(s, n)` | `(Series, int) -> Series` | 时间序列排序 (1..n) |
| `ts_quantile(s, n, q)` | `(Series, int, float) -> Series` | n 天滚动 q 分位数 |
| `ts_reg_resid(s, n)` | `(Series, int) -> Series` | 线性趋势残差 |
| `ts_reg_r2(s, n)` | `(Series, int) -> Series` | 线性趋势 R² |
| `cs_rank(s)` | `(Series) -> Series` | 截面百分位排序 (0,1] |
| `cs_zscore(s)` | `(Series) -> Series` | 截面 z-score |
| `signed_power(s, e)` | `(Series, float) -> Series` | sign(s) * \|s\|^e |
| `safe_div(a, b)` | `(Series, Series) -> Series` | 安全除法 (零分母保护) |

## 安全数学函数

`log(x)` `abs(x)` `sqrt(x)` `sign(x)` `exp(x)` `log2(x)` `log10(x)` `round(x)` `min(a,b)` `max(a,b)`

## 支持的运算符

算术: `+ - * / ** %`
比较: `== != < <= > >=`
逻辑: `and or not`
三元: `x if cond else y`
括号分组

## 公式示例

```yaml
# 简单动量
formula: "adj_close / ts_delay(adj_close, 5) - 1"

# 波动率调整动量
formula: "safe_div(adj_close / ts_delay(adj_close, 20) - 1, volatility_20 + 0.005)"

# 趋势质量
formula: "ts_reg_r2(close, 20)"

# 放量突破 (量 + 价双条件)
formula: "safe_div(volume, ts_mean(volume, 20)) * (close / ts_delay(close, 1) - 1)"

# 反转因子
formula: "-signed_power(adj_close / ts_delay(adj_close, 5) - 1, 2)"

# 量价背离
formula: "-ts_corr(cs_rank(ts_delta(log(volume), 1)), cs_rank((close - open) / open), 6)"
```

## 验证与调试

```bash
# 校验所有 YAML 公式
python -m src.data.factors.registry validate

# 列出所有因子
python -m src.data.factors.registry list --category momentum

# 搜索因子
python -m src.data.factors.registry search volume

# 查看单个因子详情
python -m src.data.factors.registry info my_factor_id

# 测试加载含公式因子的面板
USE_FORMULA_FACTORS=1 uv run python -c "
from data import load_factor_panel
panel = load_factor_panel(start_date='2024-06-01', end_date='2024-06-30')
print([c for c in panel.columns if c.startswith('mom_') or c.startswith('vol_')])
"
```

## 依赖管理

系统自动从公式 AST 提取列名依赖并生成拓扑排序。不需要手动声明依赖，除非公式使用了**不透明的动态查找**（极少见，此时用 `explicit_deps`）。

依赖会自动解析，支持多层：
- Level 1: 依赖原生因子 → 直接可用
- Level 2: 依赖其他公式因子 → 自动排序到生产者之后
- Level 3: 依赖不存在 → 警告并跳过

## 启用/禁用

- 全局开关：`config.py` 的 `USE_FORMULA_FACTORS`（默认 `0` = 关闭）
- 库级排除：`_manifest.yaml` 的 `exclude_ids`
- 子集模式：`FORMULA_FACTOR_SUBSET` 环境变量或 config 设置
- 单因子开关：YAML 中的 `enabled: false`
