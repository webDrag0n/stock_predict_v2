"""模式 → 步骤序列 + 可调参数 schema 的注册表。

run_session.py 按 STEPS[mode] 顺序执行步骤；前端按 PARAM_SCHEMA[mode]
动态渲染创建会话的参数表单。新增模式/步骤只改这里。

参数同时被写入 os.environ（PARAM_ENV 映射）→ config 在子进程启动时读取，
从而无侵入地覆盖 TOPK_RATIO / TRAIN_WINDOW / 增量集成等引擎超参；其余参数
（资金/初始持仓/日期区间/择时阈值）由步骤函数从 ctx.params 直接读取。
"""
from __future__ import annotations

from . import steps_backtest as B
from . import steps_nextday as N

# mode -> [(step_id, title, fn)...]
STEPS = {
    "backtest": [
        ("data_import", "数据导入", B.data_import),
        ("factor_analysis", "因子分析", B.factor_analysis),
        ("daily_replay", "每日重放回测", B.daily_replay),
        ("performance", "绩效分析", B.performance),
    ],
    "nextday": [
        ("data_import", "数据导入（实盘）", N.data_import),
        ("train_predict", "训练 + 预测", N.train_predict),
        ("generate_orders", "生成次日订单", N.generate_orders),
    ],
}

MODE_LABELS = {"backtest": "回测", "nextday": "次日交易"}

# 参数 schema：前端按此渲染表单。type ∈ number|select|bool|text
PARAM_SCHEMA = {
    "backtest": [
        # --- 账户初始状态 ---
        {"key": "capital", "label": "初始资金(元)", "type": "number",
         "default": 1000000, "min": 10000, "max": 1000000000, "step": 10000},
        {"key": "initial_holdings", "label": "初始持仓(每行 代码:股数)", "type": "textarea",
         "default": "", "placeholder": "600519.SH:200\n000001.SZ:1000",
         "hint": "留空=空仓"},
        {"key": "start_date", "label": "回测开始日(YYYY-MM-DD)", "type": "text",
         "default": "", "hint": "留空=最近 3 个月 (今天 - 92 天)"},
        {"key": "end_date", "label": "回测结束日(YYYY-MM-DD)", "type": "text",
         "default": "", "hint": "留空=到最新"},
        # --- 选股 / 择时 ---
        {"key": "topk", "label": "候选池比例 TOPK", "type": "number",
         "default": 0.15, "min": 0.02, "max": 0.50, "step": 0.01},
        {"key": "max_positions", "label": "最大持仓数", "type": "number",
         "default": 30, "min": 1, "max": 200, "step": 1},
        {"key": "max_weight", "label": "单股权重上限", "type": "number",
         "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01},
        {"key": "use_bb_timing", "label": "启用布林/RSI择时", "type": "bool",
         "default": True, "hint": "关=仅按ML排序选股"},
        {"key": "bb_buy_k", "label": "布林下轨买入容差", "type": "number",
         "default": 0.02, "min": 0.0, "max": 0.20, "step": 0.01,
         "hint": "close≤下轨×(1+此值)才买"},
        {"key": "bb_sell_k", "label": "布林上轨卖出容差", "type": "number",
         "default": 0.02, "min": 0.0, "max": 0.20, "step": 0.01},
        {"key": "rsi_buy_max", "label": "RSI买入上限", "type": "number",
         "default": 70, "min": 0, "max": 100, "step": 5,
         "hint": "RSI高于此不追买；0=禁用"},
        {"key": "rsi_sell_min", "label": "RSI卖出下限", "type": "number",
         "default": 80, "min": 0, "max": 100, "step": 5,
         "hint": "RSI高于此强制卖；100=禁用"},
        {"key": "order_price_mode", "label": "挂单价模型", "type": "select",
         "default": "limit_bb",
         "options": ["limit_bb", "next_open"],
         "hint": "limit_bb=布林下轨挂限价(换手/成本最低,扫描最优);next_open=次日开盘市价"},
        {"key": "min_notional", "label": "最小订单金额(元)", "type": "number",
         "default": 30000, "min": 0, "max": 1000000, "step": 1000,
         "hint": "越大换手越低;扫描显示≥3万显著降成本"},
        # --- 动态集中度(基于滚动IC) ---
        {"key": "use_dynamic_topk", "label": "启用动态TOPK", "type": "bool",
         "default": False, "hint": "IC高时集中,IC低时分散;默认关,需实验验证"},
        {"key": "dynamic_topk_ic_full", "label": "动态TOPK满仓IC阈值", "type": "number",
         "default": 0.02, "min": -0.05, "max": 0.10, "step": 0.005,
         "hint": "滚动IC≥此值时使用最小topk"},
        {"key": "dynamic_topk_ic_zero", "label": "动态TOPK低仓IC阈值", "type": "number",
         "default": -0.01, "min": -0.10, "max": 0.05, "step": 0.005,
         "hint": "滚动IC≤此值时使用最大topk"},
        {"key": "dynamic_topk_low", "label": "动态TOPK最小值", "type": "number",
         "default": 0.10, "min": 0.02, "max": 0.50, "step": 0.01},
        {"key": "dynamic_topk_high", "label": "动态TOPK最大值", "type": "number",
         "default": 0.25, "min": 0.02, "max": 0.80, "step": 0.01},
        {"key": "use_beta_constraint", "label": "启用组合Beta约束", "type": "bool",
         "default": False, "hint": "用return_1d估算 trailing beta, 高beta替换为低beta候选"},
        {"key": "max_portfolio_beta", "label": "最大组合Beta", "type": "number",
         "default": 1.10, "min": 0.50, "max": 2.00, "step": 0.05},
        {"key": "beta_lookback", "label": "Beta估算窗口(日)", "type": "number",
         "default": 60, "min": 20, "max": 252, "step": 5},
        # --- 增量集成 ---
        {"key": "train_window", "label": "训练窗口(交易日)", "type": "number",
         "default": 500, "min": 120, "max": 1500, "step": 30},
        {"key": "rebuild_days", "label": "全量重建周期(交易日)", "type": "number",
         "default": 60, "min": 10, "max": 250, "step": 10,
         "hint": "其余日做增量续训"},
        {"key": "incr_trees", "label": "增量续训树数", "type": "number",
         "default": 60, "min": 10, "max": 300, "step": 10},
        {"key": "ic_window", "label": "滚动IC定权窗口", "type": "number",
         "default": 20, "min": 5, "max": 120, "step": 5},
        {"key": "n_seeds", "label": "模型 seed 数", "type": "number",
         "default": 1, "min": 1, "max": 5, "step": 1,
         "hint": "越多越稳但越慢"},
        {"key": "use_impact", "label": "计入市场冲击成本", "type": "bool",
         "default": True},
    ],
    "nextday": [
        {"key": "capital", "label": "组合资金(元)", "type": "number",
         "default": 1000000, "min": 10000, "max": 1000000000, "step": 10000},
        {"key": "initial_holdings", "label": "当前持仓(每行 代码:股数)", "type": "textarea",
         "default": "", "placeholder": "600519.SH:200\n000001.SZ:1000",
         "hint": "留空=空仓"},
        {"key": "topk", "label": "候选池比例 TOPK", "type": "number",
         "default": 0.15, "min": 0.02, "max": 0.50, "step": 0.01},
        {"key": "max_positions", "label": "最大持仓数", "type": "number",
         "default": 30, "min": 1, "max": 200, "step": 1},
        {"key": "max_weight", "label": "单股权重上限", "type": "number",
         "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01},
        {"key": "use_bb_timing", "label": "启用布林/RSI择时", "type": "bool",
         "default": True},
        {"key": "order_price_mode", "label": "挂单价模型", "type": "select",
         "default": "limit_bb", "options": ["limit_bb", "next_open"],
         "hint": "limit_bb=布林下轨挂限价(换手/成本最低)"},
        {"key": "min_notional", "label": "最小订单金额(元)", "type": "number",
         "default": 30000, "min": 0, "max": 1000000, "step": 1000},
        # --- 动态集中度(基于滚动IC) ---
        {"key": "use_dynamic_topk", "label": "启用动态TOPK", "type": "bool",
         "default": False, "hint": "IC高时集中,IC低时分散;默认关"},
        {"key": "dynamic_topk_ic_full", "label": "动态TOPK满仓IC阈值", "type": "number",
         "default": 0.02, "min": -0.05, "max": 0.10, "step": 0.005},
        {"key": "dynamic_topk_ic_zero", "label": "动态TOPK低仓IC阈值", "type": "number",
         "default": -0.01, "min": -0.10, "max": 0.05, "step": 0.005},
        {"key": "dynamic_topk_low", "label": "动态TOPK最小值", "type": "number",
         "default": 0.10, "min": 0.02, "max": 0.50, "step": 0.01},
        {"key": "dynamic_topk_high", "label": "动态TOPK最大值", "type": "number",
         "default": 0.25, "min": 0.02, "max": 0.80, "step": 0.01},
        {"key": "use_beta_constraint", "label": "启用组合Beta约束", "type": "bool",
         "default": False, "hint": "高beta替换为低beta候选"},
        {"key": "max_portfolio_beta", "label": "最大组合Beta", "type": "number",
         "default": 1.10, "min": 0.50, "max": 2.00, "step": 0.05},
        {"key": "n_seeds", "label": "模型 seed 数", "type": "number",
         "default": 1, "min": 1, "max": 5, "step": 1},
        {"key": "train_window", "label": "训练窗口(交易日)", "type": "number",
         "default": 500, "min": 120, "max": 1500, "step": 30,
         "hint": "越小越快，建议实盘 250~500"},
        {"key": "rebuild_days", "label": "全量重建周期(交易日)", "type": "number",
         "default": 60, "min": 10, "max": 250, "step": 10},
        {"key": "incr_trees", "label": "增量续训树数", "type": "number",
         "default": 60, "min": 10, "max": 300, "step": 10},
        {"key": "ic_window", "label": "滚动IC定权窗口", "type": "number",
         "default": 20, "min": 5, "max": 120, "step": 5},
    ],
}

# 参数 key -> config 读取的环境变量名（在子进程启动前写入 os.environ）
PARAM_ENV = {
    "train_window": "TRAIN_WINDOW",
    "topk": "TOPK_RATIO",
    "max_positions": "MAX_POSITIONS",
    "n_seeds": "N_SEEDS",
    "rebuild_days": "REBUILD_DAYS",
    "incr_trees": "INCR_TREES",
    "ic_window": "IC_WINDOW",
    "order_price_mode": "ORDER_PRICE_MODE",
    "dynamic_topk_ic_full": "DYNAMIC_TOPK_IC_FULL",
    "dynamic_topk_ic_zero": "DYNAMIC_TOPK_IC_ZERO",
    "dynamic_topk_low": "DYNAMIC_TOPK_LOW",
    "dynamic_topk_high": "DYNAMIC_TOPK_HIGH",
    "max_portfolio_beta": "MAX_PORTFOLIO_BETA",
    "beta_lookback": "BETA_LOOKBACK",
}


def env_overrides(params: dict) -> dict:
    """从 params 提取应写入 os.environ 的引擎超参（字符串化）。"""
    env = {}
    for key, envname in PARAM_ENV.items():
        if key in params and params[key] is not None:
            env[envname] = str(params[key])
    if "use_impact" in params:
        env["USE_IMPACT_COST"] = "1" if params["use_impact"] else "0"
    if "use_bb_timing" in params:
        env["USE_BB_TIMING"] = "1" if params["use_bb_timing"] else "0"
    if "use_dynamic_topk" in params:
        env["USE_DYNAMIC_TOPK"] = "1" if params["use_dynamic_topk"] else "0"
    if "use_beta_constraint" in params:
        env["USE_BETA_CONSTRAINT"] = "1" if params["use_beta_constraint"] else "0"
    return env


def default_params(mode: str) -> dict:
    return {p["key"]: p["default"] for p in PARAM_SCHEMA.get(mode, [])}
