"""模式 → 步骤序列 + 可调参数 schema 的注册表。

run_session.py 按 STEPS[mode] 顺序执行步骤；前端按 PARAM_SCHEMA[mode]
动态渲染创建会话的参数表单。新增模式/步骤只改这里。

模式分两类（MODE_CATEGORY）：
  research   研究/验证类（模型实验室「新建任务」面板）：walkforward 是唯一
             回测方法；train 训练候选 → staging；modeleval 信号评估；
             factoreval 因子库评估。
  production 生产类（交易台）：nextday 次日交易出单。

参数同时被写入 os.environ（PARAM_ENV 映射）→ config 在子进程启动时读取，
从而无侵入地覆盖 TOPK_RATIO / TRAIN_WINDOW / 增量集成等引擎超参；其余参数
（资金/初始持仓/日期区间/择时阈值）由步骤函数从 ctx.params 直接读取。
"""

from __future__ import annotations

from . import steps_factoreval as FE
from . import steps_modeleval as M
from . import steps_nextday as N
from . import steps_train as T
from . import steps_walkforward as W

# PARAM_ENV / env_overrides 定义在 pipeline.param_env（不触发 config import），
# 这里 re-export 保持旧调用点可用
from .param_env import PARAM_ENV, env_overrides  # noqa: F401

# mode -> [(step_id, title, fn)...]
STEPS = {
    "walkforward": [
        ("wf_data_import", "数据导入", W.wf_data_import),
        ("wf_run", "滚动 Walk-Forward", W.wf_run),
        ("wf_report", "OOS 绩效报告", W.wf_report),
    ],
    "train": [
        ("data_import", "数据导入", T.data_import),
        ("train_candidate", "训练候选模型", T.train_candidate),
        ("gate_eval", "门控评估（→staging）", T.gate_eval),
    ],
    "modeleval": [
        ("data_import", "数据导入", M.data_import),
        ("signal_eval", "信号评估", M.signal_eval),
        ("eval_report", "信号评估报告", M.eval_report),
    ],
    "factoreval": [
        ("data_import", "数据导入", FE.data_import),
        ("eval_factors", "因子评估", FE.eval_factors),
        ("eval_report", "评估报告", FE.eval_report),
    ],
    "nextday": [
        ("data_import", "数据导入（实盘）", N.data_import),
        ("train_predict", "训练 + 预测", N.train_predict),
        ("generate_orders", "生成次日订单", N.generate_orders),
    ],
}

MODE_LABELS = {
    "walkforward": "滚动Walk-Forward",
    "train": "训练候选模型",
    "modeleval": "信号评估",
    "factoreval": "因子库评估",
    "nextday": "次日交易",
    # 历史模式（已从创建入口移除，仅为旧会话的显示保留标签）
    "backtest": "回测",
    "robustness": "多Seed稳健性",
}

# research = 模型实验室「新建任务」；production = 交易台
MODE_CATEGORY = {
    "walkforward": "research",
    "train": "research",
    "modeleval": "research",
    "factoreval": "research",
    "nextday": "production",
}

# --- 参数 schema（前端按此渲染表单，type ∈ number|select|bool|text）----------
# 多模式共用参数收敛在 _SHARED，组装时用 _sp() 覆盖 label/hint 等文案差异。
_SHARED: dict[str, dict] = {
    "capital": {
        "key": "capital",
        "type": "number",
        "default": 1000000,
        "min": 10000,
        "max": 1000000000,
        "step": 10000,
    },
    "initial_holdings": {
        "key": "initial_holdings",
        "type": "textarea",
        "default": "",
        "placeholder": "600519.SH:200\n000001.SZ:1000",
        "hint": "留空=空仓",
    },
    "topk": {
        "key": "topk",
        "label": "候选池比例 TOPK",
        "type": "number",
        "default": 0.15,
        "min": 0.02,
        "max": 0.50,
        "step": 0.01,
    },
    "max_positions": {
        "key": "max_positions",
        "label": "最大持仓数",
        "type": "number",
        "default": 30,
        "min": 1,
        "max": 200,
        "step": 1,
    },
    "max_weight": {
        "key": "max_weight",
        "label": "单股权重上限",
        "type": "number",
        "default": 0.10,
        "min": 0.01,
        "max": 1.0,
        "step": 0.01,
    },
    "use_bb_timing": {
        "key": "use_bb_timing",
        "label": "启用布林/RSI择时",
        "type": "bool",
        "default": True,
    },
    "order_price_mode": {
        "key": "order_price_mode",
        "label": "挂单价模型",
        "type": "select",
        "default": "limit_bb",
        "options": ["limit_bb", "next_open"],
    },
    "min_notional": {
        "key": "min_notional",
        "label": "最小订单金额(元)",
        "type": "number",
        "default": 30000,
        "min": 0,
        "max": 1000000,
        "step": 1000,
    },
    "use_dynamic_topk": {
        "key": "use_dynamic_topk",
        "label": "启用动态TOPK",
        "type": "bool",
        "default": False,
    },
    "dynamic_topk_ic_full": {
        "key": "dynamic_topk_ic_full",
        "label": "动态TOPK满仓IC阈值",
        "type": "number",
        "default": 0.02,
        "min": -0.05,
        "max": 0.10,
        "step": 0.005,
    },
    "dynamic_topk_ic_zero": {
        "key": "dynamic_topk_ic_zero",
        "label": "动态TOPK低仓IC阈值",
        "type": "number",
        "default": -0.01,
        "min": -0.10,
        "max": 0.05,
        "step": 0.005,
    },
    "dynamic_topk_low": {
        "key": "dynamic_topk_low",
        "label": "动态TOPK最小值",
        "type": "number",
        "default": 0.10,
        "min": 0.02,
        "max": 0.50,
        "step": 0.01,
    },
    "dynamic_topk_high": {
        "key": "dynamic_topk_high",
        "label": "动态TOPK最大值",
        "type": "number",
        "default": 0.25,
        "min": 0.02,
        "max": 0.80,
        "step": 0.01,
    },
    "use_beta_constraint": {
        "key": "use_beta_constraint",
        "label": "启用组合Beta约束",
        "type": "bool",
        "default": False,
    },
    "max_portfolio_beta": {
        "key": "max_portfolio_beta",
        "label": "最大组合Beta",
        "type": "number",
        "default": 1.10,
        "min": 0.50,
        "max": 2.00,
        "step": 0.05,
    },
    "n_seeds": {
        "key": "n_seeds",
        "label": "模型 seed 数",
        "type": "number",
        "default": 1,
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "train_window": {
        "key": "train_window",
        "label": "训练窗口(交易日)",
        "type": "number",
        "default": 500,
        "min": 120,
        "max": 1500,
        "step": 30,
    },
    "rebuild_days": {
        "key": "rebuild_days",
        "label": "全量重建周期(交易日)",
        "type": "number",
        "default": 60,
        "min": 10,
        "max": 250,
        "step": 10,
    },
    "incr_trees": {
        "key": "incr_trees",
        "label": "增量续训树数",
        "type": "number",
        "default": 60,
        "min": 10,
        "max": 300,
        "step": 10,
    },
    "ic_window": {
        "key": "ic_window",
        "label": "滚动IC定权窗口",
        "type": "number",
        "default": 20,
        "min": 5,
        "max": 120,
        "step": 5,
    },
}


def _sp(key: str, **over) -> dict:
    """取共用参数定义的副本，按模式覆盖 label/hint 等差异字段。"""
    d = dict(_SHARED[key])
    d.update(over)
    return d


PARAM_SCHEMA = {
    "walkforward": [
        _sp("capital", label="初始资金(元)", hint="仅用于净值图的金额口径"),
        _sp("train_window"),
        {
            "key": "test_window",
            "label": "测试窗口(交易日)",
            "type": "number",
            "default": 60,
            "min": 10,
            "max": 250,
            "step": 10,
        },
        {
            "key": "wf_step",
            "label": "窗口滚动步长(交易日)",
            "type": "number",
            "default": 60,
            "min": 10,
            "max": 250,
            "step": 10,
            "hint": "默认=测试窗口（OOS 窗口互不重叠）",
        },
        {
            "key": "select_features",
            "label": "窗口内因子筛选",
            "type": "bool",
            "default": False,
            "hint": "每窗口仅用其训练段做因子准入（隔离，无泄露）",
        },
        {
            "key": "strategy",
            "label": "持仓策略",
            "type": "select",
            "default": "long_topk",
            "options": ["long_topk", "long_short_topk", "tanh", "cross_sectional"],
        },
    ],
    "train": [
        {
            "key": "ensemble_models",
            "label": "集成子模型",
            "type": "text",
            "default": "",
            "hint": "逗号分隔的内部模型键(如 gbm_l2,ic_linear)；留空=config 默认",
        },
        _sp("n_seeds", hint="越多越稳但越慢"),
        _sp("train_window"),
        _sp("rebuild_days", hint="其余日做增量续训"),
        _sp("incr_trees"),
        _sp("ic_window"),
        {
            "key": "eval_days",
            "label": "门控评估窗口(交易日)",
            "type": "number",
            "default": 60,
            "min": 20,
            "max": 250,
            "step": 10,
            "hint": "候选 vs champion 双方重放对比的区间",
        },
        # --- 门控双回放的决策口径（与 nextday 下单同参数） ---
        _sp("topk", hint="门控重放口径，与正式下单一致"),
        _sp("max_positions"),
        _sp("max_weight"),
        _sp("order_price_mode"),
        _sp("min_notional"),
    ],
    "modeleval": [
        {
            "key": "model_ref",
            "label": "评估对象",
            "type": "text",
            "default": "rolling",
            "hint": "rolling=滚动重训；champion=冻结冠军工件；staging:<id>=冻结候选",
        },
        {
            "key": "lookback_days",
            "label": "评估区间(交易日)",
            "type": "number",
            "default": 120,
            "min": 20,
            "max": 1500,
            "step": 10,
            "hint": "rolling 路径另需 train_window 训练历史",
        },
        _sp("train_window", hint="仅 rolling 路径生效"),
        {
            "key": "test_window",
            "label": "测试窗口(交易日)",
            "type": "number",
            "default": 60,
            "min": 10,
            "max": 250,
            "step": 10,
            "hint": "仅 rolling 路径生效",
        },
        {
            "key": "wf_step",
            "label": "窗口滚动步长(交易日)",
            "type": "number",
            "default": 60,
            "min": 10,
            "max": 250,
            "step": 10,
            "hint": "仅 rolling 路径生效",
        },
        {
            "key": "n_quantiles",
            "label": "分位数分层数",
            "type": "number",
            "default": 5,
            "min": 3,
            "max": 10,
            "step": 1,
        },
    ],
    "factoreval": [
        {
            "key": "factor_subset",
            "label": "因子子集",
            "type": "text",
            "default": "",
            "hint": "逗号分隔因子 id；留空=全库启用因子（覆盖写 factor_eval.csv）",
        },
        {
            "key": "lookback_days",
            "label": "回看天数",
            "type": "number",
            "default": 1825,
            "min": 120,
            "max": 3650,
            "step": 30,
            "hint": "与起止日期二选一；显式日期优先",
        },
        {
            "key": "start_date",
            "label": "面板起始日(YYYY-MM-DD)",
            "type": "text",
            "default": "",
            "hint": "留空=按回看天数推",
        },
        {
            "key": "end_date",
            "label": "面板结束日(YYYY-MM-DD)",
            "type": "text",
            "default": "",
            "hint": "留空=到今天",
        },
        {
            "key": "n_layers",
            "label": "分层回测层数",
            "type": "number",
            "default": 5,
            "min": 3,
            "max": 10,
            "step": 1,
        },
        {
            "key": "dump_layers",
            "label": "落盘分层净值/逐日RankIC",
            "type": "bool",
            "default": True,
            "hint": "准入通过因子 → factor_layer_/factor_ic_daily_ parquet",
        },
        {
            "key": "with_diagnostics",
            "label": "追加衰减/相关矩阵诊断",
            "type": "bool",
            "default": False,
            "hint": "耗时；产出 diag_factor_ic_decay.csv / diag_factor_corr.csv",
        },
        {
            "key": "diag_with_model",
            "label": "诊断含模型对比",
            "type": "bool",
            "default": False,
            "hint": "更耗时；诊断3「模型 vs 最强单因子」",
        },
    ],
    "nextday": [
        _sp("capital", label="组合资金(元)"),
        _sp("initial_holdings", label="当前持仓(每行 代码:股数)"),
        _sp("topk"),
        _sp("max_positions"),
        _sp("max_weight"),
        _sp("use_bb_timing"),
        _sp("order_price_mode", hint="limit_bb=布林下轨挂限价(换手/成本最低)"),
        _sp("min_notional"),
        # --- 动态集中度(基于滚动IC) ---
        _sp("use_dynamic_topk", hint="IC高时集中,IC低时分散;默认关"),
        _sp("dynamic_topk_ic_full"),
        _sp("dynamic_topk_ic_zero"),
        _sp("dynamic_topk_low"),
        _sp("dynamic_topk_high"),
        _sp("use_beta_constraint", hint="高beta替换为低beta候选"),
        _sp("max_portfolio_beta"),
        _sp("n_seeds"),
        _sp("train_window", hint="越小越快，建议实盘 250~500"),
        _sp("rebuild_days"),
        _sp("incr_trees"),
        _sp("ic_window"),
    ],
}


def default_params(mode: str) -> dict:
    return {p["key"]: p["default"] for p in PARAM_SCHEMA.get(mode, [])}
