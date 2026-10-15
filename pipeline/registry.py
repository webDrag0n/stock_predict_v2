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
from . import steps_walkforward as W
from . import steps_robustness as R
# PARAM_ENV / env_overrides 定义在 pipeline.param_env（不触发 config import），
# 这里 re-export 保持旧调用点可用
from .param_env import PARAM_ENV, env_overrides  # noqa: F401

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
    "walkforward": [
        ("wf_data_import", "数据导入", W.wf_data_import),
        ("wf_run", "滚动 Walk-Forward", W.wf_run),
        ("wf_report", "OOS 绩效报告", W.wf_report),
    ],
    "robustness": [
        ("ms_run", "多 Seed 回测", R.ms_run),
        ("ms_report", "稳健性报告", R.ms_report),
    ],
}

MODE_LABELS = {"backtest": "回测", "nextday": "次日交易",
               "walkforward": "滚动Walk-Forward", "robustness": "多Seed稳健性"}

# --- 参数 schema（前端按此渲染表单，type ∈ number|select|bool|text）----------
# backtest 与 nextday 大量参数重复：共用定义收敛在 _SHARED，两模式只在
# label/hint 等文案上有差异，组装时用 _sp() 覆盖。改动共用参数只动一处。
_SHARED: dict[str, dict] = {
    "capital": {"key": "capital", "type": "number",
                "default": 1000000, "min": 10000, "max": 1000000000, "step": 10000},
    "initial_holdings": {"key": "initial_holdings", "type": "textarea",
                         "default": "", "placeholder": "600519.SH:200\n000001.SZ:1000",
                         "hint": "留空=空仓"},
    "topk": {"key": "topk", "label": "候选池比例 TOPK", "type": "number",
             "default": 0.15, "min": 0.02, "max": 0.50, "step": 0.01},
    "max_positions": {"key": "max_positions", "label": "最大持仓数", "type": "number",
                      "default": 30, "min": 1, "max": 200, "step": 1},
    "max_weight": {"key": "max_weight", "label": "单股权重上限", "type": "number",
                   "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01},
    "use_bb_timing": {"key": "use_bb_timing", "label": "启用布林/RSI择时", "type": "bool",
                      "default": True},
    "order_price_mode": {"key": "order_price_mode", "label": "挂单价模型", "type": "select",
                         "default": "limit_bb",
                         "options": ["limit_bb", "next_open"]},
    "min_notional": {"key": "min_notional", "label": "最小订单金额(元)", "type": "number",
                     "default": 30000, "min": 0, "max": 1000000, "step": 1000},
    "use_dynamic_topk": {"key": "use_dynamic_topk", "label": "启用动态TOPK", "type": "bool",
                         "default": False},
    "dynamic_topk_ic_full": {"key": "dynamic_topk_ic_full", "label": "动态TOPK满仓IC阈值",
                             "type": "number", "default": 0.02,
                             "min": -0.05, "max": 0.10, "step": 0.005},
    "dynamic_topk_ic_zero": {"key": "dynamic_topk_ic_zero", "label": "动态TOPK低仓IC阈值",
                             "type": "number", "default": -0.01,
                             "min": -0.10, "max": 0.05, "step": 0.005},
    "dynamic_topk_low": {"key": "dynamic_topk_low", "label": "动态TOPK最小值",
                         "type": "number",
                         "default": 0.10, "min": 0.02, "max": 0.50, "step": 0.01},
    "dynamic_topk_high": {"key": "dynamic_topk_high", "label": "动态TOPK最大值",
                          "type": "number",
                          "default": 0.25, "min": 0.02, "max": 0.80, "step": 0.01},
    "use_beta_constraint": {"key": "use_beta_constraint", "label": "启用组合Beta约束",
                            "type": "bool", "default": False},
    "max_portfolio_beta": {"key": "max_portfolio_beta", "label": "最大组合Beta",
                           "type": "number",
                           "default": 1.10, "min": 0.50, "max": 2.00, "step": 0.05},
    "n_seeds": {"key": "n_seeds", "label": "模型 seed 数", "type": "number",
                "default": 1, "min": 1, "max": 5, "step": 1},
    "train_window": {"key": "train_window", "label": "训练窗口(交易日)", "type": "number",
                     "default": 500, "min": 120, "max": 1500, "step": 30},
    "rebuild_days": {"key": "rebuild_days", "label": "全量重建周期(交易日)", "type": "number",
                     "default": 60, "min": 10, "max": 250, "step": 10},
    "incr_trees": {"key": "incr_trees", "label": "增量续训树数", "type": "number",
                   "default": 60, "min": 10, "max": 300, "step": 10},
    "ic_window": {"key": "ic_window", "label": "滚动IC定权窗口", "type": "number",
                  "default": 20, "min": 5, "max": 120, "step": 5},
}


def _sp(key: str, **over) -> dict:
    """取共用参数定义的副本，按模式覆盖 label/hint 等差异字段。"""
    d = dict(_SHARED[key])
    d.update(over)
    return d


PARAM_SCHEMA = {
    "backtest": [
        # --- 账户初始状态 ---
        _sp("capital", label="初始资金(元)"),
        _sp("initial_holdings", label="初始持仓(每行 代码:股数)"),
        {"key": "start_date", "label": "回测开始日(YYYY-MM-DD)", "type": "text",
         "default": "", "hint": "留空=最近 3 个月 (今天 - 92 天)"},
        {"key": "end_date", "label": "回测结束日(YYYY-MM-DD)", "type": "text",
         "default": "", "hint": "留空=到最新"},
        # --- 选股 / 择时 ---
        _sp("topk"),
        _sp("max_positions"),
        _sp("max_weight"),
        _sp("use_bb_timing", hint="关=仅按ML排序选股"),
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
        _sp("order_price_mode",
            hint="limit_bb=布林下轨挂限价(换手/成本最低,扫描最优);next_open=次日开盘市价"),
        _sp("min_notional", hint="越大换手越低;扫描显示≥3万显著降成本"),
        # --- 动态集中度(基于滚动IC) ---
        _sp("use_dynamic_topk", hint="IC高时集中,IC低时分散;默认关,需实验验证"),
        _sp("dynamic_topk_ic_full", hint="滚动IC≥此值时使用最小topk"),
        _sp("dynamic_topk_ic_zero", hint="滚动IC≤此值时使用最大topk"),
        _sp("dynamic_topk_low"),
        _sp("dynamic_topk_high"),
        _sp("use_beta_constraint",
            hint="用return_1d估算 trailing beta, 高beta替换为低beta候选"),
        _sp("max_portfolio_beta"),
        {"key": "beta_lookback", "label": "Beta估算窗口(日)", "type": "number",
         "default": 60, "min": 20, "max": 252, "step": 5},
        # --- 增量集成 ---
        _sp("train_window"),
        _sp("rebuild_days", hint="其余日做增量续训"),
        _sp("incr_trees"),
        _sp("ic_window"),
        _sp("n_seeds", hint="越多越稳但越慢"),
        {"key": "use_impact", "label": "计入市场冲击成本", "type": "bool",
         "default": True},
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
    "walkforward": [
        _sp("capital", label="初始资金(元)", hint="仅用于净值图的金额口径"),
        _sp("train_window"),
        {"key": "test_window", "label": "测试窗口(交易日)", "type": "number",
         "default": 60, "min": 10, "max": 250, "step": 10},
        {"key": "wf_step", "label": "窗口滚动步长(交易日)", "type": "number",
         "default": 60, "min": 10, "max": 250, "step": 10,
         "hint": "默认=测试窗口（OOS 窗口互不重叠）"},
        {"key": "select_features", "label": "窗口内因子筛选", "type": "bool",
         "default": False, "hint": "每窗口仅用其训练段做因子准入（隔离，无泄露）"},
        {"key": "strategy", "label": "持仓策略", "type": "select",
         "default": "long_topk",
         "options": ["long_topk", "long_short_topk", "tanh", "cross_sectional"]},
    ],
    "robustness": [
        {"key": "seeds", "label": "随机种子列表", "type": "text",
         "default": "42,123,7",
         "hint": "逗号分隔；有 data_cache/seed_oos_<seed>.parquet 缓存则秒回"},
        _sp("capital", label="初始资金(元)", hint="仅用于净值/成本的金额口径"),
        _sp("train_window", hint="仅 seed 缓存未命中、需重训时生效"),
    ],
}


def default_params(mode: str) -> dict:
    return {p["key"]: p["default"] for p in PARAM_SCHEMA.get(mode, [])}
