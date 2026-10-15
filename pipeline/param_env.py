"""会话参数 → 环境变量 的映射（独立轻量模块，**不 import steps_*/config**）。

时序约束：config 在 import 时读取 os.environ，而 `pipeline.registry` import
steps_* 即触发 config。因此 run_session.py 必须先从本模块拿到映射、写好
os.environ，再去 import registry——否则 PARAM_ENV 通道整个是死代码。
registry 从这里 re-export，保持旧调用点（registry.env_overrides）可用。
"""
from __future__ import annotations

# 参数 key -> config 读取的环境变量名（在子进程启动前写入 os.environ）
PARAM_ENV = {
    "train_window": "TRAIN_WINDOW",
    "test_window": "TEST_WINDOW",
    "wf_step": "WF_STEP",
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
