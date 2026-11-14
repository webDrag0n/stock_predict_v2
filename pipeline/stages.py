"""端到端量化流水线的阶段注册表（可视化流水线页面的后端元数据）。

与 registry.py 的分工：registry 描述「会话模式 → 步骤序列 + 参数 schema」
（执行视角）；本模块描述「用户工作流 → 有序阶段」（操作视角）。每个阶段声明：

  id        稳定标识（前端 pattern-matching id 用）
  title/en  显示名（中英）
  page      该阶段的主工作台路由（「打开工作台」链接）
  mode      该阶段可一键发起的会话模式（registry.STEPS 的键）；None=不可发起
  param_keys 面板里暴露的可编辑参数（registry.PARAM_SCHEMA[mode] 的键子集，
            顺序即渲染顺序；空列表=该阶段无可调参数，只做状态展示）
  desc      一句话职责说明
  status_hint 状态灯的含义说明（悬停提示）

本模块不做任何 IO：状态灯的实际取值由 webapp 页面用 dataio 现算，保证
pipeline/ 不反向依赖 webapp/。新增/调整阶段只改 PIPELINE_STAGES。
"""

from __future__ import annotations

PIPELINE_STAGES: list[dict] = [
    {
        "id": "data",
        "title": "数据",
        "en": "Data",
        "page": "/market",
        "mode": None,
        "param_keys": [],
        "desc": "全市场日线与因子面板缓存（daily_run.sh 每日更新）",
        "status_hint": "最新行情缓存日期；落后最新交易日说明该跑 daily_run.sh",
    },
    {
        "id": "factor",
        "title": "因子研发",
        "en": "Factors",
        "page": "/factors",
        "mode": "factoreval",
        "param_keys": ["factor_subset", "lookback_days", "n_layers", "with_diagnostics"],
        "desc": "因子库编辑、单因子/全库 RankIC 评估、分层回测",
        "status_hint": "启用因子数与最近一次全库评估时间",
    },
    {
        "id": "train",
        "title": "模型训练",
        "en": "Train",
        "page": "/models",
        "mode": "train",
        "param_keys": [
            "ensemble_models",
            "n_seeds",
            "train_window",
            "rebuild_days",
            "incr_trees",
            "ic_window",
            "eval_days",
        ],
        "desc": "按配方训练候选模型，门控双回放后落 staging",
        "status_hint": "staging 中待晋升的候选数量",
    },
    {
        "id": "validate",
        "title": "回测验证",
        "en": "Validate",
        "page": "/models",
        "mode": "walkforward",
        "param_keys": ["train_window", "test_window", "wf_step", "strategy"],
        "desc": "滚动 Walk-Forward 全市场 OOS 回测，验证配方可信度",
        "status_hint": "最近一次 walkforward 的 OOS Sharpe 与生成时间",
    },
    {
        "id": "gate",
        "title": "门控晋升",
        "en": "Gate",
        "page": "/models",
        "mode": None,
        "param_keys": [],
        "desc": "候选 vs 冠军双回放对比，达标才允许替换线上 champion",
        "status_hint": "当前 champion 年龄；超龄（>84 自然日）亮黄灯",
    },
    {
        "id": "trade",
        "title": "交易执行",
        "en": "Trade",
        "page": "/",
        "mode": "nextday",
        "param_keys": [
            "capital",
            "topk",
            "max_positions",
            "max_weight",
            "use_bb_timing",
            "order_price_mode",
            "min_notional",
        ],
        "desc": "champion 出信号、生成次日订单、执行与对账",
        "status_hint": "最新订单日期；与最新数据日期一致说明今日已出单",
    },
]


def stage_ids() -> list[str]:
    return [s["id"] for s in PIPELINE_STAGES]


def stage(stage_id: str) -> dict | None:
    return next((s for s in PIPELINE_STAGES if s["id"] == stage_id), None)


def stage_param_schema(stage_id: str) -> list[dict]:
    """该阶段面板要渲染的参数 schema（按 param_keys 顺序从 registry.PARAM_SCHEMA 抽取）。

    阶段无 mode 或键缺失时跳过；返回 [] 表示该阶段只做状态展示。
    """
    from . import registry

    st = stage(stage_id)
    if not st or not st.get("mode"):
        return []
    full = {p["key"]: p for p in registry.PARAM_SCHEMA.get(st["mode"], [])}
    return [full[k] for k in st["param_keys"] if k in full]
