"""自适应重训触发（决策层）：用跟踪指标决定今晚是否需要重训。

消费 live_tracking 的采集结果，三个条件**任一触发（OR）**：

  1. IC 退化    : 活体 OOS rank-IC 样本充足（≥RETRAIN_IC_MIN_SAMPLES）且
                  尾部 RETRAIN_IC_WINDOW 均值 < RETRAIN_IC_MIN —— 预测质量退化；
  2. 收益落后   : 有真实净值时，尾部 RETRAIN_UNDERPERF_DAYS 实盘收益 <
                  等权基准同窗收益 − RETRAIN_UNDERPERF_MARGIN —— 组合层面失效；
  3. 模型超龄   : train_end_date 距今 > MAX_MODEL_AGE_DAYS 个交易日 ——
                  硬上限，防无限冻结。

每个触发条件输出人类可读原因（进 stdout / 步骤报告 / tracking 快照）。
USE_ADAPTIVE_RETRAIN=0 时调用方应跳过本模块（退回每晚重训旧行为）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def model_age_days(champion_manifest: dict, panel: pd.DataFrame) -> int | None:
    """champion 的年龄：panel 交易日序列里 train_end_date 距最新日的交易日数。

    train_end_date 缺失时回退 promotion.decided_at；都没有 → None（无法判定）。
    """
    if not champion_manifest or not len(panel):
        return None
    ref = champion_manifest.get("train_end_date")
    if not ref:
        ref = (champion_manifest.get("promotion") or {}).get("decided_at")
    if not ref:
        return None
    try:
        ref64 = np.datetime64(str(ref)[:10])
    except ValueError:
        return None
    all_dates = np.sort(panel["date"].unique())
    i = int(np.searchsorted(all_dates, ref64))
    if i >= len(all_dates):
        return 0          # ref 不早于面板最新日 → 年龄 0
    # ref64 非交易日时 i 指向其后第一个交易日，年龄取保守（偏小）值
    return int(max(len(all_dates) - 1 - i, 0))


def should_retrain(champion_manifest: dict | None, ic_stats: dict,
                   nav_stats: dict, panel: pd.DataFrame,
                   force: bool = False) -> tuple[bool, list[str]]:
    """OR 判定：任一条件触发即重训。返回 (是否触发, 原因列表)。"""
    if force:
        return True, ["人工强制重训（--force-retrain）"]
    if not champion_manifest:
        return True, ["无 champion（首次引导 bootstrap）"]

    reasons = []

    # 1) IC 退化
    min_samples = int(getattr(C, "RETRAIN_IC_MIN_SAMPLES", 10))
    ic_min = float(getattr(C, "RETRAIN_IC_MIN", 0.0))
    w = int(getattr(C, "RETRAIN_IC_WINDOW", 20))
    n_ic = int(ic_stats.get("n_ic", 0))
    ic_mean_w = ic_stats.get("ic_mean_w")
    if n_ic >= min_samples and ic_mean_w is not None and ic_mean_w < ic_min:
        reasons.append(f"活体 IC 退化：尾部{w}日 OOS IC 均值 {ic_mean_w:+.4f} "
                       f"< {ic_min:+.4f}（{n_ic} 个成熟样本）")

    # 2) 收益落后（需真实净值）
    margin = float(getattr(C, "RETRAIN_UNDERPERF_MARGIN", 0.03))
    uw = int(getattr(C, "RETRAIN_UNDERPERF_DAYS", 20))
    live_ret = nav_stats.get("live_ret_w")
    bench_ret = nav_stats.get("bench_ret_w")
    if live_ret is not None and bench_ret is not None:
        if live_ret < bench_ret - margin:
            reasons.append(f"实盘收益落后：近{uw}日实盘 {live_ret:+.1%} vs "
                           f"基准 {bench_ret:+.1%}（容忍 {-margin:.1%}）")

    # 3) 模型超龄（硬上限）
    max_age = int(getattr(C, "MAX_MODEL_AGE_DAYS", 60))
    age = model_age_days(champion_manifest, panel)
    if age is not None and age > max_age:
        reasons.append(f"模型超龄：已 {age} 个交易日未重训"
                       f"（硬上限 {max_age} 天）")

    return bool(reasons), reasons
