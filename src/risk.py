"""风控叠乘链的唯一实现 —— 回测(backtest_daily)与实盘(live_state)共用。

五层风控按固定顺序叠乘（本模块是唯一真相源，任何一层都不许在回测/实盘
各自另写实现，改这里两端同时生效）：

  A. vol target ：trailing 实现波动高于目标时 risk_scale = min(1, target/realized)
  B. 回撤刹车   ：回撤超 dd_threshold 再乘 dd_scale
  C. IC 降仓    ：近期【已成熟 OOS】rank-IC 均值线性插值到 [ic_min_scale, 1.0]
  D. regime     ：大盘下行 regime 时乘 regime_min（默认关；risk_on 由调用方判）
  E. IV 降仓    ：高隐含波动时乘 iv_scale（由调用方按面板预计算）

防泄露：vol/dd 只用【已实现】净值历史；IC 只用【已成熟】预测日（调用方保证
d_prev + FORWARD_WINDOW ≤ 决策日才入队）；regime/IV 只用 ≤ 决策日行情。

默认值只从 config 读（params 可显式覆盖），消除历史上 config=0.0 与
backtest_daily getattr 兜底 0.20 的双默认分叉。
"""
from __future__ import annotations

import numpy as np

import config as C


def resolve_risk_params(params: dict | None = None) -> dict:
    """汇总五层风控参数：params 显式覆盖优先，否则唯一来源 config。"""
    p = dict(params or {})
    return {
        "use_vol_target": bool(p.get("use_vol_target", C.USE_VOL_TARGET)),
        "target_vol": float(p.get("target_vol", C.TARGET_VOL)),
        "vol_window": int(p.get("vol_window", C.VOL_WINDOW)),
        "use_dd_brake": bool(p.get("use_dd_brake", C.USE_DD_BRAKE)),
        "dd_threshold": float(p.get("dd_threshold", C.DD_THRESHOLD)),
        "dd_scale": float(p.get("dd_scale", C.DD_SCALE)),
        "use_ic_scale": bool(p.get("use_ic_scale", C.USE_IC_SCALE)),
        "ic_scale_window": int(p.get("ic_scale_window", C.IC_SCALE_WINDOW)),
        "ic_full_scale": float(p.get("ic_full_scale", C.IC_FULL_SCALE)),
        "ic_zero_scale": float(p.get("ic_zero_scale", C.IC_ZERO_SCALE)),
        "ic_min_scale": float(p.get("ic_min_scale", C.IC_MIN_SCALE)),
        "use_regime": bool(p.get("use_regime", C.USE_REGIME_FILTER)),
        "regime_min": float(p.get("regime_min", C.REGIME_MIN)),
        "use_iv_scale": bool(p.get("use_iv_scale", C.USE_IV_SCALE)),
    }


def compute_risk_scale(total_values, ic_queue, *,
                       params: dict | None = None,
                       risk_on: bool = True,
                       iv_scale: float = 1.0) -> tuple[float, float, float | None]:
    """五层风控叠乘链（回测=实盘唯一实现）。

    Args:
        total_values : 历史每日总估值序列（不含今天），供 vol/dd 两层。
                       长度 < 6 时这两层不触发（与回测一致）。
        ic_queue     : 最近【已成熟 OOS】rank-IC 序列；只看尾部 ic_scale_window 个。
        params       : 覆盖 config 的风控参数（见 resolve_risk_params）。
        risk_on      : 决策日是否上行 regime（False → 乘 regime_min）。
        iv_scale     : 决策日 IV 缩放系数（调用方用 ≤ 决策日数据预计算）。

    Returns:
        (risk_scale, ic_scale, roll_ic)。
        roll_ic 为 None 表示 IC 历史不足（此时 ic_scale=1.0，不参与叠乘）。
    """
    rp = resolve_risk_params(params)
    risk_scale = 1.0

    # C. IC 降仓：线性插值 [ic_zero_scale, ic_full_scale] → [ic_min_scale, 1.0]
    ic_scale = 1.0
    roll_ic = None
    if rp["use_ic_scale"]:
        q = [float(x) for x in list(ic_queue)[-rp["ic_scale_window"]:]]
        if len(q) >= max(3, rp["ic_scale_window"] // 4):
            roll_ic = float(np.mean(q))
            t = ((roll_ic - rp["ic_zero_scale"]) /
                 (rp["ic_full_scale"] - rp["ic_zero_scale"] + 1e-9))
            t = max(0.0, min(1.0, t))
            ic_scale = rp["ic_min_scale"] + t * (1.0 - rp["ic_min_scale"])

    # A/B. vol target（min 截断）+ 回撤刹车（叠乘）
    tvs = np.asarray(list(total_values), dtype=np.float64)
    if (rp["use_vol_target"] or rp["use_dd_brake"]) and len(tvs) >= 6:
        rets = tvs[1:] / tvs[:-1] - 1.0
        if rp["use_vol_target"] and len(rets) >= 5:
            rv = float(np.std(rets[-rp["vol_window"]:])) + 1e-12
            target_daily = rp["target_vol"] / np.sqrt(252.0)
            risk_scale = min(1.0, target_daily / rv)
        if rp["use_dd_brake"]:
            dd = tvs[-1] / tvs.max() - 1.0
            if dd < -rp["dd_threshold"]:
                risk_scale *= rp["dd_scale"]

    # 按固定顺序叠乘：vol/dd → IC → regime → IV
    risk_scale *= ic_scale
    if rp["use_regime"] and not risk_on:
        risk_scale *= rp["regime_min"]
    if rp["use_iv_scale"]:
        risk_scale *= float(iv_scale)
    return float(risk_scale), float(ic_scale), roll_ic
