"""实盘风险状态持久化与风控缩放计算。

把 backtest_daily.py 里每日计算的 vol/dd/IC/regime/IV 五层风控抽象成纯函数，
供 live 路径复用。状态通过 JSON 持久化到磁盘，跨 predict_tomorrow 运行保留。
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C


# ---------------------------------------------------------------------------
@dataclass
class RiskState:
    """跨次日运行保留的风控状态。"""
    dates: list[str] = field(default_factory=list)           # ISO 决策日序列
    total_values: list[float] = field(default_factory=list)  # 对应 total_value
    ic_queue: deque = field(default_factory=lambda: deque(maxlen=100))  # 成熟 IC
    last_update: str | None = None

    def __post_init__(self):
        if not isinstance(self.ic_queue, deque):
            self.ic_queue = deque(self.ic_queue, maxlen=100)

    def to_dict(self) -> dict:
        return {
            "dates": self.dates,
            "total_values": self.total_values,
            "ic_queue": list(self.ic_queue),
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        return cls(
            dates=list(d.get("dates", [])),
            total_values=list(d.get("total_values", [])),
            ic_queue=deque(d.get("ic_queue", []), maxlen=100),
            last_update=d.get("last_update"),
        )


# ---------------------------------------------------------------------------
def load_risk_state(path: str) -> RiskState:
    if not os.path.exists(path):
        return RiskState()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return RiskState.from_dict(json.load(f))
    except Exception:
        return RiskState()


def save_risk_state(path: str, state: RiskState) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
def _date_to_str(d: object) -> str:
    return str(pd.Timestamp(d).date())


def _regime_risk_on(panel: pd.DataFrame, d: object,
                    regime_window: int, regime_floor_ret: float) -> bool:
    """判断决策日 d 是否处于大盘下行 regime（仅 ≤d 行情）。返回 True=上行/无风险，False=下行。"""
    if "return_1d" not in panel.columns and "pct_chg" not in panel.columns:
        return True
    _rc = "return_1d" if "return_1d" in panel.columns else "pct_chg"
    mkt = panel.groupby("date")[_rc].mean().sort_index()
    if _rc == "pct_chg":
        mkt = mkt / 100.0
    mkt_dates = mkt.index.to_numpy()
    mkt_v = mkt.to_numpy(dtype=np.float64)
    idx = int(np.searchsorted(mkt_dates, np.datetime64(d), side="right")) - 1
    if idx < 0:
        return True
    lo = max(0, idx - regime_window + 1)
    trail = float(np.prod(1.0 + mkt_v[lo:idx + 1]) - 1.0)
    return trail > regime_floor_ret


def _iv_scale(panel: pd.DataFrame, d: object,
              iv_feature: str, iv_scale_window: int,
              iv_min_scale: float) -> float:
    if iv_feature not in panel.columns:
        return 1.0
    iv_series = panel.groupby("date")[iv_feature].first().sort_index()
    iv_dates = iv_series.index.to_numpy()
    iv_v = iv_series.to_numpy(dtype=np.float64)
    idx = int(np.searchsorted(iv_dates, np.datetime64(d), side="right")) - 1
    if idx < 0 or not np.isfinite(iv_v[idx]):
        return 1.0
    lo = max(0, idx - iv_scale_window + 1)
    trail = iv_v[lo:idx]
    if len(trail) < iv_scale_window // 3:
        return 1.0
    lo_q, hi_q = np.nanpercentile(trail, [33.0, 66.0])
    v = iv_v[idx]
    if v >= hi_q:
        return iv_min_scale
    if v <= lo_q:
        return 1.0
    t = (v - lo_q) / (hi_q - lo_q + 1e-12)
    return 1.0 + t * (iv_min_scale - 1.0)


def _update_ic_queue(state: RiskState, panel: pd.DataFrame,
                     d: object, fw: int) -> None:
    """把新成熟的预测日打分算入 IC 队列。panel 需含 date, symbol, TARGET。"""
    if C.TARGET not in panel.columns:
        return
    all_dates = np.sort(panel["date"].unique())
    di = int(np.searchsorted(all_dates, np.datetime64(d), side="right")) - 1
    if di < 0:
        return
    # 找那些预测日 + fw 已成熟的预测历史（这里简化为：若 state 里存了 pred_hist）
    # 实际上 live 端不保存 pred_hist，此函数在 live 主要依赖已持久化的 ic_queue。
    # 为 future-proof：如果后续把 pred_hist 也持久化，可在这里计算。


def compute_risk_scale(
    panel: pd.DataFrame,
    decision_date: object,
    state: RiskState | None = None,
    params: dict | None = None,
) -> tuple[float, RiskState]:
    """计算与 backtest_daily 一致的风控缩放系数，并更新 RiskState。

    Returns:
        (risk_scale, updated_state)
    """
    p = dict(params or {})
    state = state or RiskState()

    use_vol_target = bool(p.get("use_vol_target", C.USE_VOL_TARGET))
    target_vol = float(p.get("target_vol", C.TARGET_VOL))
    vol_window = int(p.get("vol_window", C.VOL_WINDOW))
    use_dd_brake = bool(p.get("use_dd_brake", C.USE_DD_BRAKE))
    dd_threshold = float(p.get("dd_threshold", C.DD_THRESHOLD))
    dd_scale = float(p.get("dd_scale", C.DD_SCALE))
    use_ic_scale = bool(p.get("use_ic_scale", C.USE_IC_SCALE))
    ic_zero_scale = float(p.get("ic_zero_scale", C.IC_ZERO_SCALE))
    ic_full_scale = float(p.get("ic_full_scale", C.IC_FULL_SCALE))
    ic_min_scale = float(p.get("ic_min_scale", C.IC_MIN_SCALE))
    use_regime = bool(p.get("use_regime", C.USE_REGIME_FILTER))
    regime_window = int(p.get("regime_window", C.REGIME_WINDOW))
    regime_floor_ret = float(p.get("regime_floor_ret", C.REGIME_FLOOR_RET))
    regime_min = float(p.get("regime_min", C.REGIME_MIN))
    use_iv_scale = bool(p.get("use_iv_scale", C.USE_IV_SCALE))
    iv_scale_window = int(p.get("iv_scale_window", C.IV_SCALE_WINDOW))
    iv_feature = p.get("iv_feature_for_scale", C.IV_FEATURE_FOR_SCALE)
    iv_min_scale = float(p.get("iv_min_scale", C.IV_MIN_SCALE))

    risk_scale = 1.0

    # IC 信号失效降仓
    ic_scale = 1.0
    if use_ic_scale and len(state.ic_queue) >= 3:
        roll_ic = float(np.mean(state.ic_queue))
        t = ((roll_ic - ic_zero_scale) /
             (ic_full_scale - ic_zero_scale + 1e-9))
        t = max(0.0, min(1.0, t))
        ic_scale = ic_min_scale + t * (1.0 - ic_min_scale)
    risk_scale *= ic_scale

    # vol target + dd brake
    if (use_vol_target or use_dd_brake) and len(state.total_values) >= 6:
        tvs = np.array(state.total_values, dtype=np.float64)
        rets = tvs[1:] / tvs[:-1] - 1.0
        if use_vol_target and len(rets) >= 5:
            rv = float(np.std(rets[-vol_window:])) + 1e-12
            target_daily = target_vol / np.sqrt(252.0)
            risk_scale = min(risk_scale, target_daily / rv)
        if use_dd_brake:
            dd = tvs[-1] / tvs.max() - 1.0
            if dd < -dd_threshold:
                risk_scale *= dd_scale

    # regime 择时
    if use_regime and not _regime_risk_on(
        panel, decision_date, regime_window, regime_floor_ret
    ):
        risk_scale *= regime_min

    # IV 缩放
    if use_iv_scale:
        risk_scale *= _iv_scale(
            panel, decision_date, iv_feature, iv_scale_window, iv_min_scale
        )

    return float(risk_scale), state


def update_risk_state(
    state: RiskState,
    decision_date: object,
    total_value: float,
    new_ic: float | None = None,
) -> RiskState:
    """每次 live 运行后追加最新状态。"""
    dstr = _date_to_str(decision_date)
    if state.dates and state.dates[-1] == dstr:
        # 覆盖同日（重跑）
        state.total_values[-1] = float(total_value)
    else:
        state.dates.append(dstr)
        state.total_values.append(float(total_value))
    # 只保留最近 252 日
    if len(state.dates) > 252:
        state.dates = state.dates[-252:]
        state.total_values = state.total_values[-252:]
    if new_ic is not None and new_ic == new_ic:
        state.ic_queue.append(float(new_ic))
    state.last_update = dstr
    return state
