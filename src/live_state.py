"""实盘风险状态持久化与风控缩放计算。

五层风控叠乘链的唯一实现在 src/risk.py（与 backtest_daily 共用）；本模块只做
实盘侧的状态读写与"预计算"（regime 标志 / IV 缩放 / 成熟 IC 入队），保证
回测与实盘数学逐项一致。

状态文件（随 predict_tomorrow 每日运行累积，默认在 outputs/ 下）：
  · live_risk_state.json           —— RiskState（净值/IC 队列/已打分日期）
  · live_risk_state_pred_hist.parquet —— 持久化预测历史（date, symbol, score,
    pred_<model>...），供 IC 降仓（本模块）与 IC 定权（live.py，真 OOS）复用。
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
import risk as _risk


# ---------------------------------------------------------------------------
@dataclass
class RiskState:
    """跨次日运行保留的风控状态。"""
    dates: list[str] = field(default_factory=list)           # ISO 决策日序列
    total_values: list[float] = field(default_factory=list)  # 对应 total_value
    ic_queue: deque = field(default_factory=lambda: deque(maxlen=100))  # 成熟 IC
    scored_ic_dates: list[str] = field(default_factory=list)  # 已打分/已跳过的预测日
    last_update: str | None = None

    def __post_init__(self):
        if not isinstance(self.ic_queue, deque):
            self.ic_queue = deque(self.ic_queue, maxlen=100)

    def to_dict(self) -> dict:
        return {
            "dates": self.dates,
            "total_values": self.total_values,
            "ic_queue": list(self.ic_queue),
            "scored_ic_dates": self.scored_ic_dates,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        return cls(
            dates=list(d.get("dates", [])),
            total_values=list(d.get("total_values", [])),
            ic_queue=deque(d.get("ic_queue", []), maxlen=100),
            scored_ic_dates=list(d.get("scored_ic_dates", [])),
            last_update=d.get("last_update"),
        )


# ---------------------------------------------------------------------------
def load_risk_state(path: str) -> RiskState:
    if not os.path.exists(path):
        return RiskState()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return RiskState.from_dict(json.load(f))
    except Exception as e:
        # 不静默清零：备份损坏文件后从空状态重建，并醒目提示人工核对
        bak = path + ".corrupt-" + pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + ".bak"
        try:
            os.replace(path, bak)
        except OSError:
            bak = ""
        print(f"[警告] 风控状态文件 {path} 损坏（{type(e).__name__}: {e}），"
              f"已从空状态重建" + (f"，原文件已备份到 {bak}，请人工核对。" if bak
                                   else "，备份失败，请尽快人工核查。"))
        return RiskState()


def save_risk_state(path: str, state: RiskState) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 持久化预测历史（pred_hist）：小 parquet，与 risk_state JSON 同目录同行命名
# ---------------------------------------------------------------------------
def pred_hist_path_for(risk_state_path: str) -> str:
    """risk_state JSON 路径 → 同行的 pred_hist parquet 路径。"""
    base, _ = os.path.splitext(risk_state_path)
    return base + "_pred_hist.parquet"


def load_pred_hist(path: str | None) -> pd.DataFrame:
    """读取持久化预测历史（columns: date[str], symbol, score, pred_<model>...）。"""
    empty = pd.DataFrame(columns=["date", "symbol", "score"])
    if not path or not os.path.exists(path):
        return empty
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[警告] 预测历史 {path} 读取失败（{type(e).__name__}: {e}），"
              f"本次按空历史处理（IC 降仓/定权将退化）。")
        return empty


def append_pred_hist(path: str, date, frame: pd.DataFrame,
                     keep_days: int = 120) -> pd.DataFrame:
    """把当日预测分落盘（同日重跑覆盖旧行），并裁剪到最近 keep_days 个预测日。

    frame 需含 symbol/score 列；pred_<model> 列若存在一并持久化（供实盘
    OOS IC 定权复用，见 live.ensemble_predict_latest）。
    """
    cols = ["symbol", "score"] + [c for c in frame.columns
                                  if c.startswith("pred_") and c != "pred_"]
    new = frame[cols].copy()
    new.insert(0, "date", str(pd.Timestamp(date).date()))
    dstr = new["date"].iloc[0]

    hist = load_pred_hist(path)
    if len(hist):
        hist = hist[hist["date"] != dstr]
        # 列对齐（模型腿变化时老文件可能缺 pred_* 列）
        hist = hist.reindex(columns=new.columns)
        hist = pd.concat([hist, new], ignore_index=True)
    else:
        hist = new
    days = sorted(hist["date"].unique())
    if len(days) > keep_days:
        hist = hist[hist["date"].isin(set(days[-keep_days:]))]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hist.to_parquet(path, index=False)
    return hist


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


def collect_matured_ics(panel: pd.DataFrame, pred_hist: pd.DataFrame,
                        d: object, fw: int) -> tuple[dict, list]:
    """计算截至决策日 d 已成熟预测日的 rank-IC（纯函数，不维护状态）。

    防泄露口径：预测日 d_prev 在 d_prev + fw ≤ d（按 panel 交易日序数）时成熟。
    样本门槛：截面 ≥30 只、有效对 ≥20。

    Returns:
        (ics, done)：ics = {预测日 str: rank-IC}（只含成功打分的）；done =
        截至 d 已定局的预测日列表（打分成功或永久性无法打分：滚出面板/样本不足），
        未成熟的预测日不在 done 中（留待未来运行）。
    """
    ics: dict[str, float] = {}
    done: list[str] = []
    if pred_hist is None or not len(pred_hist) or C.TARGET not in panel.columns:
        return ics, done
    all_dates = np.sort(panel["date"].unique())
    if not len(all_dates):
        return ics, done
    d64 = np.datetime64(d)
    di = int(np.searchsorted(all_dates, d64))
    if di >= len(all_dates) or all_dates[di] != d64:
        return ics, done
    tgt_by_date = {dd: sub for dd, sub in panel.groupby("date")}
    for d_prev in sorted(pred_hist["date"].unique()):
        d_prev64 = np.datetime64(d_prev)
        i_prev = int(np.searchsorted(all_dates, d_prev64))
        if i_prev >= len(all_dates) or all_dates[i_prev] != d_prev64:
            done.append(d_prev)   # 预测日已老出当前面板窗口：永远无法打分
            continue
        if i_prev + fw > di:
            continue              # 标签未成熟，留待 fw 天后的运行
        done.append(d_prev)       # 已成熟：打分或（样本不足时）永久跳过
        sub = tgt_by_date.get(d_prev64)
        if sub is None:
            continue
        ph = pred_hist[pred_hist["date"] == d_prev]
        joined = ph.merge(sub[["symbol", C.TARGET]], on="symbol", how="inner")
        if len(joined) < 30:
            continue
        ok = np.isfinite(joined[C.TARGET].to_numpy(float)) & \
            np.isfinite(joined["score"].to_numpy(float))
        if ok.sum() < 20:
            continue
        ic_val = spearmanr(joined["score"].to_numpy(float)[ok],
                           joined[C.TARGET].to_numpy(float)[ok]).correlation
        if ic_val == ic_val:
            ics[d_prev] = float(ic_val)
    return ics, done


def _update_ic_queue(state: RiskState, panel: pd.DataFrame,
                     d: object, fw: int,
                     pred_hist: pd.DataFrame | None = None) -> None:
    """把新成熟的持久化预测日打分算入 IC 队列（实盘 IC 降仓的数据来源）。

    防泄露（与 backtest_daily 同口径）：只对满足 d_prev + fw ≤ d 的预测日打分
    （此时 responder 标签已成熟）；未成熟的留待未来运行。样本量门槛与回测一致
    （截面 ≥30 只、有效对 ≥20）。打分过的预测日记入 state.scored_ic_dates，
    跨运行不重复入队。打分核心在 collect_matured_ics（纯函数，tracking 复用）。
    """
    if pred_hist is None or not len(pred_hist):
        return
    ics, done = collect_matured_ics(panel, pred_hist, d, fw)
    scored = set(state.scored_ic_dates)
    for d_prev in done:
        if d_prev in scored:
            continue
        if d_prev in ics:
            state.ic_queue.append(ics[d_prev])
        state.scored_ic_dates.append(d_prev)


def compute_risk_scale(
    panel: pd.DataFrame,
    decision_date: object,
    state: RiskState | None = None,
    params: dict | None = None,
    pred_hist: pd.DataFrame | None = None,
) -> tuple[float, RiskState]:
    """实盘风控缩放：薄包装 = 状态读写 + 预计算，叠乘链调 src/risk.py 唯一实现。

    Args:
        panel     : 完整历史面板（多日期），用于 regime/IV/成熟 IC 打分。
        pred_hist : 持久化预测历史（load_pred_hist 的结果）；提供后本函数会
                    先把新成熟的预测日打分入 ic_queue（修复原空函数死代码）。

    Returns:
        (risk_scale, updated_state)
    """
    p = dict(params or {})
    state = state or RiskState()

    if bool(p.get("use_ic_scale", C.USE_IC_SCALE)):
        _update_ic_queue(state, panel, decision_date,
                         int(getattr(C, "FORWARD_WINDOW", 20)), pred_hist)

    risk_on = True
    if bool(p.get("use_regime", C.USE_REGIME_FILTER)):
        risk_on = _regime_risk_on(
            panel, decision_date,
            int(p.get("regime_window", C.REGIME_WINDOW)),
            float(p.get("regime_floor_ret", C.REGIME_FLOOR_RET)))
    iv_scale = 1.0
    if bool(p.get("use_iv_scale", C.USE_IV_SCALE)):
        iv_scale = _iv_scale(
            panel, decision_date,
            p.get("iv_feature_for_scale", C.IV_FEATURE_FOR_SCALE),
            int(p.get("iv_scale_window", C.IV_SCALE_WINDOW)),
            float(p.get("iv_min_scale", C.IV_MIN_SCALE)))

    scale, _, _ = _risk.compute_risk_scale(
        state.total_values, state.ic_queue, params=p,
        risk_on=risk_on, iv_scale=iv_scale)
    return scale, state


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
