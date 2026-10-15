"""回测=实盘风控一致性 + 各层手算数值断言（阶段 2 修复的防线）。

覆盖：
  · 同一 state/参数下，backtest_daily 内联调用路径（risk.compute_risk_scale）
    与 live_state 薄包装路径的 risk_scale 逐项相等；
  · vol target / 回撤刹车 / IC 降仓的手算数值断言；
  · 实盘 IC 队列从持久化 pred_hist 成熟打分（原 _update_ic_queue 空函数死代码）；
  · risk_state JSON 损坏时备份而非静默清零；
  · simulate_fills 跳过 blocked 订单（与实盘 --commit 一致）；
  · live 检测 USE_DYNAMIC_TOPK=1 显式报错。
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import config as C
import risk
from live_state import (RiskState, load_risk_state, save_risk_state,
                        compute_risk_scale as live_compute_risk_scale,
                        _update_ic_queue, _regime_risk_on, _iv_scale,
                        pred_hist_path_for, load_pred_hist, append_pred_hist)
from trading import PortfolioState, simulate_fills
from live import make_orders_from_state


# ---------------------------------------------------------------------------
def _make_panel(n_days: int = 30, n_sym: int = 40, start: str = "2026-06-01"):
    """合成面板：date × symbol，含 return_1d / opt_vix_like / TARGET。"""
    dates = pd.bdate_range(start, periods=n_days)
    syms = [f"{600000 + i}.SH" for i in range(n_sym)]
    rng = np.random.default_rng(7)
    rows = []
    for d in dates:
        for s in syms:
            rows.append({
                "date": d, "symbol": s,
                "return_1d": float(rng.normal(0, 0.01)),
                "opt_vix_like": float(rng.uniform(0.15, 0.35)),
                C.TARGET: float(rng.normal(0, 1)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------- 手算数值断言
def test_vol_target_handcalc():
    """日收益交替 ±2% → 实现波动 0.02 → scale = (0.15/√252)/0.02 ≈ 0.4725。"""
    tvs = [100.0]
    for k in range(10):
        tvs.append(tvs[-1] * (1.02 if k % 2 == 0 else 1.0 / 1.02))
    p = {"use_vol_target": True, "target_vol": 0.15, "vol_window": 20,
         "use_dd_brake": False, "use_ic_scale": False,
         "use_regime": False, "use_iv_scale": False}
    scale, ic_s, roll = risk.compute_risk_scale(tvs, [], params=p)
    # 交替 ±2% 的 rets 并非精确等幅（乘除不对称），用同一序列手算校验
    rets = np.diff(tvs) / np.array(tvs[:-1])
    expected = (0.15 / np.sqrt(252.0)) / (float(np.std(rets)) + 1e-12)
    assert scale == pytest.approx(expected, rel=1e-9)
    assert scale < 1.0    # 高波动确实触发了去杠杆
    assert ic_s == 1.0 and roll is None


def test_dd_brake_handcalc():
    """峰值 120 → 现值 90，回撤 −25% 超阈值 20% → ×0.5。"""
    tvs = [100.0, 100.0, 100.0, 100.0, 100.0, 120.0, 90.0]
    p = {"use_vol_target": False, "use_dd_brake": True, "dd_threshold": 0.20,
         "dd_scale": 0.5, "use_ic_scale": False,
         "use_regime": False, "use_iv_scale": False}
    scale, _, _ = risk.compute_risk_scale(tvs, [], params=p)
    assert scale == pytest.approx(0.5, rel=1e-12)


def test_ic_scale_handcalc():
    """roll_ic=0.005 在 [−0.01, 0.02] 的中点 → ic_scale = ic_min + 0.5×(1−ic_min)。"""
    p = {"use_vol_target": False, "use_dd_brake": False,
         "use_ic_scale": True, "ic_scale_window": 20,
         "ic_full_scale": 0.02, "ic_zero_scale": -0.01, "ic_min_scale": 0.0,
         "use_regime": False, "use_iv_scale": False}
    scale, ic_s, roll = risk.compute_risk_scale([], [0.005] * 5, params=p)
    # 代码插值分母带 1e-9 保护项，手算同步口径
    t_exp = (0.005 - (-0.01)) / (0.02 - (-0.01) + 1e-9)
    assert roll == pytest.approx(0.005, rel=1e-12)
    assert ic_s == pytest.approx(t_exp, rel=1e-12)
    assert scale == pytest.approx(t_exp, rel=1e-12)

    # IC 深度为负（≤ic_zero）→ 打到 ic_min_scale 下限
    p2 = dict(p, ic_min_scale=0.2)
    scale2, ic_s2, _ = risk.compute_risk_scale([], [-0.02] * 5, params=p2)
    assert ic_s2 == pytest.approx(0.2, rel=1e-12)
    assert scale2 == pytest.approx(0.2, rel=1e-12)

    # IC 历史不足（< max(3, window//4)）→ 不参与叠乘
    scale3, ic_s3, roll3 = risk.compute_risk_scale([], [0.005] * 2, params=p)
    assert scale3 == 1.0 and ic_s3 == 1.0 and roll3 is None


def test_stack_order_vol_then_dd_then_ic():
    """叠乘顺序：vol min 截断先定基数，再乘 dd、再乘 IC（不是对 IC 结果取 min）。"""
    tvs = [100.0]
    for k in range(10):
        tvs.append(tvs[-1] * (1.02 if k % 2 == 0 else 1.0 / 1.02))
    p = {"use_vol_target": True, "target_vol": 0.15, "vol_window": 20,
         "use_dd_brake": False, "use_ic_scale": True, "ic_scale_window": 20,
         "ic_full_scale": 0.02, "ic_zero_scale": -0.01, "ic_min_scale": 0.0,
         "use_regime": False, "use_iv_scale": False}
    rets = np.diff(tvs) / np.array(tvs[:-1])
    vol_leg = min(1.0, (0.15 / np.sqrt(252.0)) / (float(np.std(rets)) + 1e-12))
    scale, ic_s, _ = risk.compute_risk_scale(tvs, [0.005] * 5, params=p)
    t_exp = (0.005 - (-0.01)) / (0.02 - (-0.01) + 1e-9)   # 代码插值含 1e-9 保护项
    assert scale == pytest.approx(vol_leg * t_exp, rel=1e-9)


# ---------------------------------------------------------- 回测=实盘一致性
@pytest.mark.parametrize("use_regime", [False, True])
@pytest.mark.parametrize("use_iv", [False, True])
def test_parity_backtest_vs_live(use_regime, use_iv):
    """同一 state/参数：backtest 内联调用路径 == live_state 薄包装路径。"""
    panel = _make_panel()
    d = panel["date"].max()
    state = RiskState(
        dates=[str(pd.Timestamp(x).date()) for x in panel["date"].unique()[:9]],
        total_values=[100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 96.0, 95.0],
        ic_queue=[0.01, 0.005, -0.005, 0.02, 0.0],
    )
    p = {"use_vol_target": True, "use_dd_brake": True, "use_ic_scale": True,
         "use_regime": use_regime, "use_iv_scale": use_iv}
    # live 路径（薄包装内部自行预计算 regime/IV）
    live_scale, _ = live_compute_risk_scale(
        panel, d, state=RiskState.from_dict(state.to_dict()), params=p)
    # 回测路径：与 backtest_daily 主循环相同的调用方式（regime/IV 预计算后传入）
    risk_on = (_regime_risk_on(panel, d, C.REGIME_WINDOW, C.REGIME_FLOOR_RET)
               if use_regime else True)
    iv_s = (_iv_scale(panel, d, C.IV_FEATURE_FOR_SCALE, C.IV_SCALE_WINDOW,
                      C.IV_MIN_SCALE) if use_iv else 1.0)
    bt_scale, _, _ = risk.compute_risk_scale(
        state.total_values, state.ic_queue, params=p,
        risk_on=risk_on, iv_scale=iv_s)
    assert live_scale == pytest.approx(bt_scale, rel=1e-12)


# ---------------------------------------------------------- 实盘 IC 队列（死代码修复）
def test_update_ic_queue_maturity():
    """pred_hist 中满足 d_prev + fw ≤ today 的预测日才打分入队（防泄露口径）。"""
    panel = _make_panel(n_days=5, n_sym=40)
    dates = np.sort(panel["date"].unique())
    d0, d1 = dates[0], dates[1]
    # d0 的预测分与当日 TARGET 完全同序 → spearman IC = 1.0
    sub0 = panel[panel["date"] == d0]
    hist = pd.DataFrame({
        "date": [str(pd.Timestamp(d0).date())] * len(sub0),
        "symbol": sub0["symbol"].to_numpy(),
        "score": sub0[C.TARGET].to_numpy(),
    })
    state = RiskState()
    _update_ic_queue(state, panel, d1, fw=1, pred_hist=hist)
    assert list(state.ic_queue) == pytest.approx([1.0], rel=1e-9)
    assert str(pd.Timestamp(d0).date()) in state.scored_ic_dates

    # 同日重跑不重复入队；d1 自身未成熟（d1+fw > d1）不打分
    _update_ic_queue(state, panel, d1, fw=1, pred_hist=hist)
    assert len(state.ic_queue) == 1


def test_pred_hist_roundtrip_and_overwrite(tmp_path):
    """pred_hist parquet 落盘/读取/同日覆盖。"""
    p = str(tmp_path / "live_risk_state.json")
    pq = pred_hist_path_for(p)
    assert pq.endswith("_pred_hist.parquet")
    df1 = pd.DataFrame({"symbol": ["a", "b"], "score": [0.1, 0.2],
                        "pred_gbm_l2": [1.0, 2.0]})
    append_pred_hist(pq, "2026-07-08", df1)
    # 同日重跑覆盖旧行而非追加
    df1b = df1.assign(score=[0.3, 0.4])
    append_pred_hist(pq, "2026-07-08", df1b)
    append_pred_hist(pq, "2026-07-09", df1)
    hist = load_pred_hist(pq)
    assert hist["date"].nunique() == 2 and len(hist) == 4
    day8 = hist[hist["date"] == "2026-07-08"].sort_values("symbol")
    assert list(day8["score"]) == [0.3, 0.4]
    assert "pred_gbm_l2" in hist.columns


# ---------------------------------------------------------- 状态文件损坏兜底
def test_corrupt_risk_state_backed_up(tmp_path):
    p = str(tmp_path / "live_risk_state.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write("{not valid json!!!")
    state = load_risk_state(p)
    assert isinstance(state, RiskState) and len(state.dates) == 0
    baks = [x for x in os.listdir(tmp_path) if x.startswith("live_risk_state.json.corrupt-")]
    assert len(baks) == 1          # 原文件被备份，而非静默清零
    assert not os.path.exists(p)   # 原路径已挪走（os.replace）


# ---------------------------------------------------------- blocked 订单撮合语义
def test_blocked_order_skipped_in_fills():
    """blocked=True 的订单记 0 成交跳过，与实盘 --commit 跳过 blocked 一致。"""
    orders = pd.DataFrame([{
        "symbol": "600519.SH", "name": "MT", "side": "buy", "shares": 100,
        "limit_price": 100.0, "ref_close": 100.0, "reason": "涨停拦截:再平衡:建仓",
        "rank_pct": 0.0, "bb_pos": 0.5, "rsi": 50.0, "blocked": True,
        "asof_date": "2026-07-09",
    }])
    next_bars = pd.DataFrame({
        "symbol": ["600519.SH"], "open": [101.0], "high": [102.0],
        "low": [100.0], "close": [101.5], "amount": [10000.0],
        "pre_close": [100.0], "pct_chg": [1.5],
    })
    state = PortfolioState(cash=1_000_000)
    res = simulate_fills(state, orders, next_bars)
    assert res["n_fill"] == 0
    assert res["fills"]["status"].iloc[0] == "决策层拦截"
    assert state.cash == 1_000_000 and not state.holdings


# ---------------------------------------------------------- dynamic_topk 显式拒绝
def test_dynamic_topk_rejected_in_live():
    today = pd.DataFrame({
        "symbol": ["600519.SH", "000001.SZ"],
        "close": [100.0, 15.0],
        "date": pd.to_datetime(["2026-07-09"] * 2),
    })
    score = np.array([1.0, 0.5])
    with pytest.raises(ValueError, match="DYNAMIC_TOPK"):
        make_orders_from_state(today, score, {}, 1_000_000,
                               params={"use_dynamic_topk": True},
                               price_col="close")


# ---------------------------------------------------------- 2.4 实盘 IC 定权（真 OOS）
def _make_ensemble_panel(n_days: int = 45, n_sym: int = 30):
    """供 ensemble_predict_latest 的合成面板（含 TARGET/WEIGHT/因子列）。"""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2026-05-01", periods=n_days)
    syms = [f"{600000 + i}.SH" for i in range(n_sym)]
    rows = []
    for d in dates:
        f0 = rng.normal(0, 1, n_sym)
        f1 = rng.normal(0, 1, n_sym)
        tgt = 0.5 * f0 + rng.normal(0, 0.5, n_sym)   # f0 与标签有真相关
        for i, s in enumerate(syms):
            rows.append({"date": d, "symbol": s,
                         "feature_00": f0[i], "feature_01": f1[i],
                         C.TARGET: float(tgt[i]), C.WEIGHT: 1.0})
    return pd.DataFrame(rows), ["feature_00", "feature_01"]


def _fake_pred_hist(panel, dates, models):
    """构造持久化预测历史：models[0] 与标签正相关、其余负相关。"""
    rng = np.random.default_rng(3)
    rows = []
    for d in dates:
        sub = panel[panel["date"] == d]
        for (_, r), noise in zip(sub.iterrows(),
                                 rng.normal(0, 0.01, (len(sub), len(models)))):
            row = {"date": str(pd.Timestamp(d).date()), "symbol": r["symbol"],
                   "score": float(r[C.TARGET])}
            for m_i, m in enumerate(models):
                sign = 1.0 if m_i == 0 else -1.0
                row[f"pred_{m}"] = float(sign * r[C.TARGET] + noise[m_i])
            rows.append(row)
    return pd.DataFrame(rows)


def test_live_ic_weights_use_oos_pred_hist(monkeypatch):
    """2.4：live IC 定权用持久化 OOS 预测历史——正相关腿应拿到全部权重；
    无 pred_hist 时退化等权（旧实现用训练窗内日期补 observe，样本内偏差）。"""
    from live import ensemble_predict_latest

    monkeypatch.setenv("FULL_TREES", "30")     # 小数据，加速全量重建
    panel, feats = _make_ensemble_panel()
    models = ("gbm_l2", "ic_linear")
    fw = int(getattr(C, "FORWARD_WINDOW", 20))
    all_dates = np.sort(panel["date"].unique())
    # 已成熟（d + fw ≤ latest）的历史预测日
    mature_dates = all_dates[:-fw][-8:]
    hist = _fake_pred_hist(panel, mature_dates, models)
    params = {"ensemble_models": models, "n_seeds": 1, "train_window": 30,
              "ic_window": 10, "rebuild_days": 999, "incr_trees": 5,
              "pred_hist": hist}

    today, info = ensemble_predict_latest(panel, feats, params=params)
    w = info["weights"]
    # gbm_l2 在 OOS 历史上 IC≈+1、ic_linear IC≈−1 → 截断归一后 gbm_l2 拿满权重
    assert w["gbm_l2"] == pytest.approx(1.0, abs=1e-9), w
    assert w["ic_linear"] == pytest.approx(0.0, abs=1e-9), w

    # 无持久化历史 → 退化等权
    p2 = {k: v for k, v in params.items() if k != "pred_hist"}
    _, info2 = ensemble_predict_latest(panel, feats, params=p2)
    assert all(abs(info2["weights"][m] - 0.5) < 1e-9 for m in models), info2["weights"]
