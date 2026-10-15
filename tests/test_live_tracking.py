"""live_tracking / live_state.collect_matured_ics 测试。

覆盖：成熟 IC 纯函数与 _update_ic_queue 口径一致；IC/净值统计；
快照追加与同日覆盖；报告结构与告警（IC 退化/收益落后/无净值降级）。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_live_tracking.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from live_state import (RiskState, _update_ic_queue, collect_matured_ics,
                        save_risk_state)
from live_tracking import (append_daily_snapshot, compute_report, load_daily,
                           matured_ic_stats, nav_stats)

N_SYMS = 35
FW = int(getattr(C, "FORWARD_WINDOW", 20))


def _make_panel(n_dates: int = 60, seed: int = 1):
    """35 只 × n_dates 的面板；返回 (panel, z)，z[(date_str, sym)] 是当日
    与 TARGET 正相关的共享因子（pred_hist 的 score 用它 → 正 IC）。"""
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rng = np.random.RandomState(seed)
    syms = [f"S{j:03d}" for j in range(N_SYMS)]
    z = {}
    rows = []
    for d in dates:
        zd = rng.randn(N_SYMS)
        for i, s in enumerate(syms):
            z[(str(d.date()), s)] = float(zd[i])
            rows.append({"date": d, "symbol": s,
                         "return_1d": float(rng.randn() * 0.02),
                         C.TARGET: float(zd[i] * 0.02 + rng.randn() * 0.01)})
    return pd.DataFrame(rows), z


def _pred_hist(panel, z, day_indices, sign=1.0):
    dates = sorted(panel["date"].unique())
    rows = []
    for i in day_indices:
        d = dates[i]
        ds = str(pd.Timestamp(d).date())
        for s in panel[panel["date"] == d]["symbol"]:
            rows.append({"date": ds, "symbol": s,
                         "score": sign * z[(ds, s)]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def test_collect_matured_ics_matches_update_ic_queue():
    """纯函数与状态版逐项一致：同样的成熟日、同样的 IC 值。"""
    panel, z = _make_panel()
    pred_idx = list(range(5, 25))          # i + FW ≤ 59，全部成熟
    ph = _pred_hist(panel, z, pred_idx)
    latest = panel["date"].max()

    ics, done = collect_matured_ics(panel, ph, latest, FW)
    assert len(ics) == len(pred_idx)
    assert all(v > 0 for v in ics.values()), "score 与 target 正相关，IC 应为正"

    state = RiskState()
    _update_ic_queue(state, panel, latest, FW, pred_hist=ph)
    assert list(state.ic_queue) == [ics[d] for d in sorted(ics)]
    assert set(state.scored_ic_dates) == set(done)

    # 未成熟日不打分：只给最后 FW-1 天的预测 → 全未成熟
    late_idx = [len(panel["date"].unique()) - 1]
    ics2, done2 = collect_matured_ics(panel, _pred_hist(panel, z, late_idx),
                                      latest, FW)
    assert ics2 == {} and done2 == []


def test_matured_ic_stats_and_sign():
    panel, z = _make_panel()
    ph = _pred_hist(panel, z, range(5, 25))
    stats = matured_ic_stats(panel, ph, window=10)
    assert stats["n_ic"] == 20
    assert stats["ic_mean"] > 0 and stats["ic_mean_w"] > 0
    # 反相关 → 负 IC
    stats_neg = matured_ic_stats(panel, _pred_hist(panel, z, range(5, 25), sign=-1.0))
    assert stats_neg["ic_mean_w"] < 0
    # 空 pred_hist 降级
    empty = matured_ic_stats(panel, pd.DataFrame(columns=["date", "symbol", "score"]))
    assert empty["n_ic"] == 0 and empty["ic_mean_w"] is None


def test_nav_stats_with_and_without_nav(tmp_path):
    panel, _ = _make_panel()
    rs_path = str(tmp_path / "risk.json")

    # 无文件 → 无净值降级
    nav = nav_stats(rs_path, panel, window=20)
    assert nav["has_nav"] is False and nav["live_ret_w"] is None

    # 25 个净值点（上行 + 噪声，避免日收益零波动导致 Sharpe 退化为 0）
    dates = sorted(panel["date"].unique())[-25:]
    rng = np.random.RandomState(11)
    tvs = []
    v = 1_000_000.0
    for r in 0.003 + rng.randn(25) * 0.004:
        v *= 1 + float(r)
        tvs.append(v)
    save_risk_state(rs_path, RiskState(
        dates=[str(pd.Timestamp(d).date()) for d in dates], total_values=tvs))
    nav = nav_stats(rs_path, panel, window=20)
    assert nav["has_nav"] is True and nav["n_nav"] == 25
    assert nav["live_ret_w"] > 0
    assert nav["sharpe_w"] is not None and nav["sharpe_w"] > 0
    assert nav["bench_ret_w"] is not None      # panel 有 return_1d
    assert nav["max_dd_w"] <= 0


def test_snapshot_append_and_same_day_overwrite(tmp_path):
    tdir = str(tmp_path / "tracking")
    ic = {"n_ic": 5, "ic_mean_w": 0.01, "ics": {}}
    nav = {"total_value": 1e6, "live_ret_1d": 0.001, "bench_ret_1d": 0.0,
           "live_ret_w": 0.02, "bench_ret_w": 0.01}
    append_daily_snapshot("2026-09-30", "m1", "frozen", 5000, ic, nav,
                          tracking_dir=tdir)
    append_daily_snapshot("2026-10-01", "m1", "trained", 5100, ic, nav,
                          tracking_dir=tdir)
    # 同日重跑覆盖
    append_daily_snapshot("2026-10-01", "m2", "promoted", 5100, ic, nav,
                          tracking_dir=tdir)
    df = load_daily(tdir)
    assert len(df) == 2
    row = df[df["date"] == "2026-10-01"].iloc[0]
    assert row["model_id"] == "m2" and row["action"] == "promoted"
    assert row["n_ic_total"] == 5


def test_compute_report_alerts():
    champ = {"model_id": "m1", "train_end_date": "2026-08-01",
             "promotion": {"decided_at": "2026-08-02",
                           "candidate_metrics": {"sharpe": 1.2, "max_drawdown": -8.0}}}
    # IC 退化 + 收益落后 + 方向背离 同时亮
    ic_stats = {"n_ic": 30, "ic_mean": -0.01, "ic_mean_w": -0.02, "window": 20,
                "ics": {}}
    nav = {"has_nav": True, "n_nav": 30, "window": 20,
           "live_ret_w": -0.05, "bench_ret_w": 0.02, "sharpe_w": -0.8,
           "max_dd_w": -0.07, "total_value": 950000.0}
    report = compute_report(champ, ic_stats, nav)
    txt = "；".join(report["alerts"])
    assert "IC 退化" in txt
    assert "落后基准" in txt
    assert "方向背离" in txt
    assert report["divergence"]["sharpe_gap"] is not None
    assert report["champion"]["backtest_expectation"]["sharpe"] == 1.2

    # 无净值降级：有告警提示、方向背离不触发
    nav_none = {"has_nav": False, "n_nav": 0, "window": 20}
    report2 = compute_report(champ, ic_stats, nav_none)
    txt2 = "；".join(report2["alerts"])
    assert "无真实净值" in txt2
    assert "方向背离" not in txt2
    assert report2["divergence"]["sharpe_gap"] is None

    # 一切正常：无 IC/收益告警（可能有无净值提示）
    good_ic = {"n_ic": 30, "ic_mean": 0.03, "ic_mean_w": 0.03, "window": 20,
               "ics": {}}
    good_nav = {"has_nav": True, "n_nav": 30, "window": 20,
                "live_ret_w": 0.04, "bench_ret_w": 0.01, "sharpe_w": 1.0,
                "max_dd_w": -0.02, "total_value": 1.04e6}
    report3 = compute_report(champ, good_ic, good_nav)
    assert report3["alerts"] == []


if __name__ == "__main__":
    import tempfile
    test_collect_matured_ics_matches_update_ic_queue()
    test_matured_ic_stats_and_sign()
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        test_nav_stats_with_and_without_nav(Path(td))
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        test_snapshot_append_and_same_day_overwrite(Path(td))
    test_compute_report_alerts()
    print("live_tracking tests passed")
