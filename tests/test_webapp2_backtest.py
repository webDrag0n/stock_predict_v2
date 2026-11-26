"""回测页：窗口 Sharpe 复算（T17 之前的增量产物存档值偏高）、可信度、对比、HOLDOUT/DSR。"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import run_registry as rr
from webapp2 import app as appmod
from webapp2 import backtest_logic as bl
from webapp2 import data
from webapp2.pages import backtest

from tests.test_webapp2_pages import ids_of, text_of  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    data.refresh()
    yield tmp_path
    data.refresh()


def make_curve(daily_ret):
    """由日收益序列造 oos_curve：equity = 累计净 PnL / 初始资金 = 总资产 - 1。"""
    tv = np.cumprod(1.0 + np.asarray(daily_ret, dtype=float))
    dates = pd.bdate_range("2025-01-02", periods=len(tv))
    return pd.DataFrame({"date": dates, "equity": tv - 1.0})


def put_run(root, name, meta, windows=None, curve=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta))
    if windows is not None:
        windows.to_parquet(d / "window_metrics.parquet", index=False)
    if curve is not None:
        curve.to_parquet(d / "oos_curve.parquet", index=False)
    data.refresh()
    return d


def meta(engine=None, **kw):
    base = {
        "generated_at": "2026-11-15T16:27:28",
        "metrics": {"sharpe": 1.1, "ann_return": 0.2, "max_drawdown": -0.17, "pred_ic": 0.024},
        "n_windows": 3,
        "predic_pos_ratio": 0.667,
        "sharpe_pos_ratio": 0.889,
        "config": {
            "engine": engine,
            "test_window": 20,
            "initial_capital": 1e6,
            "model_objective": "fusion",
        },
        "universe": {"n_symbols": 5000, "oos_start": "2025-01-02", "oos_end": "2025-03-27"},
    }
    base.update(kw)
    return base


# ----------------------------------------------------------------------------- 复算
def test_local_window_sharpe_matches_a_hand_computation():
    r = [0.01, -0.01] * 10 + [0.02, 0.0] * 10  # 两个 20 日窗口
    sh = bl.local_window_sharpe(make_curve(r), 20)
    a = np.array(r[:20])
    assert sh[0] == pytest.approx(a.mean() / a.std(ddof=1) * np.sqrt(252))
    assert sh[1] > sh[0] and len(sh) == 2


def test_window_boundary_return_uses_previous_windows_ending_value():
    """第二个窗口的首日收益必须相对上一窗口末的总资产，而不是相对 1.0（T17 的错误正是基准错了）。"""
    r = [0.05] * 20 + [0.001, 0.002, -0.001, 0.0] * 5
    sh = bl.local_window_sharpe(make_curve(r), 20)
    seg = np.array(r[20:])
    assert sh[1] == pytest.approx(seg.mean() / seg.std(ddof=1) * np.sqrt(252))


def test_local_window_sharpe_handles_degenerate_input():
    assert bl.local_window_sharpe(pd.DataFrame(), 20) == []
    assert bl.local_window_sharpe(make_curve([0.01] * 5), 0) == []
    flat = bl.local_window_sharpe(make_curve([0.0] * 20), 20)
    assert len(flat) == 1 and np.isnan(flat[0])  # 零波动不除零


# ----------------------------------------------------------------------------- 窗口表
def _inc_case(stored_sharpe):
    rets = [0.01, -0.012] * 10 + [-0.01, 0.008] * 10 + [0.012, -0.009] * 10
    curve = make_curve(rets)
    w = pd.DataFrame(
        {
            "window": [0, 1, 2],
            "test": ["a", "b", "c"],
            "n_feat": [160] * 3,
            "Sharpe": stored_sharpe,
            "MaxDD": [-0.05] * 3,
            "PredIC": [0.01] * 3,
        }
    )
    return curve, w


def test_incremental_artifact_with_inflated_archive_is_flagged_and_recomputed():
    """真实情形：T17 之前的增量产物，存档 8/9 窗口为正，复算只有 5/9。"""
    curve, w = _inc_case([2.0, 2.0, 2.0])  # 存档说三个窗口都为正
    wt = bl.window_table(meta("incremental"), w, curve)
    assert wt["recomputed"] and wt["stale"]
    assert wt["pos_stored"] == 3 and wt["pos_used"] < 3  # 页面采用复算值
    assert "Sharpe_复算" in wt["df"].columns


def test_incremental_artifact_that_matches_is_not_flagged():
    curve, _ = _inc_case([0, 0, 0])
    mine = bl.local_window_sharpe(curve, 20)
    _, w = _inc_case(mine)
    wt = bl.window_table(meta("incremental"), w, curve)
    assert wt["recomputed"] and not wt["stale"] and wt["pos_used"] == wt["pos_stored"]


def test_windowed_artifact_keeps_stored_values():
    """滚动窗口引擎的窗口各自独立重放，曲线分段还原不出来，不能复算。"""
    curve, w = _inc_case([2.0, 2.0, -1.0])
    wt = bl.window_table(meta(None), w, curve)
    assert not wt["recomputed"] and not wt["stale"] and wt["pos_used"] == 2


def test_window_table_with_mismatched_curve_length_does_not_guess():
    curve, w = _inc_case([2.0, 2.0, 2.0])
    wt = bl.window_table(meta("incremental"), w, curve.iloc[:25])  # 窗口数对不上
    assert not wt["recomputed"] and wt["pos_used"] == 3


def test_window_table_empty():
    wt = bl.window_table(meta("incremental"), pd.DataFrame(), pd.DataFrame())
    assert wt["n"] == 0 and wt["pos_used"] is None


# ----------------------------------------------------------------------------- 徽章 / holdout / DSR
def test_trust_badges_thresholds():
    b = dict(bl.trust_badges({"predic_pos_ratio": 0.4, "nw_t": 1.05}, 5, 9))
    assert b["Sharpe>0 窗口 5/9（56%）"] == "bad"  # <60% 判红
    assert b["PredIC>0 窗口 40%"] == "bad" and b["NW-t 1.05"] == "bad"
    good = dict(bl.trust_badges({"predic_pos_ratio": 0.8, "nw_t": 2.5}, 8, 9))
    assert set(good.values()) == {"ok"}
    assert bl.trust_badges({}, None, 0) == []


def test_holdout_and_dsr():
    m = {"pre_holdout": {"sharpe": 2.0, "n_days": 400}, "holdout": {"sharpe": -0.5, "n_days": 100}}
    rows = bl.holdout_rows(m)
    assert [r["n_days"] for r in rows] == [400, 100] and "HOLDOUT" in rows[1]["seg"]
    assert bl.holdout_rows({}) == [] and bl.holdout_rows({"holdout": {}}) == []
    text, lv = bl.dsr_note({"deflated_sharpe": 0.62, "n_trials": 7})
    assert "0.62" in text and "7 次试验" in text and lv == "bad"
    assert bl.dsr_note({"deflated_sharpe": 0.99})[1] == "ok"
    assert bl.dsr_note({}) is None and bl.dsr_note({"deflated_sharpe": float("nan")}) is None


# ----------------------------------------------------------------------------- 对比
def test_compare_rows_use_recomputed_positive_windows_and_flag_snapshot_mismatch(sandbox):
    exp = sandbox / "outputs" / "experiments"
    curve, w = _inc_case([2.0, 2.0, 2.0])
    put_run(
        sandbox / "outputs",
        "walkforward",
        meta(None, universe={"oos_start": "2024-03-25", "oos_end": "2026-08-11"}),
        w,
        curve,
    )
    put_run(exp / "inc", "walkforward", meta("incremental"), w, curve)
    rows = bl.compare_rows(data.experiment_runs())
    by = {r["label"]: r for r in rows}
    inc = next(r for k, r in by.items() if "inc" in k)
    canon = by["当前 canonical"]
    assert inc["stale"] and inc["pos"] < 3 and not canon["stale"] and canon["pos"] == 3
    assert bl.snapshot_mismatch(rows)
    assert not bl.snapshot_mismatch(rows[:1])


# ----------------------------------------------------------------------------- 页面
def test_page_on_empty_disk(sandbox):
    t = text_of(backtest.layout())
    assert "还没有任何 walk-forward 产物" in t and "没有实验产物" in t
    assert "没有 impact_sensitivity.csv" in t and "还没有 walk-forward 试验记录" in t


def test_page_flags_stale_incremental_archive(sandbox):
    curve, w = _inc_case([2.0, 2.0, 2.0])
    d = put_run(sandbox / "outputs", "walkforward", meta("incremental"), w, curve)
    t = text_of(backtest.detail(str(d)))
    assert "不一致" in t and "T17" in t and "复算" in t
    # 徽章必须用复算值（1/3），而不是存档的 3/3；警告文字里提到「存档 3/3」是正常的
    assert "Sharpe>0 窗口 1/3" in t
    assert "Sharpe>0 窗口 3/3" not in t


def test_page_windowed_gets_independent_replay_note(sandbox):
    curve, w = _inc_case([2.0, 2.0, 2.0])
    d = put_run(sandbox / "outputs", "walkforward", meta(None), w, curve)
    t = text_of(backtest.detail(str(d)))
    assert "各自从空仓独立重放" in t and "不一致" not in t


def test_page_handles_missing_and_corrupt_artifacts(sandbox):
    assert "不存在或已损坏" in text_of(backtest.detail(str(sandbox / "nope")))
    d = sandbox / "outputs" / "walkforward"
    d.mkdir()
    (d / "meta.json").write_text("{broken")
    data.refresh()
    assert "不存在或已损坏" in text_of(backtest.detail(str(d)))


def test_page_without_optional_files_renders(sandbox):
    d = put_run(sandbox / "outputs", "walkforward", meta(None))  # 没有窗口表也没有曲线
    t = text_of(backtest.detail(str(d)))
    assert "没有 oos_curve.parquet" in t and "没有 window_metrics.parquet" in t
    assert "没有 HOLDOUT 分段" in t and "没有 Deflated Sharpe" in t


def test_page_shows_holdout_and_dsr_when_present(sandbox):
    m = meta(
        None,
        pre_holdout={"sharpe": 2.1, "pred_ic": 0.05, "n_days": 400},
        holdout={"sharpe": -0.4, "pred_ic": 0.0, "n_days": 90},
        deflated_sharpe=0.7,
        n_trials=5,
    )
    d = put_run(sandbox / "outputs", "walkforward", m)
    t = text_of(backtest.detail(str(d)))
    assert "HOLDOUT（只用于最终报告）" in t and "-0.40" in t and "Deflated Sharpe 0.70" in t


def test_impact_and_trials_cards(sandbox):
    exp = sandbox / "outputs" / "experiments"
    exp.mkdir(parents=True)
    pd.DataFrame(
        {
            "k": [0.0, 0.1],
            "sharpe": [2.6, 2.4],
            "ann_return": [0.4, 0.39],
            "max_drawdown": [-0.08, -0.09],
            "total_cost": [0.03, 0.13],
        }
    ).to_csv(exp / "impact_sensitivity.csv", index=False)
    (exp / "trials.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-11-15T23:35:08",
                "kind": "walkforward",
                "config": {"engine": "windowed", "model_objective": "fusion"},
                "metrics": {"sharpe": 2.44},
                "config_hash": "abc",
            }
        )
        + "\n"
        + json.dumps(
            {
                "ts": "2026-11-15T23:36:00",
                "kind": "impact_sensitivity",
                "config": {},
                "metrics": {},
                "config_hash": "zzz",
            }
        )
        + "\n"
    )
    data.refresh()
    t = text_of(backtest.lower_cards())
    assert "+2.44" in t and "abc" in t and "zzz" not in t  # 只列 walk-forward 试验
    assert "K=0.1 是默认口径" in t


def test_dropdown_and_launch_buttons_outside_polled_regions(sandbox):
    ids = ids_of(backtest.layout())
    assert {
        "bt-run",
        "bt-refresh",
        "bt-launch-inc",
        "bt-launch-win",
        "bt-tick",
        "bt-sig",
        "bt-detail",
        "bt-run-status",
    } <= ids
    assert not (
        {"bt-run", "bt-refresh", "bt-launch-inc", "bt-run-status"} & ids_of(backtest.detail(None))
    )


# ----------------------------------------------------------------------------- 刷新门控
def test_signature_changes_for_artifacts_but_not_run_progress(sandbox):
    sig0 = backtest.signature()
    assert backtest.signature() == sig0
    with rr.RunReporter("walkforward", run_id="w1", root=str(sandbox / "outputs" / "runs")) as run:
        run.progress("windows", 1, 3)
        assert backtest.signature() == sig0
    assert backtest.signature() == sig0
    put_run(sandbox / "outputs", "walkforward", meta(None))
    assert backtest.signature() != sig0


def test_refresh_defers_artifact_updates_until_active_run_finishes(sandbox):
    d = put_run(sandbox / "outputs", "walkforward", meta(None))
    sig0 = backtest.signature()
    assert backtest.refresh_decision(sig0, str(d), active=False) == (None, None, sig0)

    with rr.RunReporter("walkforward", run_id="w2", root=str(sandbox / "outputs" / "runs")) as run:
        run.progress("windows", 2, 3)
        meta_path = d / "meta.json"
        meta_path.write_text(json.dumps(meta(None, generated_at="2026-11-16T00:00:00")))
        mtime = meta_path.stat().st_mtime
        os.utime(meta_path, (mtime + 2, mtime + 2))
        assert backtest.signature() != sig0
        assert backtest.refresh_decision(sig0, str(d)) == (None, None, sig0)

    new_detail, new_lower, sig1 = backtest.refresh_decision(sig0, str(d))
    assert new_detail is not None and new_lower is not None
    assert sig1 != sig0
    assert "2026-11-16" in text_of(new_detail)


def test_run_status_updates_progress_in_its_own_region(sandbox):
    with rr.RunReporter("walkforward", run_id="w3", root=str(sandbox / "outputs" / "runs")) as run:
        run.progress("windows", 1, 4, message="第 1 个窗口")
        t = text_of(backtest._run_status())
        assert "正在运行 walk-forward" in t and "25%" in t and "第 1 个窗口" in t
        assert "w3" in t and "查看实时进度" in t
    assert "完成" in text_of(backtest._run_status())


def test_default_value_prefers_canonical_and_routes(sandbox):
    assert backtest.default_value() is None
    d = put_run(sandbox / "outputs", "walkforward", meta(None))
    assert backtest.default_value() == str(d)
    assert ("/backtest", "回测") in appmod.NAV
    assert "选择产物" in text_of(appmod.route("/backtest"))
