from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
from walkforward import run_walkforward, save_walkforward_result


def _make_panel(n_days: int = 81, n_sym: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rows = []
    for sym_i in range(n_sym):
        ret = rng.normal(0.0, 0.01, n_days)
        close = (10.0 + sym_i) * np.cumprod(1.0 + ret)
        pre_close = np.concatenate([[close[0]], close[:-1]])
        feature = rng.normal(0.0, 1.0, n_days)
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{sym_i:03d}",
                    "close": close[i],
                    "adj_close": close[i],
                    "open": pre_close[i],
                    "high": max(close[i], pre_close[i]) * 1.01,
                    "low": min(close[i], pre_close[i]) * 0.99,
                    "amount": 1e9,
                    "pre_close": pre_close[i],
                    "pct_chg": ret[i] * 100.0,
                    "limit_flag": 0,
                    "return_1d": ret[i],
                    "bb_upper_20": close[i] * 1.05,
                    "bb_lower_20": close[i] * 0.95,
                    "rsi_14": 50.0,
                    C.TARGET: 0.005 * feature[i] + rng.normal(0.0, 0.02),
                    C.WEIGHT: 1.0,
                    "feature_00": feature[i],
                    "feature_01": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def test_incremental_matches_windowed_contract(monkeypatch, tmp_path):
    import lightgbm as lgb

    original_regressor = lgb.LGBMRegressor

    def _single_thread(**kwargs):
        kwargs["n_jobs"] = 1
        return original_regressor(**kwargs)

    monkeypatch.setattr(lgb, "LGBMRegressor", _single_thread)
    monkeypatch.setattr(C, "TRAIN_WINDOW", 30)
    monkeypatch.setattr(C, "TEST_WINDOW", 10)
    monkeypatch.setattr(C, "WF_STEP", 10)
    monkeypatch.setattr(C, "N_SEEDS", 1)
    monkeypatch.setattr(C, "FULL_TREES", 10)
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path / "outputs"))
    panel = _make_panel()
    factors = ["feature_00", "feature_01"]
    replay_params = {
        "n_seeds": 1,
        "train_window": 30,
        "incr_trees": 2,
        "rebuild_days": 100,
        "ic_window": 5,
        "topk": 0.3,
        "max_positions": 3,
        "max_weight": 0.34,
        "min_notional": 1000,
        "use_vol_target": False,
        "use_dd_brake": False,
        "use_ic_scale": False,
        "use_regime": False,
        "use_iv_scale": False,
    }

    windowed = run_walkforward(
        panel.copy(), factors, replay_params=replay_params, save=False, verbose=False
    )
    incremental = run_walkforward(
        panel.copy(),
        factors,
        replay_params=replay_params,
        save=False,
        verbose=False,
        engine="incremental",
    )

    assert incremental["engine"] == "incremental"
    assert set(windowed) == set(incremental) - {"engine"}
    assert windowed["n_windows"] == incremental["n_windows"] == 3
    assert len(incremental["window_metrics"]) == 3
    assert incremental["oos"]["date"].min() == windowed["oos"]["date"].min()
    assert incremental["oos"]["date"].max() == windowed["oos"]["date"].max()
    assert list(incremental["window_metrics"].columns) == list(windowed["window_metrics"].columns)

    for engine, result in (("windowed", windowed), ("incremental", incremental)):
        outdir = save_walkforward_result(result, outdir=str(tmp_path / engine))
        with open(f"{outdir}/meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        assert "nw_t" in meta
        assert "alpha_ann" in meta
        assert meta["config"]["engine"] == engine


def test_pipeline_wf_run_passes_incremental_engine(monkeypatch, tmp_path):
    from pipeline.context import StepContext
    from pipeline import steps_walkforward as steps

    monkeypatch.setattr(C, "TRAIN_WINDOW", 30)
    monkeypatch.setattr(C, "TEST_WINDOW", 10)
    monkeypatch.setattr(C, "WF_STEP", 10)
    monkeypatch.setattr(C, "FULL_TREES", 10)
    monkeypatch.setattr(C, "INCR_TREES", 2)
    monkeypatch.setattr(C, "REBUILD_DAYS", 100)
    monkeypatch.setattr(C, "IC_WINDOW", 5)
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path / "outputs"))
    ctx = StepContext(
        session_dir=str(tmp_path / "session"),
        params={
            "engine": "incremental",
            "select_features": False,
            "strategy": "long_topk",
            "capital": 1_000_000,
        },
    )
    panel = _make_panel()
    dates = np.sort(panel["date"].unique())
    monkeypatch.setattr(C, "HOLDOUT_START", str(pd.Timestamp(dates[60]).date()))
    ctx.shared["panel"] = panel
    ctx.shared["factors"] = ["feature_00", "feature_01"]

    steps.wf_run(ctx)

    assert ctx.shared["wf"]["engine"] == "incremental"
    assert (tmp_path / "session" / "artifacts" / "oos_curve.parquet").exists()
    with open(tmp_path / "outputs" / "walkforward" / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["config"]["engine"] == "incremental"
    assert meta["pre_holdout"]["n_days"] == 10
    assert meta["holdout"]["n_days"] == 20
    assert meta["n_trials"] == 1
    assert "deflated_sharpe" in meta
    trials = tmp_path / "outputs" / "experiments" / "trials.jsonl"
    assert len(trials.read_text(encoding="utf-8").splitlines()) == 1
