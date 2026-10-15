"""seed_backtest 纯计算 + robustness 会话模式报告结构测试。

seed_backtest 在小合成 OOS 帧（prediction 与 return_1d 正相关）上指标有限且
符号合理（多头 topk 应赚正 PnL）；ms_run 用 monkeypatch 掉真实训练（stub
run_one_seed 返回合成 OOS），ms_report 验证 6 图 + 汇总表结构齐全。

路径由 tests/conftest.py 统一处理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from multiseed_eval import seed_backtest
from pipeline.context import StepContext
from pipeline import steps_robustness as R


def _make_oos(n_dates: int = 80, n_syms: int = 30) -> pd.DataFrame:
    """合成 OOS 帧：信号按 20 日块恒定，prediction 与 return_1d 同块相关。

    回测引擎把持仓 shift 一日（pred_t 赚 ret_{t+1}），块内信号恒定使该相关性
    穿透到持仓期，多头 topk 应为正收益。
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    block = np.arange(n_dates) // 20
    rows = []
    for i in range(n_syms):
        sig_blk = rng.normal(0, 0.01, block.max() + 1)
        sig = sig_blk[block]
        ret = sig + rng.normal(0, 0.01, n_dates)
        pred = sig + rng.normal(0, 0.005, n_dates)
        for j, d in enumerate(dates):
            rows.append({
                "date": d, "symbol": f"S{i:03d}",
                "return_1d": ret[j],
                C.TARGET: sig[j] + rng.normal(0, 0.01),
                "prediction": pred[j],
            })
    return pd.DataFrame(rows)


def _amount_map(oos: pd.DataFrame) -> pd.DataFrame:
    am = oos[["date", "symbol"]].copy()
    am["amount"] = 50000.0
    return am


def _fake_run(net: np.ndarray, bench: np.ndarray, dts: np.ndarray) -> dict:
    """构造一个结构与 seed_backtest 输出一致的 run dict（供 ms_report 测试）。"""
    n = min(len(net), len(bench))
    strat_cum = np.cumprod(1 + net) - 1.0
    bench_cum = np.cumprod(1 + bench[:n]) - 1.0
    return {
        "net": net, "bench": bench[:n],
        "strat_cum_pct": strat_cum * 100, "bench_cum_pct": bench_cum * 100,
        "alpha_pct": (strat_cum[:n] - bench_cum) * 100,
        "tc_comm": np.full(len(net), 1e-5), "tc_impact": np.full(len(net), 2e-5),
        "sharpe": 1.5, "calmar": 0.8, "maxdd": -0.05, "alpha_ann": 0.12,
        "nw_t": 2.5, "beta": 0.9, "total_pnl": 0.03,
        "cost_comm": 1e-4, "cost_impact": 2e-4, "hit": 0.55,
        "dts": dts[:len(net)],
    }


def test_seed_backtest_synthetic_metrics():
    """正相关信号 → total_pnl/sharpe 为正；全部指标有限。"""
    oos = _make_oos()
    r = seed_backtest(oos, _amount_map(oos))
    for key in ("net", "bench", "strat_cum_pct", "bench_cum_pct", "alpha_pct",
                "tc_comm", "tc_impact"):
        arr = np.asarray(r[key], dtype=np.float64)
        assert np.isfinite(arr).all(), f"{key} 含非有限值"
    for key in ("sharpe", "calmar", "maxdd", "alpha_ann", "nw_t", "beta",
                "total_pnl", "cost_comm", "cost_impact", "hit"):
        v = float(r[key])
        assert np.isfinite(v), f"{key} 非有限: {v}"
    assert r["total_pnl"] > 0, "正相关信号下多头 topk 应为正 PnL"
    assert r["sharpe"] > 0
    assert r["maxdd"] <= 0
    assert 0.0 <= r["hit"] <= 1.0
    assert r["cost_comm"] >= 0 and r["cost_impact"] >= 0
    # alpha 序列 = 策略累计 − 基准累计
    assert len(r["alpha_pct"]) == len(r["net"])
    np.testing.assert_allclose(
        r["alpha_pct"], r["strat_cum_pct"][:len(r["alpha_pct"])]
        - r["bench_cum_pct"], rtol=1e-9)


def test_ms_run_stubbed_training(tmp_path, monkeypatch):
    """ms_run：stub 掉真实训练（2 seed），验证逐 seed artifact 与 shared 结构。"""
    oos = _make_oos()
    _amount_map(oos).to_parquet(tmp_path / "panel.parquet", index=False)
    monkeypatch.setattr(R, "_PANEL_CACHE", str(tmp_path / "panel.parquet"))
    monkeypatch.setattr(R, "_SEED_CACHE",
                        str(tmp_path / "seed_oos_{seed}.parquet"))
    monkeypatch.setattr(R, "_run_one_seed",
                        lambda panel, factors, seed: oos)

    ctx = StepContext(session_dir=str(tmp_path / "sess"),
                      params={"seeds": "42,123", "capital": 1000000})
    rep = R.ms_run(ctx)
    assert rep["step"] == "ms_run"
    runs = ctx.shared["ms_runs"]
    assert sorted(runs.keys()) == [42, 123]
    for sd in (42, 123):
        assert (tmp_path / "sess" / "artifacts"
                / f"seed_{sd}_series.parquet").exists()
    assert (tmp_path / "sess" / "artifacts" / "seed_metrics.parquet").exists()


def test_ms_report_structure(tmp_path):
    """ms_report：6 图（line×4 + area_stacked + heatmap）+ 汇总表 + MEAN±SD 行。"""
    rng = np.random.default_rng(0)
    T = 130
    dts = pd.date_range("2025-01-01", periods=T, freq="B").values
    bench = rng.normal(0, 0.01, T)
    runs = {sd: _fake_run(rng.normal(5e-4, 0.01, T), bench, dts)
            for sd in (42, 123)}
    ctx = StepContext(session_dir=str(tmp_path),
                      params={"seeds": "42,123", "capital": 1000000})
    ctx.shared["ms_runs"] = runs

    rep = R.ms_report(ctx)
    assert rep["step"] == "ms_report"
    types = [f["type"] for f in rep["figures"]]
    assert types == ["line", "line", "line", "line", "area_stacked",
                     "heatmap"], types

    fig_a, fig_b, _, fig_d, _, _ = rep["figures"]
    # 图A：2 条 seed 淡线 + 均值 + 基准
    assert len(fig_a["series"]) == 4
    assert fig_a["series"][0]["opacity"] == 0.35
    # 图B：均值线带 ±1σ band；图D：均值回撤带 min/max band
    assert fig_b["series"][-1]["band"] is not None
    assert len(fig_b["series"][-1]["band"]["upper"]) == len(
        fig_b["series"][-1]["y"])
    assert fig_d["series"][0]["band"] is not None

    # 表G：2 seed + MEAN±SD 行
    tbl = rep["tables"][0]
    assert tbl["n_total"] == 3
    seed_col = tbl["columns"].index("seed")
    assert tbl["rows"][-1][seed_col] == "MEAN±SD"

    labels = [k["label"] for k in rep["kpis"]]
    assert any("判定" in lb for lb in labels)


def test_ms_report_from_artifacts_only(tmp_path, monkeypatch):
    """shared 缺失时 ms_report 仅凭 artifact 重建（模拟跨进程回读）。"""
    oos = _make_oos()
    _amount_map(oos).to_parquet(tmp_path / "panel.parquet", index=False)
    monkeypatch.setattr(R, "_PANEL_CACHE", str(tmp_path / "panel.parquet"))
    monkeypatch.setattr(R, "_SEED_CACHE",
                        str(tmp_path / "seed_oos_{seed}.parquet"))
    monkeypatch.setattr(R, "_run_one_seed",
                        lambda panel, factors, seed: oos)

    ctx = StepContext(session_dir=str(tmp_path / "sess"),
                      params={"seeds": "42,123", "capital": 1000000})
    R.ms_run(ctx)
    ctx.shared.pop("ms_runs")
    rep = R.ms_report(ctx)
    types = [f["type"] for f in rep["figures"]]
    assert types == ["line", "line", "line", "line", "area_stacked",
                     "heatmap"], types


def test_registry_robustness_wired():
    from pipeline import registry
    assert [s[0] for s in registry.STEPS["robustness"]] == [
        "ms_run", "ms_report"]
    assert registry.MODE_LABELS["robustness"] == "多Seed稳健性"
    keys = {p["key"] for p in registry.PARAM_SCHEMA["robustness"]}
    assert {"seeds", "capital", "train_window"} <= keys
    assert registry.default_params("robustness")["seeds"] == "42,123,7"
