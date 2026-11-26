"""webapp2 数据层与写操作：只用 tmp_path，不碰真实 outputs/models。"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

from webapp2 import actions, data


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out, models = tmp_path / "outputs", tmp_path / "models"
    out.mkdir()
    models.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(models))
    monkeypatch.setattr(data, "FACTOR_LIBRARY", str(tmp_path / "factor_library"))
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    data.refresh()
    yield tmp_path
    data.refresh()


# ----------------------------------------------------------------------- 读：空态
def test_every_reader_returns_empty_state_not_exception(sandbox):
    orders, day = data.latest_orders()
    assert orders.empty and day is None
    assert data.holdings().empty and data.holdings_history().empty and data.account().empty
    assert data.risk_state() is None and data.rebalance_state() is None
    assert data.tracking_report() is None and data.champion_manifest() is None
    assert data.live_daily().empty and data.pred_hist().empty
    assert data.alerts_log() == [] and data.trials() == []
    assert data.staging_models() == [] and data.archived_models() == []
    assert data.walkforward_meta() is None and data.experiment_runs() == []
    assert data.impact_sensitivity().empty and data.factor_eval_table().empty
    ov = data.overview()
    assert ov["orders_day"] is None and ov["account"]["exists"] is False


def test_corrupt_files_are_treated_as_missing(sandbox):
    (sandbox / "outputs" / "live_risk_state.json").write_text("{broken")
    (sandbox / "outputs" / "holdings.csv").write_bytes(b"\x00\x01\x02")
    data.refresh()
    assert data.risk_state() is None
    assert data.holdings().empty or isinstance(data.holdings(), pd.DataFrame)


def test_latest_orders_picks_newest_by_filename(sandbox):
    out = sandbox / "outputs"
    pd.DataFrame({"symbol": ["A"], "side": ["buy"]}).to_csv(
        out / "orders_20261104.csv", index=False
    )
    pd.DataFrame({"symbol": ["B"], "side": ["sell"]}).to_csv(
        out / "orders_20261106.csv", index=False
    )
    (out / "orders_notadate.csv").write_text("x")
    df, day = data.latest_orders()
    assert day == "20261106" and list(df["symbol"]) == ["B"]


def test_cache_invalidates_when_cli_rewrites_file(sandbox):
    """CLI 在另一个进程里改文件，前端下一次读取必须看到新内容。"""
    p = sandbox / "outputs" / "live_risk_state.json"
    p.write_text(json.dumps({"total_values": [1]}))
    assert data.risk_state()["total_values"] == [1]
    os.utime(p, (os.path.getmtime(p) + 5, os.path.getmtime(p) + 5))
    p.write_text(json.dumps({"total_values": [1, 2]}))
    os.utime(p, (os.path.getmtime(p) + 10, os.path.getmtime(p) + 10))
    assert data.risk_state()["total_values"] == [1, 2]


def test_account_status_staleness(sandbox):
    out = sandbox / "outputs"
    assert data.account_status()["stale"] is True  # 缺失也算过期
    (out / "account.csv").write_text("date,cash\n2026-11-06,30000\n")
    data.refresh()
    s = data.account_status()
    assert s["cash"] == 30000.0 and s["stale"] is False
    old = os.path.getmtime(out / "account.csv") - 10 * 86400
    os.utime(out / "account.csv", (old, old))
    data.refresh()
    assert data.account_status()["stale"] is True


# ----------------------------------------------------------------------- 写：现金
def test_save_cash_creates_and_appends(sandbox):
    ok, msg = actions.save_cash(31234.5, "2026-11-16")
    assert ok, msg
    ok, _ = actions.save_cash(40000, "2026-11-17")
    assert ok
    text = (sandbox / "outputs" / "account.csv").read_text()
    assert text.splitlines() == ["date,cash", "2026-11-16,31234.50", "2026-11-17,40000.00"]
    assert data.account_status()["cash"] == 40000.0


@pytest.mark.parametrize("bad", [-1, "abc", None, float("nan"), float("inf")])
def test_save_cash_rejects_invalid(sandbox, bad):
    ok, _ = actions.save_cash(bad)
    assert not ok
    assert not (sandbox / "outputs" / "account.csv").exists()


def test_save_cash_keeps_existing_cash_only_header(sandbox):
    """旧 account.csv 只有 cash 一列（真实数据就是这样）：追加不得错位。"""
    (sandbox / "outputs" / "account.csv").write_text("cash\n30000\n")
    ok, msg = actions.save_cash(35000, "2026-11-16")
    assert ok, msg
    assert (sandbox / "outputs" / "account.csv").read_text().splitlines() == [
        "cash",
        "30000",
        "35000.00",
    ]


def test_save_cash_refuses_header_without_cash(sandbox):
    (sandbox / "outputs" / "account.csv").write_text("date,amount\n2026-11-06,1\n")
    ok, msg = actions.save_cash(1)
    assert not ok and "cash" in msg


# ----------------------------------------------------------------------- 写：执行勾选
def test_execution_marks_roundtrip_and_validation(sandbox):
    assert actions.set_execution_mark("20261106", "buy", "000785.SZ", "rejected")[0]
    assert actions.set_execution_mark("20261106", "sell", "600000.SH", "done")[0]
    st = data.execution_state("20261106")
    assert st == {"buy-000785.SZ": "rejected", "sell-600000.SH": "done"}
    from execution import load_rejected_rids

    assert load_rejected_rids(str(sandbox / "outputs"), "20261106") == {"buy-000785.SZ"}
    assert actions.set_execution_mark("20261106", "buy", "000785.SZ", None)[0]
    assert "buy-000785.SZ" not in data.execution_state("20261106")
    assert not actions.set_execution_mark("../x", "buy", "A", "done")[0]
    assert not actions.set_execution_mark("20261106", "buy", "A", "weird")[0]


# ----------------------------------------------------------------------- 写：成交确认
def _orders(path):
    pd.DataFrame(
        [
            ["000785.SZ", "buy", 1100, 2.12, 2.12, False, "2026-11-06"],
            ["600170.SH", "buy", 1100, 2.17, 2.17, False, "2026-11-06"],
            ["600001.SH", "buy", 100, 8.0, 8.0, True, "2026-11-06"],
        ],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked", "asof_date"],
    ).to_csv(path, index=False)


def test_confirm_fills_writes_holdings_like_commit(sandbox):
    _orders(sandbox / "outputs" / "orders_20261106.csv")
    ok, msg = actions.confirm_fills()
    assert ok, msg
    h = pd.read_csv(sandbox / "outputs" / "holdings.csv").set_index("symbol")
    assert h.loc["000785.SZ", "shares"] == 1100 and h.loc["600170.SH", "shares"] == 1100
    assert "600001.SH" not in h.index  # 拦截单不成交
    assert h.loc["000785.SZ", "entry_date"] == "2026-11-06"


def test_confirm_fills_respects_rejected_orders(sandbox):
    from datetime import datetime

    _orders(sandbox / "outputs" / "orders_20261106.csv")
    today = datetime.now().strftime("%Y%m%d")
    actions.set_execution_mark(today, "buy", "600170.SH", "rejected")
    ok, msg = actions.confirm_fills()
    assert ok, msg
    h = pd.read_csv(sandbox / "outputs" / "holdings.csv").set_index("symbol")
    assert "000785.SZ" in h.index and "600170.SH" not in h.index


def test_confirm_fills_without_orders_or_all_blocked(sandbox):
    ok, msg = actions.confirm_fills()
    assert not ok and "没有订单" in msg
    pd.DataFrame(
        [["600001.SH", "buy", 100, 8.0, 8.0, True, "2026-11-06"]],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked", "asof_date"],
    ).to_csv(sandbox / "outputs" / "orders_20261106.csv", index=False)
    data.refresh()
    ok, msg = actions.confirm_fills()
    assert not ok and not (sandbox / "outputs" / "holdings.csv").exists()


def test_confirm_fills_day_mismatch(sandbox):
    _orders(sandbox / "outputs" / "orders_20261106.csv")
    ok, msg = actions.confirm_fills("20261101")
    assert not ok and "不一致" in msg


# ----------------------------------------------------------------------- 写：模型
def _make_model(root, sub, mid, train_end="2026-09-03"):
    d = root / "models" / sub / mid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"model_id": mid, "train_end_date": train_end, "config_snapshot": {}})
    )
    (d / "ensemble_state.json").write_text("{}")
    return d


def test_promote_validates_id_and_archives_old_champion(sandbox, monkeypatch):
    assert not actions.promote_staging("../evil")[0]
    assert not actions.promote_staging("nonexistent")[0]
    _make_model(sandbox, "staging", "cand1")
    champ = sandbox / "models" / "champion"
    champ.mkdir()
    (champ / "manifest.json").write_text(json.dumps({"model_id": "old"}))
    (champ / "ensemble_state.json").write_text("{}")
    monkeypatch.setattr("model_store.C.MODEL_DIR", str(sandbox / "models"), raising=False)
    ok, msg = actions.promote_staging("cand1", "测试用晋升，无门控评估")
    assert ok, msg
    assert data.champion_manifest()["model_id"] == "cand1"
    assert (sandbox / "models" / "archive" / "old").is_dir()
    assert "PROMOTE cand1" in (sandbox / "models" / "promotions.log").read_text()


def test_delete_staging(sandbox):
    _make_model(sandbox, "staging", "c")
    assert not actions.delete_staging("../c")[0]
    assert actions.delete_staging("c")[0]
    assert data.staging_models() == []
    assert not actions.delete_staging("c")[0]


# ----------------------------------------------------------------------- 启动 / 终止 CLI
def test_launch_cli_rejects_unknown_entry_and_unsafe_args(sandbox):
    assert not actions.launch_cli("rm -rf /")[0]
    assert not actions.launch_cli("nightly", ["--x; rm -rf /"])[0]
    assert not actions.launch_cli("nightly", ["$(whoami)"])[0]
    assert not actions.launch_cli("nightly", ["a b"])[0]


def test_launch_cli_runs_whitelisted_entry_and_reserves_run_dir(sandbox, monkeypatch):
    """用一个假的入口脚本验证：子进程拿到 RUN_ID、日志重定向到 run.log、前端立即得到 run_id。"""
    script = sandbox / "fake_cli.py"
    script.write_text(
        "import os, sys; print('RUN_ID=' + os.environ['RUN_ID']); print('args=' + ','.join(sys.argv[1:]))"
    )
    monkeypatch.setitem(actions.CLI_ENTRIES, "nightly", [str(script)])
    monkeypatch.setattr(actions, "PY", sys.executable)
    ok, msg, rid = actions.launch_cli("nightly", ["--capital", "30000"])
    assert ok and rid, msg
    import time

    log = sandbox / "outputs" / "runs" / rid / "run.log"
    for _ in range(100):
        if log.exists() and "args=" in log.read_text():
            break
        time.sleep(0.1)
    text = log.read_text()
    assert f"RUN_ID={rid}" in text and "args=--capital,30000" in text


def test_stop_run_refuses_finished_and_unknown(sandbox):
    import run_registry as rr

    root = str(sandbox / "outputs" / "runs")
    with rr.RunReporter("nightly", run_id="done_run", root=root):
        pass
    assert not actions.stop_run("done_run")[0]
    assert not actions.stop_run("nope")[0]
    assert not actions.stop_run("../x")[0]
