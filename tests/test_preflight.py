from __future__ import annotations

import os

import pandas as pd
import pytest

import config as C
from preflight import run_preflight


def _make_panel(latest_count: int = 10) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=20)
    rows = []
    for date in dates[:-1]:
        for i in range(10):
            rows.append({"date": date, "symbol": f"{600000 + i}.SH"})
    for i in range(latest_count):
        rows.append({"date": dates[-1], "symbol": f"{600000 + i}.SH"})
    return pd.DataFrame(rows)


def _account(path, today, cash: bool = True):
    frame = pd.DataFrame({"cash": [30000.0]} if cash else {"other": [30000.0]})
    frame.to_csv(path, index=False)
    timestamp = pd.Timestamp(today).timestamp()
    os.utime(path, (timestamp, timestamp))
    return path


def test_latest_coverage_drop_fails(tmp_path):
    panel = _make_panel(latest_count=5)
    today = panel["date"].max()
    account = _account(tmp_path / "account.csv", today)
    failures = run_preflight(panel, str(account), today=today)
    assert any("覆盖" in reason for reason in failures)


def test_stale_panel_fails(tmp_path):
    panel = _make_panel()
    today = panel["date"].max() + pd.Timedelta(days=10)
    account = _account(tmp_path / "account.csv", today)
    failures = run_preflight(panel, str(account), today=today)
    assert any("面板最新日" in reason for reason in failures)


def test_missing_account_fails(tmp_path):
    panel = _make_panel()
    failures = run_preflight(panel, str(tmp_path / "missing.csv"), today=panel["date"].max())
    assert any("不存在" in reason for reason in failures)


def test_account_without_cash_fails(tmp_path):
    panel = _make_panel()
    today = panel["date"].max()
    account = _account(tmp_path / "account.csv", today, cash=False)
    failures = run_preflight(panel, str(account), today=today)
    assert any("cash" in reason for reason in failures)


def test_stale_account_fails(tmp_path):
    panel = _make_panel()
    today = panel["date"].max()
    account = _account(tmp_path / "account.csv", today - pd.Timedelta(days=10))
    failures = run_preflight(panel, str(account), today=today, max_cash_age_days=4)
    assert any("现金" in reason and "10" in reason for reason in failures)


def test_valid_panel_and_account_pass(tmp_path):
    panel = _make_panel()
    today = panel["date"].max()
    account = _account(tmp_path / "account.csv", today)
    assert run_preflight(panel, str(account), today=today) == []


def test_cli_preflight_failure_exits_before_orders(monkeypatch, tmp_path, capsys):
    import predict_tomorrow as pt

    panel = _make_panel()
    panel[C.TARGET] = 0.01
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(C, "OUTPUT_DIR", str(out_dir))
    monkeypatch.setattr(C, "USE_PREFLIGHT", True)
    monkeypatch.setattr(pt, "load_live_panel", lambda **kwargs: (panel, ["feature_00"]))

    with pytest.raises(SystemExit) as exc_info:
        pt.main(
            [
                "--account",
                str(tmp_path / "missing.csv"),
                "--holdings",
                str(tmp_path / "holdings.csv"),
                "--risk-state",
                str(tmp_path / "risk.json"),
            ]
        )
    assert exc_info.value.code == 3
    assert "前置体检" in capsys.readouterr().out
    assert not list(out_dir.glob("orders_*.csv"))
