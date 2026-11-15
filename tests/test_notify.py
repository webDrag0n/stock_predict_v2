from __future__ import annotations

import json

import pytest

import config as C
import notify


def test_log_alerts_deduplicated_for_three_days(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "")
    day = {"value": "2026-01-01"}
    monkeypatch.setattr(notify, "_today", lambda: day["value"])
    called = []
    monkeypatch.setattr(
        notify.urllib.request,
        "urlopen",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    for expected_lines, current in (
        (1, "2026-01-01"),
        (1, "2026-01-02"),
        (1, "2026-01-03"),
        (2, "2026-01-04"),
    ):
        day["value"] = current
        notify.send_alerts(["模型超龄"], "夜间告警")
        log = tmp_path / "alerts.log"
        assert len(log.read_text(encoding="utf-8").splitlines()) == expected_lines
    assert not called


def test_webhook_failure_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(notify, "_today", lambda: "2026-01-01")

    def _fail(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(notify.urllib.request, "urlopen", _fail)
    notify.send_alerts(["IC 退化"], "夜间告警")
    assert not (tmp_path / "alerts_state.json").exists()


def test_webhook_success_uses_json_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(notify, "_today", lambda: "2026-01-01")
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _open(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", _open)
    notify.send_alerts(["IC 退化"], "夜间告警")
    request, timeout = calls[0]
    assert timeout == 5
    assert "IC 退化" in json.loads(request.data.decode("utf-8"))["text"]
    assert (tmp_path / "alerts_state.json").exists()


def test_preflight_failure_sends_alert(monkeypatch, tmp_path):
    import pandas as pd
    import predict_tomorrow as pt

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["A", "B"],
            C.TARGET: [0.01, 0.02],
        }
    )
    monkeypatch.setattr(C, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(C, "USE_PREFLIGHT", True)
    monkeypatch.setattr(pt, "load_live_panel", lambda **kwargs: (panel, []))
    calls = []
    monkeypatch.setattr(notify, "send_alerts", lambda alerts, title: calls.append((alerts, title)))
    with pytest.raises(SystemExit) as exc_info:
        pt.main(["--account", str(tmp_path / "missing.csv")])
    assert exc_info.value.code == 3
    assert calls and "前置体检" in calls[0][1]


def test_unexpected_exception_sends_alert_and_reraises(monkeypatch):
    import predict_tomorrow as pt

    def _fail(argv):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(pt, "_main", _fail)
    calls = []
    monkeypatch.setattr(notify, "send_alerts", lambda alerts, title: calls.append((alerts, title)))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        pt.main([])
    assert calls and "异常" in calls[0][1]
    assert "synthetic failure" in calls[0][0][0]
