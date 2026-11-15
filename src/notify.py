from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime

import config as C

logger = logging.getLogger(__name__)


def _today() -> str:
    return datetime.now().date().isoformat()


def _state_path() -> str:
    return os.path.join(C.OUTPUT_DIR, "alerts_state.json")


def _load_state() -> dict:
    path = _state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    path = _state_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _send_alerts(alerts: list[str], title: str) -> None:
    text = title + "\n" + "\n".join(str(alert) for alert in alerts)
    today = _today()
    state = _load_state()
    previous = state.get(text)
    if previous:
        try:
            if (
                datetime.fromisoformat(today).date() - datetime.fromisoformat(previous).date()
            ).days < 3:
                return
        except ValueError:
            pass

    webhook = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if webhook:
        request = urllib.request.Request(
            webhook,
            data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5):
            pass
    else:
        path = os.path.join(C.OUTPUT_DIR, "alerts.log")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            record = {"ts": datetime.now().isoformat(timespec="seconds"), "text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    state[text] = today
    _save_state(state)


def send_alerts(alerts: list[str], title: str) -> None:
    if not alerts:
        return
    try:
        _send_alerts(alerts, title)
    except Exception as exc:
        logger.warning("告警发送失败（不影响主流程）：%s: %s", type(exc).__name__, exc)
