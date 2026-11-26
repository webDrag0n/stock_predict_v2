from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import trading_settings as ts


def test_missing_file_uses_documented_defaults(tmp_path):
    values = ts.load_trading_settings(str(tmp_path / "missing.json"))
    assert values == ts.DEFAULT_SETTINGS
    assert values["stamp_duty_rate"] == pytest.approx(0.001)


def test_save_merges_and_persists_all_settings(tmp_path):
    path = str(tmp_path / "trading_settings.json")
    saved = ts.save_trading_settings({"stamp_duty_rate": 0.0005, "use_impact_cost": False}, path)
    assert saved["stamp_duty_rate"] == pytest.approx(0.0005)
    assert saved["commission_rate"] == pytest.approx(0.0002)
    assert not saved["use_impact_cost"]
    assert ts.load_trading_settings(path) == saved
    assert json.loads(Path(path).read_text())["stamp_duty_rate"] == pytest.approx(0.0005)


@pytest.mark.parametrize(
    "values, match",
    [
        ({"unknown": 1}, "未知交易设置"),
        ({"stamp_duty_rate": -0.1}, "必须在"),
        ({"stamp_duty_rate": 0.2}, "必须在"),
        ({"commission_rate": float("nan")}, "有限数字"),
        ({"min_commission": True}, "有限数字"),
        ({"use_impact_cost": 0}, "布尔值"),
    ],
)
def test_invalid_settings_are_rejected(tmp_path, values, match):
    with pytest.raises(ValueError, match=match):
        ts.save_trading_settings(values, str(tmp_path / "settings.json"))
    assert not (tmp_path / "settings.json").exists()


def test_corrupt_settings_fail_closed(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{broken")
    with pytest.raises(ValueError, match="无法读取"):
        ts.load_trading_settings(str(path))


def test_new_process_reads_settings_and_env_still_wins(tmp_path):
    path = tmp_path / "settings.json"
    ts.save_trading_settings(
        {
            "commission_rate": 0.0003,
            "stamp_duty_rate": 0.0005,
            "min_commission": 0.0,
            "use_impact_cost": False,
        },
        str(path),
    )
    code = """
import json
import config as C
from trading import trade_costs
commission, stamp, impact = trade_costs(10000.0, 'sell')
print(json.dumps({
    'commission_rate': C.COMMISSION_RATE,
    'stamp_duty_rate': C.STAMP_DUTY_RATE,
    'use_impact': C.USE_IMPACT_COST,
    'commission': float(commission),
    'stamp': float(stamp),
    'impact': float(impact),
}))
"""
    env = os.environ.copy()
    env["TRADING_SETTINGS_FILE"] = str(path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parents[1]), str(Path(__file__).parents[1] / "src")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    values = json.loads(result.stdout)
    assert values["commission_rate"] == pytest.approx(0.0003)
    assert values["stamp_duty_rate"] == pytest.approx(0.0005)
    assert values["commission"] == pytest.approx(3.0)
    assert values["stamp"] == pytest.approx(5.0)
    assert values["impact"] == 0.0 and not values["use_impact"]

    env["STAMP_DUTY_RATE"] = "0.002"
    overridden = subprocess.run(
        [sys.executable, "-c", "import config as C; print(C.STAMP_DUTY_RATE)"],
        cwd=Path(__file__).parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert float(overridden.stdout) == pytest.approx(0.002)
