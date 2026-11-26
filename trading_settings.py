"""Persistent per-user trading costs and fill constraints shared by CLI and webapp."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS = {
    "commission_rate": 0.0002,
    "stamp_duty_rate": 0.001,
    "min_commission": 5.0,
    "impact_k": 0.1,
    "use_impact_cost": True,
    "max_participation": 0.10,
}
_LIMITS = {
    "commission_rate": (0.0, 0.1),
    "stamp_duty_rate": (0.0, 0.1),
    "min_commission": (0.0, 100000.0),
    "impact_k": (0.0, 100.0),
    "max_participation": (0.0, 1.0),
}


def settings_path() -> str:
    return os.environ.get("TRADING_SETTINGS_FILE", str(ROOT / "outputs" / "trading_settings.json"))


def validate_settings(values: dict) -> dict:
    if not isinstance(values, dict):
        raise ValueError("交易设置必须是 JSON 对象")
    unknown = sorted(set(values) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"未知交易设置：{', '.join(unknown)}")
    clean = {}
    for key, value in values.items():
        if key == "use_impact_cost":
            if not isinstance(value, bool):
                raise ValueError("use_impact_cost 必须是布尔值")
            clean[key] = value
            continue
        if isinstance(value, bool):
            raise ValueError(f"{key} 必须是有限数字")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是有限数字") from exc
        if not math.isfinite(number):
            raise ValueError(f"{key} 必须是有限数字")
        low, high = _LIMITS[key]
        if not low <= number <= high:
            raise ValueError(f"{key} 必须在 [{low}, {high}] 内")
        clean[key] = number
    return clean


def load_trading_settings(path: str | None = None) -> dict:
    target = path or settings_path()
    if not os.path.exists(target):
        return dict(DEFAULT_SETTINGS)
    try:
        with open(target, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        raise ValueError(f"交易设置文件无法读取：{target}: {exc}") from exc
    clean = validate_settings(raw)
    return {**DEFAULT_SETTINGS, **clean}


def save_trading_settings(values: dict, path: str | None = None) -> dict:
    """合并并原子写入设置，返回完整生效配置；环境变量覆盖在 config 中处理。"""
    target = path or settings_path()
    clean = validate_settings(values)
    merged = {**load_trading_settings(target), **clean}
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    tmp = f"{target}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return merged
