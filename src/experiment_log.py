from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime

from scipy.stats import norm

import config as C


def _trials_path(path=None) -> str:
    return (
        os.fspath(path)
        if path is not None
        else os.path.join(C.OUTPUT_DIR, "experiments", "trials.jsonl")
    )


def _config_hash(config_dict: dict) -> str:
    payload = json.dumps(config_dict, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def log_trial(kind, config_dict, metrics_dict, path=None) -> dict:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": str(kind),
        "config": config_dict,
        "metrics": metrics_dict,
        "config_hash": _config_hash(config_dict),
    }
    destination = _trials_path(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    with open(destination, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def count_trials(kind=None, path=None) -> int:
    destination = _trials_path(path)
    if not os.path.exists(destination):
        return 0
    hashes = set()
    with open(destination, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if kind is not None and record.get("kind") != kind:
                continue
            config_hash = record.get("config_hash")
            if config_hash is not None:
                hashes.add(config_hash)
    return len(hashes)


def deflated_sharpe(
    sr_daily,
    n_obs,
    n_trials,
    skew=0.0,
    kurt=3.0,
    sr_std_daily=None,
) -> float:
    """返回考虑多重试验后真实 Sharpe 大于 0 的概率。

    输入 sr_daily 和 sr_std_daily 均为日度 Sharpe（年化 Sharpe 除以 sqrt(252)）。
    n_trials<=1 时 sr0=0；否则 sr0 = sr_std_daily × [(1-γ)×Z(1-1/N) +
    γ×Z(1-1/(N×e))]，γ=0.5772156649，sr_std_daily 缺省为
    1/sqrt(n_obs-1)。令 den = sqrt(1-skew×sr_daily +
    (kurt-1)/4×sr_daily²)，最终返回 Φ((sr_daily-sr0)×sqrt(n_obs-1)/den)。
    """
    n_obs = int(n_obs)
    n_trials = int(n_trials)
    if n_obs <= 1:
        return float("nan")
    sr_daily = float(sr_daily)
    if sr_std_daily is None:
        sr_std_daily = 1.0 / math.sqrt(n_obs - 1)
    if n_trials <= 1:
        sr0 = 0.0
    else:
        gamma = 0.5772156649
        sr0 = float(sr_std_daily) * (
            (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
            + gamma * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    radicand = 1.0 - float(skew) * sr_daily + (float(kurt) - 1.0) / 4.0 * sr_daily**2
    if radicand <= 0:
        return float("nan")
    return float(norm.cdf((sr_daily - sr0) * math.sqrt(n_obs - 1) / math.sqrt(radicand)))
