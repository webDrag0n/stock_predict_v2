from __future__ import annotations

import importlib

import config as C
import pipeline.registry as registry

CONFIG_KEYS = {
    "use_bb_timing": "USE_BB_TIMING",
    "order_price_mode": "ORDER_PRICE_MODE",
    "min_notional": "DAILY_MIN_NOTIONAL",
    "topk": "TOPK_RATIO",
    "max_positions": "MAX_POSITIONS",
    "max_weight": "MAX_WEIGHT_PER_NAME",
    "use_dynamic_topk": "USE_DYNAMIC_TOPK",
    "use_beta_constraint": "USE_BETA_CONSTRAINT",
    "train_window": "TRAIN_WINDOW",
    "rebuild_days": "REBUILD_DAYS",
    "incr_trees": "INCR_TREES",
    "ic_window": "IC_WINDOW",
    "n_seeds": "ENSEMBLE_N_SEEDS",
}


def test_nextday_defaults_match_config():
    defaults = registry.default_params("nextday")
    for key, config_key in CONFIG_KEYS.items():
        assert defaults[key] == getattr(C, config_key), key
        assert isinstance(defaults[key], type(getattr(C, config_key))), key


def test_registry_defaults_refresh_on_reload(monkeypatch):
    monkeypatch.setattr(C, "USE_BB_TIMING", not C.USE_BB_TIMING)
    expected = C.USE_BB_TIMING
    try:
        importlib.reload(registry)
        assert registry.default_params("nextday")["use_bb_timing"] is expected
    finally:
        monkeypatch.undo()
        importlib.reload(registry)
