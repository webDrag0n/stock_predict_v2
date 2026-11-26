"""Capital-aware trading presets shared by CLI, pipeline and web entry points."""

from __future__ import annotations

import math


def economic_min_notional(max_commission_rate: float = 0.00125) -> float:
    """Return the order size at which the minimum commission stays below max_commission_rate."""
    import config as C

    minimum = float(getattr(C, "MIN_COMMISSION", 0.0))
    rate = float(max_commission_rate)
    if minimum <= 0 or rate <= 0:
        return 0.0
    return float(math.ceil((minimum / rate) / 100.0) * 100)


def capital_profile_params(profile: str | None = None) -> dict:
    """Return explicit trading overrides for a named capital profile.

    The standard profile is intentionally empty so existing configurations retain
    their current behavior. Small accounts use fewer, larger positions and enable
    estimated cash-budget checks before orders are written.
    """
    name = str(profile or "standard").strip().lower()
    if name in ("", "standard", "normal"):
        return {}
    if name != "small":
        raise ValueError(f"未知资金配置: {profile!r}（可选 standard / small）")
    minimum = economic_min_notional()
    import config as C

    return {
        "max_positions": 4,
        "max_weight": 0.25,
        "min_notional": minimum,
        "min_entry_notional": minimum,
        "cash_aware": True,
        "cash_buffer": max(50.0, 10.0 * float(getattr(C, "MIN_COMMISSION", 0.0))),
        "skip_unaffordable": True,
    }


def with_capital_profile(
    profile: str | None, explicit: dict | None = None, *, ignore_standard_defaults: bool = False
) -> dict:
    """Merge profile defaults with non-None user overrides.

    Pipeline forms always submit their standard defaults. When requested, those
    unchanged values are treated as absent so they do not mask a small-profile preset.
    """
    values = dict(explicit or {})
    if ignore_standard_defaults and str(profile or "standard").lower() == "small":
        import config as C

        defaults = {
            "max_positions": C.MAX_POSITIONS,
            "max_weight": C.MAX_WEIGHT_PER_NAME,
            "min_notional": C.DAILY_MIN_NOTIONAL,
        }
        values = {
            key: value
            for key, value in values.items()
            if key not in defaults or value != defaults[key]
        }
    out = capital_profile_params(profile)
    for key, value in values.items():
        if value is not None:
            out[key] = value
    return out
