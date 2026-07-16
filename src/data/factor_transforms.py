"""Factor transforms for expanding the linear model's capacity.

ic_linear can only use linear relationships: score = Σ w_i × z_i. Three classes
of transforms give it access to non-linear patterns without GBM-level overfitting:

1. Non-linear unary transforms (per factor, per date):
   z² — parabolic/U-shaped effects (extreme z in EITHER direction matters)
   |z| — pure magnitude (symmetric, sign-agnostic)
   max(z,0) / max(-z,0) — asymmetric (separate up/down tail effects)

2. Pairwise factor interactions (stock × stock, NOT macro × stock):
   z_A × z_B — genuine cross-sectional interactions
   The product has cross-sectional variance because both factors vary across stocks.

3. Factor stability filter:
   Remove factors whose training-segment IC is volatile (high noise, low reliability).

All transforms are leak-free: computed per-date from that day's cross-section only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_factors_by_ic(
    df: pd.DataFrame,
    factor_cols: list[str],
    target_col: str,
    sample_dates: int = 40,
) -> list[tuple[str, float, float]]:
    """Rank factors by absolute mean daily RankIC, using a date sample for speed.

    Returns list of (factor_name, abs_mean_ic, mean_ic) sorted by abs_mean_ic.
    """
    from scipy import stats

    dates_sorted = sorted(df["date"].unique())
    if len(dates_sorted) > sample_dates:
        step = max(1, len(dates_sorted) // sample_dates)
        sampled = dates_sorted[::step]
    else:
        sampled = dates_sorted

    rankings = []
    for f in factor_cols:
        if f not in df.columns:
            continue
        day_ics = []
        for d in sampled:
            g = df[df["date"] == d]
            fv = g[f].dropna()
            if fv.nunique() < 2:
                continue
            tv = g.loc[fv.index, target_col]
            valid = tv.notna()
            if valid.sum() < 20:
                continue
            try:
                ic = stats.spearmanr(fv[valid], tv[valid]).correlation
                if not np.isnan(ic):
                    day_ics.append(ic)
            except Exception:
                continue
        if len(day_ics) >= 5:
            mean_ic = float(np.mean(day_ics))
            rankings.append((f, abs(mean_ic), mean_ic))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def add_nonlinear_transforms(
    df: pd.DataFrame,
    factor_cols: list[str],
    top_n: int = 40,
) -> pd.DataFrame:
    """Add per-date non-linear transforms for the top-N factors by absolute IC.

    For each factor in the top-N, adds z², |z|, max(z,0), max(-z,0).
    Uses date-sampled IC computation for speed (avoid O(N_factors × N_dates)).
    """
    import config as C

    if top_n <= 0 or not factor_cols:
        return df

    target_col = getattr(C, "TARGET", "responder_6")
    if target_col not in df.columns:
        return df

    rankings = _rank_factors_by_ic(df, factor_cols, target_col)
    if not rankings:
        return df

    selected = [f for f, _, _ in rankings[:top_n]]

    df = df.copy()
    for f in selected:
        grp = df.groupby("date", group_keys=False)[f]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, 1e-9)
        z = (df[f] - mu) / sd
        z = z.clip(-4.0, 4.0)

        for suffix, transform in [
            ("_sq", lambda x: x**2),
            ("_abs", lambda x: np.abs(x)),
            ("_pos", lambda x: np.clip(x, 0, None)),
            ("_neg", lambda x: np.clip(-x, 0, None)),
        ]:
            col_name = f"nl_{f}{suffix}"
            if col_name not in df.columns:
                df[col_name] = transform(z)

    return df


def add_factor_interactions(
    df: pd.DataFrame,
    factor_cols: list[str],
    top_n: int = 15,
    max_pairs: int = 60,
) -> pd.DataFrame:
    """Add pairwise factor interactions: z_A × z_B for top-N factor pairs.

    Factors are ranked by abs(mean daily RankIC) using date sampling. The top-N
    are paired (all pairs among top-N, max_pairs limit). Each interaction is the
    product of two per-date z-scores — genuine cross-sectional STOCK×STOCK feature.
    """
    import config as C

    if top_n <= 0 or not factor_cols:
        return df

    target_col = getattr(C, "TARGET", "responder_6")
    if target_col not in df.columns:
        return df

    rankings = _rank_factors_by_ic(df, factor_cols, target_col)
    if len(rankings) < 2:
        return df

    selected = [f for f, _, _ in rankings[:min(top_n, len(rankings))]]

    # Pre-compute per-date z-scores
    z_scores = {}
    for f in selected:
        grp = df.groupby("date", group_keys=False)[f]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, 1e-9)
        z_scores[f] = (df[f] - mu) / sd

    # Generate pairwise interactions
    df = df.copy()
    added = 0
    for i, fa in enumerate(selected):
        za = z_scores[fa].clip(-4.0, 4.0)
        for fb in selected[i + 1 :]:
            if added >= max_pairs:
                break
            col_name = f"ix_{fa}__x__{fb}"
            if col_name not in df.columns:
                df[col_name] = za * z_scores[fb].clip(-4.0, 4.0)
                added += 1
        if added >= max_pairs:
            break

    return df


def filter_stable_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    min_stability: float = 0.3,
    min_ic_days: int = 20,
) -> list[str]:
    """Return factors whose training-segment IC is sufficiently stable.

    Stability = mean(daily_IC) / std(daily_IC) — the "IC t-stat" of each factor.
    Factors with low stability have high IC variance across days (noisy signal)
    and are filtered out. Only factors with at least min_ic_days valid IC
    observations are considered.

    This is intended to be called per walk-forward window to filter factors
    BEFORE training ic_linear — removing noisy factors reduces overfitting.
    """
    import config as C
    from scipy import stats

    target_col = getattr(C, "TARGET", "responder_6")
    if target_col not in df.columns:
        return list(factor_cols)

    kept = []
    for f in factor_cols:
        if f not in df.columns:
            continue
        day_ics = df.groupby("date", group_keys=False).apply(
            lambda g, fac=f: (
                stats.spearmanr(
                    g[fac].dropna(),
                    g.loc[g[fac].notna(), target_col],
                ).correlation
                if g[fac].notna().sum() >= 20
                else np.nan
            ),
            include_groups=False,
        )
        day_ics = day_ics.dropna()
        if len(day_ics) < min_ic_days:
            continue
        mean_ic = day_ics.mean()
        std_ic = day_ics.std()
        if std_ic > 0:
            stability = abs(mean_ic) / std_ic
            if stability >= min_stability:
                kept.append(f)

    return kept
