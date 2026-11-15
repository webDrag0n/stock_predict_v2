from __future__ import annotations

import numpy as np
import pandas as pd


def _return_column(panel: pd.DataFrame) -> tuple[str, float]:
    if "return_1d" in panel.columns:
        return "return_1d", 1.0
    if "pct_chg" in panel.columns:
        return "pct_chg", 0.01
    raise ValueError("归因面板需要 return_1d 或 pct_chg 列")


def small_cap_benchmark(panel: pd.DataFrame, quantile: float = 0.2) -> pd.Series:
    if not 0 < quantile <= 1:
        raise ValueError("quantile 必须在 (0, 1] 内")
    ret_col, scale = _return_column(panel)
    columns = ["date", "symbol", "ln_mv", ret_col]
    frame = panel[columns].copy()
    frame[ret_col] = pd.to_numeric(frame[ret_col], errors="coerce") * scale
    frame["ln_mv"] = pd.to_numeric(frame["ln_mv"], errors="coerce")
    frame = frame.dropna(subset=["date", "ln_mv", ret_col])
    # 防泄露：每日规模分组和收益平均只使用当日截面。
    rank = frame.groupby("date")["ln_mv"].rank(method="first", pct=True)
    selected = frame[rank <= quantile]
    return selected.groupby("date")[ret_col].mean().sort_index().rename("smallcap_ret")


def size_exposure(strat_ret: pd.Series, panel: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    window = int(window)
    if window <= 0:
        raise ValueError("window 必须为正整数")
    ret_col, scale = _return_column(panel)
    market = panel.groupby("date")[ret_col].mean().sort_index() * scale
    small = small_cap_benchmark(panel)
    strategy = pd.Series(strat_ret).copy()
    strategy.index = pd.to_datetime(strategy.index)
    strategy.name = "strat"
    joined = pd.concat(
        [
            market.rename("mkt"),
            (small - market).rename("size"),
            strategy,
        ],
        axis=1,
        join="inner",
    ).dropna()
    joined = joined.sort_index()

    rows = []
    # 防泄露：每个 t 的 OLS 只使用 t-window+1..t 的已实现收益。
    for i in range(len(joined)):
        row = {"date": joined.index[i]}
        if i + 1 < window:
            row.update({"beta_mkt": np.nan, "beta_size": np.nan, "alpha_daily": np.nan})
        else:
            sample = joined.iloc[i + 1 - window : i + 1]
            x = np.column_stack(
                [
                    np.ones(window),
                    sample["mkt"].to_numpy(dtype=float),
                    sample["size"].to_numpy(dtype=float),
                ]
            )
            y = sample["strat"].to_numpy(dtype=float)
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            row.update(
                {
                    "alpha_daily": float(coefficients[0]),
                    "beta_mkt": float(coefficients[1]),
                    "beta_size": float(coefficients[2]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", "beta_mkt", "beta_size", "alpha_daily"])
