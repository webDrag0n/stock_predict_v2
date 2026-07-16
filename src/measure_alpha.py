"""Alpha/beta 显著性检验（Newey-West HAC 标准误）。

核心问题: 策略是真有 alpha, 还是只是吃 beta(牛市裸多)? 用锁定配置跑一次
walkforward, 取连续 OOS 的策略日净收益, 与"等权持有全universe"的日收益做
CAPM 回归: strat_ret = alpha + beta * bench_ret。alpha>0 且显著 = 真主动收益。

周期调仓策略的持仓固定 REBALANCE_DAYS 天，日收益存在自相关；
OLS 标准误会低估不确定性、高估 alpha t 统计量。
Newey-West (Bartlett kernel, `lags` ≈ holding period) 给出异方差-自相关一致的
标准误，是有效判据。
"""

from __future__ import annotations

import numpy as np


def _newey_west_alpha_t(strat: np.ndarray, bench: np.ndarray, lags: int):
    """CAPM regression strat = alpha + beta*bench with Newey-West (HAC) SE on alpha.

    The periodic strategy holds positions fixed for REBALANCE_DAYS, so daily
    returns are autocorrelated; OLS SEs understate uncertainty and inflate the
    alpha t-stat. NW with `lags` ≈ holding period gives a heteroskedasticity- and
    autocorrelation-consistent SE. Returns (alpha_daily, beta, t_alpha_nw, r2).
    """
    n = len(strat)
    X = np.column_stack([np.ones(n), bench])      # [1, bench]
    b, *_ = np.linalg.lstsq(X, strat, rcond=None)
    resid = strat - X @ b
    # NW HAC covariance of OLS coefficients
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])   # lag 0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)                 # Bartlett kernel
        Xe = X * resid[:, None]
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = np.sqrt(cov[0, 0])
    t_alpha = b[0] / se_alpha if se_alpha > 1e-18 else 0.0
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((strat - strat.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    return float(b[0]), float(b[1]), float(t_alpha), float(r2)
