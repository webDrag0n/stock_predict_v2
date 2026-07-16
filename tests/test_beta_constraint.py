"""Unit tests for portfolio beta constraint.

Verifies that _apply_beta_constraint in trading.py reduces portfolio beta by
replacing high-beta stocks with lower-beta candidates.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from trading import _apply_beta_constraint


def test_beta_constraint_reduces_beta():
    """Pool of 4 stocks, beta=[2.0, 1.5, 1.0, 0.5], ranks ascending (0=best).
    Selected pool = best 2 (indices 0,1) with avg beta 1.75. Constraint=1.2.
    Expected: replace index 0 (beta 2.0) with index 3 (beta 0.5), giving pool
    (1,3) avg beta 1.0.
    """
    beta = np.array([2.0, 1.5, 1.0, 0.5])
    rank_pct = np.array([0.0, 0.33, 0.66, 1.0])  # 0=best
    pool_idx = [0, 1]
    adjusted = _apply_beta_constraint(pool_idx, beta, rank_pct, max_portfolio_beta=1.2)
    print("adjusted pool:", adjusted)
    avg_beta = float(np.mean(beta[adjusted]))
    print("avg beta:", avg_beta)
    assert avg_beta <= 1.2
    assert len(adjusted) == len(pool_idx)


def test_beta_constraint_no_change_when_already_low():
    beta = np.array([1.1, 1.0, 0.9, 0.5])
    rank_pct = np.array([0.0, 0.33, 0.66, 1.0])
    pool_idx = [0, 1]
    adjusted = _apply_beta_constraint(pool_idx, beta, rank_pct, max_portfolio_beta=1.2)
    assert adjusted == pool_idx


if __name__ == "__main__":
    test_beta_constraint_reduces_beta()
    test_beta_constraint_no_change_when_already_low()
    print("All beta constraint tests passed.")
