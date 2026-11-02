"""GTJA191 向量化 ts_rank/ts_corr 与 _base 参考实现的逐位等价回归测试。

gtja191 的 ts_rank/ts_corr 是 _base 同名原语的向量化替代（复现 pandas
rolling.rank 与 Kahan rolling sum 的精确语义，含 NaN 与停牌行缺口）。若未来
有人改动其中一侧，本测试保证两者仍然 bit-exact——下游 cs_rank 对退化窗口
（corr ≈ ±1 的并列簇）的排序对 ulp 级差异敏感，容差必须为零。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.factors import _base, gtja191


def _panel(n_sym: int = 40, n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """合成 OHLCV 面板：注入 NaN 单元格并随机丢行（模拟停牌日期缺口）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    frames = []
    for i in range(n_sym):
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": f"{i:06d}",
                    "date": dates,
                    "open": close * (1 + rng.normal(0, 0.005, n_days)),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.lognormal(15, 0.5, n_days),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    idx = rng.choice(n, int(n * 0.05), replace=False)
    cols = rng.choice(["open", "close", "volume"], len(idx))
    for c in ["open", "close", "volume"]:
        df.loc[idx[cols == c], c] = np.nan
    df = df.drop(index=rng.choice(n, int(n * 0.05), replace=False))
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _assert_bitexact(ref: pd.Series, got: pd.Series):
    assert ref.index.equals(got.index)
    r, g = ref.to_numpy(), got.to_numpy()
    assert np.array_equal(np.isnan(r), np.isnan(g)), "NaN 掩码不一致"
    assert np.array_equal(np.nan_to_num(r), np.nan_to_num(g)), "取值非 bit-exact"


@pytest.mark.parametrize("n", [1, 3, 5, 6, 18])
def test_ts_rank_matches_base(n):
    idx = _panel().set_index(["symbol", "date"])
    _assert_bitexact(_base.ts_rank(idx["close"], n), gtja191.ts_rank(idx["close"], n))
    _assert_bitexact(_base.ts_rank(idx["volume"], n), gtja191.ts_rank(idx["volume"], n))


@pytest.mark.parametrize("n", [2, 3, 5, 15, 18])
def test_ts_corr_matches_base(n):
    idx = _panel().set_index(["symbol", "date"])
    ref = _base.ts_corr(idx["close"], idx["volume"], n)
    got = gtja191.ts_corr(idx["close"], idx["volume"], n)
    _assert_bitexact(ref, got)


def test_compute_columns_unchanged():
    """compute_gtja191_factors 的输出列集不因向量化而改变。"""
    out = gtja191.compute_gtja191_factors(_panel())
    expected = {f"gtja_{aid}" for aid in gtja191.DEFAULT_SUBSET}
    assert expected <= set(out.columns)
    assert out[[f"gtja_{aid}" for aid in gtja191.DEFAULT_SUBSET]].dtypes.eq(np.float32).all()
