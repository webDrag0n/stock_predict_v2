"""merge 键 dtype 对齐的回归测试（_align_merge_key）。

全市场面板为省内存把 symbol 存成 category（load_factor_panel），而连接器
侧表（akshare 估值、tushare 财务、hk_hold）的 symbol 是 str/StringDtype。
pandas 的 merge_asof 要求 by 键 dtype 完全一致，categorical vs string 直接
MergeError —— 2026-10-20 全市场面板构建因此崩在 merge_fundamentals。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import _align_merge_key, merge_financial_factors, merge_fundamentals


def _panel(symbol_dtype: str = "category") -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "symbol": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "close": np.arange(6, dtype=float) + 10,
        }
    )
    if symbol_dtype == "category":
        df["symbol"] = df["symbol"].astype("category")
    return df


def _fund(symbol_dtype: str = "string") -> pd.DataFrame:
    fund = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["000001.SZ", "000002.SZ"],
            "pe_ttm": [10.0, 20.0],
            "pb": [1.0, 2.0],
            "ps_ttm": [2.0, 4.0],
            "total_mv": [1.0e10, 2.0e10],
        }
    )
    if symbol_dtype == "string":
        fund["symbol"] = fund["symbol"].astype("string")
    return fund


def test_align_merge_key_categorical_left():
    """category 面板 + string 侧表：侧表转为面板同款 categorical，dtype 一致。"""
    left, right = _align_merge_key(_panel("category"), _fund("string"), "symbol")
    assert isinstance(left["symbol"].dtype, pd.CategoricalDtype)
    assert left["symbol"].dtype == right["symbol"].dtype
    # 面板不被复制转换（大表省内存），只转小表
    assert list(right["symbol"].astype(str)) == ["000001.SZ", "000002.SZ"]


def test_align_merge_key_noop_when_equal():
    left_in, right_in = _panel("object"), _fund("object")
    left, right = _align_merge_key(left_in, right_in, "symbol")
    assert left is left_in and right is right_in


def test_merge_fundamentals_categorical_panel_no_mergeerror():
    """回归：category 面板 merge_asof string 估值表不再 MergeError。

    同时验证防泄露口径不变：估值快照 +1 天可见 —— 快照日（01-02）当天
    看不到该快照，次日（01-03）起可见。
    """
    out = merge_fundamentals(_panel("category"), _fund("string"))
    s1 = out[out["symbol"].astype(str) == "000001.SZ"].sort_values("date")
    assert s1["pe_ttm"].isna().iloc[0]  # 01-02 当天不可见
    assert s1["pe_ttm"].iloc[1] == 10.0  # 01-03 起可见
    assert s1["ep"].iloc[1] == 1.0 / 10.0


def test_merge_financial_factors_categorical_panel_no_mergeerror():
    fin = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": pd.Series(["000001.SZ"], dtype="string"),
            "roe": [0.15],
        }
    )
    out = merge_financial_factors(_panel("category"), fin)
    s1 = out[out["symbol"].astype(str) == "000001.SZ"].sort_values("date")
    assert s1["roe"].iloc[0] == 0.15  # 财务因子无 +1 天 shift，当日可见
    assert s1["roe_rank"].notna().all()
