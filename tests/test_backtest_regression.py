"""快速回归测试：用本地缓存的真实因子面板跑 run_daily_replay（strict_check 全开）。

此前本文件的 main() 跑全市场缓存面板，600s 超时无输出，pytest 零收集。
现在改为：
  - 直接读 data_cache/wholemkt_panel.parquet（研究脚本同款缓存），缺失即 skip；
  - 只取尾部 ~120 个自然日 × 尾部覆盖最全的 25 只股票，交易列（OHLC 等）
    由 close 合成（本测试回归的是"模型+决策+撮合+守恒"链路，不是成交价真实性，
    成交价真实性由 test_fills_realism.py 的逐笔测试覆盖）；
  - 重放窗口缩到 ~10 个决策日（start_date 截尾），秒级完成。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import config as C

PANEL_CACHE = os.path.join(os.path.dirname(__file__), "..", "data_cache",
                           "wholemkt_panel.parquet")

FEAT_COLS = ["momentum_5", "momentum_20", "volatility_20", "volume_ma_ratio_20",
             "hl_pct", "amihud_20", "max_ret_20", "a158_corr5"]

_LOAD_COLS = (["date", "symbol", "close", "amount", "pct_chg", "return_1d",
               "rsi_14", "limit_flag", C.TARGET] + FEAT_COLS)

N_SYMBOLS = 25
LOOKBACK_DAYS = 120      # 读盘窗口（自然日）
REPLAY_DAYS = 15         # 重放窗口（自然日，~10 个决策日）


def _load_tail_panel() -> pd.DataFrame:
    """读缓存面板尾部切片，并合成交易列。"""
    df = pd.read_parquet(PANEL_CACHE, columns=_LOAD_COLS)
    latest = df["date"].max()
    df = df[df["date"] >= latest - pd.Timedelta(days=LOOKBACK_DAYS)]
    # 尾部窗口内覆盖最全的 N 只股票（保证每天有完整横截面）
    counts = df.groupby("symbol")["date"].nunique()
    keep = counts.sort_values(ascending=False).head(N_SYMBOLS).index
    df = df[df["symbol"].isin(keep)].copy()
    df["symbol"] = df["symbol"].astype(str)

    # 交易列合成（缓存面板不含 OHLC/pre_close/布林带）：open=close、±1% 高低、
    # pre_close 由 pct_chg 反推，使每日前后价链一致。
    df["adj_close"] = df["close"]
    df["open"] = df["close"]
    df["high"] = df["close"] * 1.01
    df["low"] = df["close"] * 0.99
    df["pre_close"] = df["close"] / (1.0 + df["pct_chg"].fillna(0.0) / 100.0)
    df["bb_upper_20"] = df["close"] * 1.05
    df["bb_lower_20"] = df["close"] * 0.95
    df[C.WEIGHT] = 1.0
    return df


def test_daily_replay_on_cached_panel():
    if not os.path.exists(PANEL_CACHE):
        pytest.skip(f"本地缓存缺失：{PANEL_CACHE}（真实数据回归测试跳过）")

    from backtest_daily import run_daily_replay

    panel = _load_tail_panel()
    latest = panel["date"].max()
    start = latest - pd.Timedelta(days=REPLAY_DAYS)

    res = run_daily_replay(
        panel, FEAT_COLS,
        start_date=start.date(), end_date=latest.date(),
        params={"n_seeds": 1, "rebuild_days": 25, "train_window": 40,
                "warmup_days": 40, "ic_window": 5,
                "strict_check": True, "progress": False},
    )
    daily = res["daily"]
    trades = res["trades"]

    # ~10 个决策日；太少说明窗口切错了
    assert 3 <= len(daily) <= REPLAY_DAYS, f"决策日数量异常: {len(daily)}"
    assert np.isfinite(daily["total_value"]).all()
    assert (daily["total_value"] > 0).all()
    assert np.allclose(daily["total_value"],
                       daily["cash"] + daily["stock_value"], atol=1.0)
    assert (daily["cost"] >= 0).all()
    assert (daily["turnover"] >= 0).all()
    assert isinstance(trades, pd.DataFrame)


if __name__ == "__main__":
    test_daily_replay_on_cached_panel()
    print("cached-panel regression test passed")
