"""src/factor_layers.py 单元测试：分层净值 / 逐日 RankIC 落盘契约。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as C
import factor_layers


def _panel(n_dates: int = 40, n_syms: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(3)
    dates = pd.date_range("2026-06-01", periods=n_dates, freq="B")
    rows = []
    for d in dates:
        f = rng.randn(n_syms)
        rows.append(
            pd.DataFrame(
                {
                    "date": d,
                    "symbol": [f"S{i:03d}" for i in range(n_syms)],
                    "my_factor": f,
                    # 让因子与目标有正相关，保证分层/IC 非退化
                    C.TARGET: f * 0.01 + rng.randn(n_syms) * 0.02,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_safe_factor_id():
    assert factor_layers.safe_factor_id("a158_kmid") == "a158_kmid"
    assert factor_layers.safe_factor_id("a/b c:d") == "a_b_c_d"
    assert factor_layers.safe_factor_id("***") == "factor"


def test_dump_factor_layer_writes_both_parquets(tmp_path):
    panel = _panel()
    res = factor_layers.dump_factor_layer(panel, "my_factor", str(tmp_path), n_layers=5)
    assert res["ok"], res["error"]
    assert res["n_dates"] > 0

    layer = pd.read_parquet(tmp_path / "factor_layer_my_factor.parquet")
    assert list(layer.columns) == ["date", "layer", "equity"]
    # 5 层，标签 L1..L5（L1=因子最小层）
    assert sorted(layer["layer"].unique()) == ["L1", "L2", "L3", "L4", "L5"]
    # 净值从 ~1 起步且为正
    assert (layer["equity"] > 0).all()

    ic = pd.read_parquet(tmp_path / "factor_ic_daily_my_factor.parquet")
    assert list(ic.columns) == ["date", "rank_ic"]
    assert ic["rank_ic"].between(-1, 1).all()
    # 构造的数据有正相关，平均 RankIC 应显著为正
    assert ic["rank_ic"].mean() > 0.2


def test_dump_factor_layer_missing_factor(tmp_path):
    res = factor_layers.dump_factor_layer(_panel(), "nope", str(tmp_path))
    assert not res["ok"]
    assert "nope" in res["error"]


def test_dump_layers_summary(tmp_path):
    panel = _panel()
    panel["other"] = panel["my_factor"] * -1
    summary = factor_layers.dump_layers(panel, ["my_factor", "other", "ghost"], str(tmp_path))
    assert list(summary.columns) == factor_layers._SUMMARY_COLS
    assert len(summary) == 3
    assert summary["ok"].tolist() == [True, True, False]
    assert summary.loc[summary["factor"] == "ghost", "error"].iloc[0] != ""


def test_dump_layers_empty_panel(tmp_path):
    empty = pd.DataFrame(columns=["date", "symbol", "f", C.TARGET])
    res = factor_layers.dump_factor_layer(empty, "f", str(tmp_path))
    assert not res["ok"]


@pytest.mark.parametrize("bad", ["a/b", "x y"])
def test_path_uses_safe_id(bad):
    p = factor_layers.layer_equity_path("/tmp/out", bad)
    assert "/" not in p.split("/")[-1].replace("factor_layer_", "").replace(".parquet", "")
