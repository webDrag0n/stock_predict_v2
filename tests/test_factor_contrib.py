"""src/factor_contrib.py 单元测试：假 ensemble state → 贡献长表提取。

路径由 tests/conftest.py 统一处理（项目根 + src 均已入 sys.path）。
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

import factor_contrib


def _fake_state() -> dict:
    return {
        "models": ("ic_linear", "gbm_l2"),
        "weights": {"ic_linear": 0.6, "gbm_l2": 0.4},
        "linear_state": {
            "feats": ["mom_20", "nl_mom_20"],
            "weights": np.array([0.25, -0.10]),
        },
        "boosters": {},
        "last_trained_date": "2026-10-20",
    }


def test_contrib_from_state_linear_only():
    df = factor_contrib.contrib_from_state(_fake_state(), ["mom_20"])
    assert list(df.columns) == factor_contrib.COLUMNS
    assert set(df["source"]) == {"ic_linear"}
    assert set(df["kind"]) == {"ic_weight"}
    vals = df.set_index("factor")["value"]
    assert vals["mom_20"] == 0.25
    assert vals["nl_mom_20"] == -0.10
    assert (df["asof"] == "2026-10-20").all()


def test_contrib_from_state_linear_dict_weights():
    """linear_fit 的 weights 是 feat→weight dict（当前真实格式）；
    按位置 zip 会迭代到键名（float('amount') 崩溃的回归）。"""
    state = _fake_state()
    state["linear_state"] = {
        "feats": ["mom_20", "nl_mom_20"],
        "weights": {"mom_20": 0.25, "nl_mom_20": -0.10},
    }
    df = factor_contrib.contrib_from_state(state, ["mom_20"])
    vals = df.set_index("factor")["value"]
    assert vals["mom_20"] == 0.25
    assert vals["nl_mom_20"] == -0.10
    # dict 缺 feats 里的键时跳过而不是错位
    state["linear_state"]["weights"] = {"mom_20": 0.5}
    df2 = factor_contrib.contrib_from_state(state, ["mom_20"])
    assert list(df2["factor"]) == ["mom_20"] and df2["value"].iloc[0] == 0.5


def test_contrib_from_state_gbm_importance():
    import lightgbm as lgb

    rng = np.random.RandomState(0)
    X = rng.randn(120, 3)
    y = X[:, 0] * 2 + rng.randn(120) * 0.1
    bst = lgb.train(
        {"objective": "regression", "num_leaves": 7, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=10,
    )
    state = _fake_state()
    state["boosters"] = {"gbm_l2": [bst.model_to_string()]}
    feat_cols = ["f_a", "f_b", "f_c"]
    df = factor_contrib.contrib_from_state(state, feat_cols)
    gbm = df[df["source"] == "gbm_l2"]
    assert not gbm.empty
    assert set(gbm["kind"]) == {"gbm_importance"}
    # 腿内归一：重要性和为 1
    assert abs(gbm["value"].sum() - 1.0) < 1e-9
    # Column_N 已按位置映射回因子名
    assert set(gbm["factor"]) <= set(feat_cols)


def test_contrib_from_state_bad_inputs():
    assert factor_contrib.contrib_from_state(None).empty
    assert factor_contrib.contrib_from_state({}).empty
    # 损坏的 booster 文本被跳过，不影响 ic_linear 行
    state = _fake_state()
    state["boosters"] = {"gbm_l2": ["not-a-model"]}
    df = factor_contrib.contrib_from_state(state, [])
    assert set(df["source"]) == {"ic_linear"}


def test_combined_scores_weights_legs():
    df = factor_contrib.contrib_from_state(_fake_state(), [])
    scores = factor_contrib.combined_scores(df, {"ic_linear": 1.0, "gbm_l2": 0.0})
    # 仅一条腿：腿内归一 |w| → mom_20 占比 0.25/(0.25+0.10)
    assert abs(scores["mom_20"] - 0.25 / 0.35) < 1e-9
    # 腿权为 0 时全零
    zeros = factor_contrib.combined_scores(df, {"ic_linear": 0.0})
    assert (zeros == 0).all()
    assert factor_contrib.combined_scores(pd.DataFrame()).empty


def test_load_champion_contrib_roundtrip(tmp_path):
    champ = tmp_path / "champion"
    champ.mkdir()
    (champ / "manifest.json").write_text(
        json.dumps({"config_snapshot": {"feat_cols": ["mom_20"]}, "train_end_date": "2026-10-21"}),
        encoding="utf-8",
    )
    with open(champ / "ensemble_state.pkl", "wb") as f:
        pickle.dump(_fake_state(), f)

    df = factor_contrib.load_champion_contrib(str(tmp_path))
    assert not df.empty
    assert (df["asof"] == "2026-10-21").all()
    assert factor_contrib.champion_leg_weights(str(tmp_path)) == {
        "ic_linear": 0.6,
        "gbm_l2": 0.4,
    }


def test_load_champion_contrib_missing_and_corrupt(tmp_path):
    # champion 不存在 → 空表
    assert factor_contrib.load_champion_contrib(str(tmp_path)).empty
    assert factor_contrib.champion_leg_weights(str(tmp_path)) == {}
    # 损坏的 pkl → 空表（不抛异常）
    champ = tmp_path / "champion"
    champ.mkdir()
    (champ / "manifest.json").write_text("{}", encoding="utf-8")
    (champ / "ensemble_state.pkl").write_bytes(b"garbage")
    assert factor_contrib.load_champion_contrib(str(tmp_path)).empty


def test_load_session_leg_weights(tmp_path):
    art = tmp_path / "sess1" / "artifacts"
    art.mkdir(parents=True)
    wide = pd.DataFrame(
        {
            "date": ["2026-10-19", "2026-10-20"],
            "w_ic_linear": [0.5, 0.7],
            "w_gbm_l2": [0.5, 0.3],
            "ic_ic_linear": [0.02, 0.03],
            "ic_gbm_l2": [0.01, -0.01],
        }
    )
    wide.to_parquet(art / "model_weights.parquet", index=False)
    long = factor_contrib.load_session_leg_weights(str(tmp_path))
    assert len(long) == 4
    assert set(long["model"]) == {"ic_linear", "gbm_l2"}
    row = long[(long["date"] == "2026-10-20") & (long["model"] == "ic_linear")].iloc[0]
    assert row["weight"] == 0.7 and row["ic"] == 0.03
    # 目录不存在 → 空表
    assert factor_contrib.load_session_leg_weights(os.path.join(str(tmp_path), "nope")).empty
