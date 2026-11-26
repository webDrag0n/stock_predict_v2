"""成交确认写回（src/execution.py）的单元测试。

· save_holdings —— CLI（predict_tomorrow --commit）与新前端「确认成交」共用的写回实现：
  entry_date 保留/刷新、卖出减仓到 0 移除、blocked 单跳过、文件缺失时新建；
· predict_tomorrow 与 execution 为同一实现（CLI/前端行为逐位一致）；
· load_rejected_rids 读取前端写下的「拒绝」勾选。

前端侧的确认/勾选/现金逻辑见 tests/test_webapp2_trade.py 与 tests/test_webapp2_data_actions.py。
"""

from __future__ import annotations


import pandas as pd
import pytest

import execution


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _orders() -> pd.DataFrame:
    """1 卖（减仓）+ 1 买（新仓）+ 1 拦截买。"""
    return pd.DataFrame(
        [
            ["000001.SZ", "sell", 400, 11.0, False],
            ["600000.SH", "buy", 200, 5.0, False],
            ["600001.SH", "buy", 100, 8.0, True],
        ],
        columns=["symbol", "side", "shares", "ref_close", "blocked"],
    )


def _holdings_file(path) -> None:
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "shares": [1000, 500],
            "entry_date": ["2026-10-01", "2026-09-15"],
            "ref_close": [10.0, 20.0],
        }
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# execution.save_holdings：写回语义
# ---------------------------------------------------------------------------
def test_save_holdings_entry_date_and_shares(tmp_path):
    path = str(tmp_path / "holdings.csv")
    _holdings_file(path)
    current = {"000001.SZ": 1000, "000002.SZ": 500}
    execution.save_holdings(path, _orders(), current, "2026-10-27")

    df = execution.load_holdings_table(path).set_index("symbol")
    # 卖出减仓：股数 1000-400=600，entry_date 保留原建仓日，ref_close 取订单价
    assert df.loc["000001.SZ", "shares"] == 600
    assert df.loc["000001.SZ", "entry_date"] == "2026-10-01"
    assert df.loc["000001.SZ", "ref_close"] == pytest.approx(11.0)
    # 未交易持仓：原样保留
    assert df.loc["000002.SZ", "shares"] == 500
    assert df.loc["000002.SZ", "entry_date"] == "2026-09-15"
    assert df.loc["000002.SZ", "ref_close"] == pytest.approx(20.0)
    # 买入新仓：entry_date 刷新为成交确认日
    assert df.loc["600000.SH", "shares"] == 200
    assert df.loc["600000.SH", "entry_date"] == "2026-10-27"
    # 拦截单不成交
    assert "600001.SH" not in df.index


def test_save_holdings_sell_to_zero_removes(tmp_path):
    path = str(tmp_path / "holdings.csv")
    _holdings_file(path)
    orders = pd.DataFrame(
        [["000001.SZ", "sell", 1000, 11.0, False]],
        columns=["symbol", "side", "shares", "ref_close", "blocked"],
    )
    execution.save_holdings(path, orders, {"000001.SZ": 1000, "000002.SZ": 500}, "2026-10-27")
    df = execution.load_holdings_table(path)
    assert list(df["symbol"]) == ["000002.SZ"]


def test_save_holdings_creates_file_from_empty(tmp_path):
    """空仓起点（holdings.csv 不存在）：买入单成交后新建文件。"""
    path = str(tmp_path / "holdings.csv")
    orders = pd.DataFrame(
        [["600000.SH", "buy", 200, 5.0, False]],
        columns=["symbol", "side", "shares", "ref_close", "blocked"],
    )
    execution.save_holdings(path, orders, {}, "2026-10-27")
    df = execution.load_holdings_table(path)
    assert list(df["symbol"]) == ["600000.SH"]
    assert df.iloc[0]["entry_date"] == "2026-10-27"


def test_load_holdings_table_chinese_columns(tmp_path):
    """旧版中文列（代码/股数）兼容。"""
    path = tmp_path / "holdings.csv"
    pd.DataFrame({"代码": ["000001.SZ"], "股数": [1000]}).to_csv(path, index=False)
    df = execution.load_holdings_table(str(path))
    assert list(df.columns) == ["symbol", "shares", "entry_date", "ref_close"]
    assert df.iloc[0]["symbol"] == "000001.SZ" and df.iloc[0]["shares"] == 1000
    # 缺失/损坏 → 空表
    assert execution.load_holdings_table(str(tmp_path / "nope.csv")).empty


def test_cli_shares_same_impl():
    """CLI（predict_tomorrow --commit）与新前端调同一实现，行为逐位一致。"""
    import predict_tomorrow as pt

    assert pt.save_holdings is execution.save_holdings
    assert pt.load_holdings_table is execution.load_holdings_table


def test_load_fills_csv_aggregates_and_validates(tmp_path):
    path = tmp_path / "fills.csv"
    pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH", "000001.SZ"],
            "side": ["buy", "buy", "sell"],
            "shares": [60, 40, 200],
            "price": [4.0, 6.0, 11.0],
        }
    ).to_csv(path, index=False)
    fills = execution.load_fills_csv(str(path))
    buy = fills[(fills["symbol"] == "600000.SH") & (fills["side"] == "buy")].iloc[0]
    assert buy["shares"] == 100
    assert buy["price"] == pytest.approx(4.8)
    assert len(fills) == 2

    pd.DataFrame({"symbol": ["600000.SH"], "side": ["buy"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="shares.*price|price.*shares"):
        execution.load_fills_csv(str(path))


def test_save_holdings_from_fills_partial_reject_and_oversell(tmp_path):
    path = str(tmp_path / "holdings.csv")
    _holdings_file(path)
    current = {"000001.SZ": 1000, "000002.SZ": 500}
    fills = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "600000.SH"],
            "side": ["sell", "buy"],
            "shares": [1200, 100],
            "price": [11.0, 4.8],
        }
    )
    execution.save_holdings_from_fills(path, fills, current, "2026-10-27")

    df = execution.load_holdings_table(path).set_index("symbol")
    assert "000001.SZ" not in df.index
    assert df.loc["000002.SZ", "shares"] == 500
    assert df.loc["000002.SZ", "ref_close"] == pytest.approx(20.0)
    assert df.loc["600000.SH", "shares"] == 100
    assert df.loc["600000.SH", "entry_date"] == "2026-10-27"
    assert df.loc["600000.SH", "ref_close"] == pytest.approx(4.8)


def test_cli_fills_uses_real_execution_impl():
    import predict_tomorrow as pt

    assert pt.save_holdings_from_fills is execution.save_holdings_from_fills
    assert pt.load_fills_csv is execution.load_fills_csv
    assert pt.build_parser().parse_args(["--fills", "fills.csv"]).fills == "fills.csv"


def test_main_commit_uses_fills_instead_of_order_assumption(monkeypatch, tmp_path):
    import config as C
    import predict_tomorrow as pt

    dates = pd.bdate_range("2026-10-20", periods=6)
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["000001.SZ"] * len(dates) + ["600000.SH"] * len(dates),
            "close": 10.0,
            "adj_close": 10.0,
            "pct_chg": 0.0,
            "return_1d": 0.0,
            C.TARGET: 0.01,
            C.WEIGHT: 1.0,
            "feature_00": 0.5,
        }
    )
    latest = panel["date"].max()
    today = panel[panel["date"] == latest].copy()
    today["score"] = 1.0
    today["prediction"] = 1.0
    out = tmp_path / "outputs"
    monkeypatch.setattr(C, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(C, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(C, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(pt, "load_live_panel", lambda **kwargs: (panel, ["feature_00"]))
    monkeypatch.setattr(
        pt,
        "run_gated_prediction",
        lambda *args, **kwargs: {
            "today": today,
            "info": {"weights": {"gbm_l2": 1.0}},
            "gate": None,
            "action": "frozen",
            "ic_stats": None,
        },
    )
    monkeypatch.setattr(
        pt,
        "make_orders_from_state",
        lambda *args, **kwargs: _orders().assign(limit_price=lambda frame: frame["ref_close"]),
    )
    holdings = out / "holdings.csv"
    out.mkdir()
    _holdings_file(holdings)
    fills = out / "fills.csv"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "600000.SH"],
            "side": ["sell", "buy"],
            "shares": [400, 100],
            "price": [11.0, 5.0],
        }
    ).to_csv(fills, index=False)

    pt.main(
        [
            "--commit",
            "--fills",
            str(fills),
            "--skip-preflight",
            "--holdings",
            str(holdings),
            "--account",
            str(out / "account.csv"),
            "--risk-state",
            str(out / "risk.json"),
        ]
    )

    df = execution.load_holdings_table(str(holdings)).set_index("symbol")
    assert df.loc["000001.SZ", "shares"] == 600
    assert df.loc["600000.SH", "shares"] == 100
    assert df.loc["600000.SH", "ref_close"] == pytest.approx(5.0)


def test_load_rejected_rids(tmp_path):
    assert execution.load_rejected_rids(str(tmp_path), "20261027") == set()
    (tmp_path / "execution_state_20261027.json").write_text(
        '{"buy-600000.SH": "rejected", "sell-000001.SZ": true}', encoding="utf-8"
    )
    assert execution.load_rejected_rids(str(tmp_path), "20261027") == {"buy-600000.SH"}


# ---------------------------------------------------------------------------
# 持续持仓账本：完整历史快照与总览手动校正
# ---------------------------------------------------------------------------
def test_order_and_fill_commits_append_complete_history(tmp_path):
    path = str(tmp_path / "holdings.csv")
    _holdings_file(path)
    execution.save_holdings(path, _orders(), {"000001.SZ": 1000, "000002.SZ": 500}, "2026-10-27")
    after_orders = execution.load_holdings_table(path)
    fills = pd.DataFrame(
        {"symbol": ["000002.SZ"], "side": ["sell"], "shares": [500], "price": [20.0]}
    )
    current = dict(zip(after_orders["symbol"], after_orders["shares"]))
    execution.save_holdings_from_fills(path, fills, current, "2026-10-28")

    history = pd.read_csv(tmp_path / "holdings_history.csv")
    assert history["snapshot_id"].nunique() == 2
    first = history[history["action"] == "orders_commit"]
    second = history[history["action"] == "fills_commit"]
    assert len(first) == 3 and len(second) == 2
    assert set(first["symbol"]) == {"000001.SZ", "000002.SZ", "600000.SH"}
    assert set(second["symbol"]) == {"000001.SZ", "600000.SH"}
    assert set(second["effective_date"]) == {"2026-10-28"}


def test_manual_replacement_and_empty_snapshot_are_recorded(tmp_path):
    path = str(tmp_path / "holdings.csv")
    text = """symbol,shares,entry_date,ref_close
600000.SH,100,2026-10-01,4.8
000001.SZ,0,2026-09-01,10.0
"""
    holdings = execution.parse_holdings_editor(text)
    assert list(holdings["symbol"]) == ["600000.SH"]
    execution.replace_holdings(path, holdings, "2026-10-28")
    assert execution.load_holdings_table(path).iloc[0]["shares"] == 100

    empty = execution.parse_holdings_editor("symbol,shares,entry_date,ref_close\n")
    assert empty.empty
    execution.replace_holdings(path, empty, "2026-10-29")
    assert execution.load_holdings_table(path).empty
    history = pd.read_csv(tmp_path / "holdings_history.csv")
    assert history["snapshot_id"].nunique() == 2
    assert history.iloc[-1]["action"] == "manual_edit"
    assert history.iloc[-1]["shares"] == 0 and pd.isna(history.iloc[-1]["symbol"])


@pytest.mark.parametrize(
    "text, match",
    [
        ("", "内容为空"),
        ("symbol,shares\n000001.SZ,100\n", "必需列"),
        ("symbol,shares,entry_date,ref_close\nBAD,100,2026-01-01,10\n", "代码格式"),
        ("symbol,shares,entry_date,ref_close\n000001.SZ,1.5,2026-01-01,10\n", "股数"),
        (
            "symbol,shares,entry_date,ref_close\n000001.SZ,100,2026-01-01,10\n000001.SZ,200,2026-01-01,10\n",
            "代码重复",
        ),
        ("symbol,shares,entry_date,ref_close\n000001.SZ,100,not-a-date,10\n", "建仓日"),
        ("symbol,shares,entry_date,ref_close\n000001.SZ,100,2026-01-01,-1\n", "参考价"),
    ],
)
def test_holdings_editor_rejects_invalid_input(text, match):
    with pytest.raises(ValueError, match=match):
        execution.parse_holdings_editor(text)
