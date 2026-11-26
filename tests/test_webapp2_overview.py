"""总览页：待办规则（纯函数）与渲染。每条规则对应一种真实出现过的线上状况。"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

from webapp2 import actions, data
from webapp2.pages import overview

from tests.test_webapp2_pages import ids_of, text_of  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(tmp_path / "models"))
    # 行情缓存也必须隔离，否则测试结果取决于本机真实的 data_cache
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    data.refresh()
    yield tmp_path
    data.refresh()


def _put_champion(sandbox, age=45):
    (sandbox / "models" / "champion").mkdir(parents=True, exist_ok=True)
    (sandbox / "models" / "champion" / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "m1",
                "train_end_date": "2026-09-03",
                "config_snapshot": {"feat_cols": ["a"]},
                "promotion": {"decided_at": "2026-11-16", "reason": "手动"},
            }
        )
    )
    (sandbox / "outputs" / "tracking").mkdir(exist_ok=True)
    (sandbox / "outputs" / "tracking" / "tracking_report.json").write_text(
        json.dumps({"champion": {"age_days": age}, "live": {"n_ic": 0}, "alerts": []})
    )
    data.refresh()


def _fresh_account(sandbox, cash=30000):
    (sandbox / "outputs" / "account.csv").write_text(f"cash\n{cash}\n")
    data.refresh()


def _todo(sandbox, runs=()):
    return overview.todo_items(data.overview(), list(runs))


def test_everything_missing_flags_account_and_champion_as_urgent(sandbox):
    items = _todo(sandbox)
    titles = [t for _, t, _ in items]
    assert "账户现金未维护" in titles and "没有 champion" in titles
    assert all(level == "bad" for level, _, _ in items)


def test_stale_cash_blocks_nightly_and_says_why(sandbox):
    """真实场景：account.csv 11-06 之后没更新，夜间体检以退出码 3 拦下。"""
    _put_champion(sandbox)
    _fresh_account(sandbox)
    p = sandbox / "outputs" / "account.csv"
    old = time.time() - 10 * 86400
    os.utime(p, (old, old))
    data.refresh()
    items = _todo(sandbox)
    assert items[0][0] == "bad" and "已过期" in items[0][1]
    assert "退出码 3" in items[0][2] and "10 天" in items[0][2]


def test_champion_age_thresholds(sandbox):
    _fresh_account(sandbox)
    for age, expect_level, expect_text in [
        (45, None, None),
        (50, "warn", "临近超龄"),
        (66, "bad", "超龄 66"),
    ]:
        _put_champion(sandbox, age)
        hits = [(lv, t) for lv, t, _ in _todo(sandbox) if "超龄" in t]
        if expect_level is None:
            assert hits == []
        else:
            assert hits and hits[0][0] == expect_level and expect_text in hits[0][1]


def test_tracking_alerts_and_failed_or_dead_runs_surface(sandbox):
    _fresh_account(sandbox)
    _put_champion(sandbox)
    (sandbox / "outputs" / "tracking" / "tracking_report.json").write_text(
        json.dumps({"champion": {"age_days": 10}, "alerts": ["IC 退化：近 20 日均值 -0.01"]})
    )
    data.refresh()
    runs = [
        {"status": "failed", "label": "夜间出单", "failure": {"exc_msg": "KeyError pct_chg"}},
        {"status": "running", "liveness": "dead", "label": "WF"},
    ]
    t = text_of([f"{lv}|{ti}|{d}" for lv, ti, d in _todo(sandbox, runs)])
    assert "IC 退化" in t and "夜间出单" in t and "KeyError pct_chg" in t and "运行意外中断" in t


def test_sorted_by_urgency(sandbox):
    _fresh_account(sandbox)
    _put_champion(sandbox, 50)
    items = _todo(sandbox, [{"status": "running", "liveness": "dead", "label": "x"}])
    levels = [lv for lv, _, _ in items]
    assert levels == sorted(levels, key=lambda lv: {"bad": 0, "warn": 1, "info": 2}[lv])


def test_pending_orders_hint_when_not_yet_confirmed(sandbox):
    _fresh_account(sandbox)
    _put_champion(sandbox)
    pd.DataFrame(
        [["A.SZ", "buy", 100, 2.0, 2.0, False]],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked"],
    ).to_csv(sandbox / "outputs" / "orders_20261106.csv", index=False)
    data.refresh()
    assert any("笔订单待执行" in t for _, t, _ in _todo(sandbox))


def _orders_file(sandbox, day):
    pd.DataFrame(
        [["A.SZ", "buy", 100, 2.0, 2.0, False]],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked"],
    ).to_csv(sandbox / "outputs" / f"orders_{day}.csv", index=False)
    data.refresh()


def _put_daily_cache(sandbox, *days):
    d = sandbox / "data_cache" / "tushare"
    d.mkdir(parents=True, exist_ok=True)
    for day in days:
        (d / f"daily_{day}.parquet").write_bytes(b"")
    data.refresh()


def test_orders_older_than_market_data_are_flagged_expired_not_pending(sandbox, monkeypatch):
    """真实场景：订单是 11-06 的，行情缓存已到 11-13。这批订单已作废，不能再提示「待执行」。"""
    monkeypatch.setattr(data, "CACHE", str(sandbox / "data_cache"))
    _fresh_account(sandbox)
    _put_champion(sandbox)
    _orders_file(sandbox, "20261106")
    _put_daily_cache(sandbox, "20261105", "20261106", "20261113")
    items = _todo(sandbox)
    titles = [t for _, t, _ in items]
    assert any("最新订单已过期" in t and "20261113" in t for t in titles)
    assert not any("笔订单待执行" in t for t in titles)
    assert "（已过期，勿执行）" in text_of(overview._orders_card(data.overview()))


def test_orders_on_latest_data_day_are_pending_until_confirmed(sandbox, monkeypatch):
    monkeypatch.setattr(data, "CACHE", str(sandbox / "data_cache"))
    _fresh_account(sandbox)
    _put_champion(sandbox)
    _orders_file(sandbox, "20261106")
    _put_daily_cache(sandbox, "20261105", "20261106")
    assert any("笔订单待执行" in t for _, t, _ in _todo(sandbox))
    # 持仓里已有当日建仓记录 → 视为已确认，不再提示
    pd.DataFrame(
        [["A.SZ", 100, "2026-11-06", 2.0]], columns=["symbol", "shares", "entry_date", "ref_close"]
    ).to_csv(sandbox / "outputs" / "holdings.csv", index=False)
    data.refresh()
    assert not any("笔订单待执行" in t for _, t, _ in _todo(sandbox))


def test_latest_data_day_reads_filenames_and_tolerates_missing_dir(sandbox, monkeypatch):
    monkeypatch.setattr(data, "CACHE", str(sandbox / "nope"))
    data.refresh()
    assert data.latest_data_day() is None
    monkeypatch.setattr(data, "CACHE", str(sandbox / "data_cache"))
    _put_daily_cache(sandbox, "20261111", "20261113", "20261112")
    (sandbox / "data_cache" / "tushare" / "daily_notadate.parquet").write_bytes(b"")
    data.refresh()
    assert data.latest_data_day() == "20261113"


def test_all_clear_has_no_todo(sandbox):
    _fresh_account(sandbox)
    _put_champion(sandbox, 20)
    assert _todo(sandbox) == []
    assert "没有需要处理的事项" in text_of(overview._todo_card([]))


# ----------------------------------------------------------------------------- 渲染
def test_body_renders_with_empty_disk(sandbox):
    t = text_of(overview.layout())
    for needle in ("待办", "没有订单", "没有 champion", "当前空仓", "没有运行记录"):
        assert needle in t, needle


def test_body_renders_real_shaped_data(sandbox):
    _fresh_account(sandbox)
    _put_champion(sandbox)
    pd.DataFrame(
        [
            ["000785.SZ", "buy", 1100, 2.12, 2.12, "再平衡:建仓", 0.0002, False],
            ["600001.SH", "buy", 100, 8.0, 8.0, "再平衡:建仓", 0.5, True],
        ],
        columns=[
            "symbol",
            "side",
            "shares",
            "limit_price",
            "ref_close",
            "reason",
            "rank_pct",
            "blocked",
        ],
    ).to_csv(sandbox / "outputs" / "orders_20261106.csv", index=False)
    (sandbox / "outputs" / "live_risk_state.json").write_text(
        json.dumps({"total_values": [30000.0]})
    )
    data.refresh()
    t = text_of(overview.layout())
    for needle in ("000785.SZ", "买 1", "拦截 1", "30,000", "m1", "样本不足"):
        assert needle in t, needle


def test_cash_form_is_outside_polled_body_and_save_works(sandbox):
    """轮询只替换 ov-body 与 cash-status-slot；输入框与保存按钮必须在它们之外，
    否则用户输入到一半会被 5 秒一次的刷新清掉。"""
    page = overview.layout()
    ids_all = ids_of(page)
    assert {
        "ov-tick",
        "ov-body",
        "cash-input",
        "cash-save",
        "cash-status-slot",
        "cash-msg",
    } <= ids_all
    assert not ({"cash-input", "cash-save"} & ids_of(overview.body()))
    ok, _ = actions.save_cash(31000, "2026-11-17")
    assert ok and "31,000" in text_of(overview._cash_status(data.overview()))


def test_orders_card_strikes_blocked_rows(sandbox):
    pd.DataFrame(
        [
            ["A.SZ", "buy", 100, 2.0, 2.0, "r", 0.1, False],
            ["B.SZ", "buy", 100, 2.0, 2.0, "r", 0.2, True],
        ],
        columns=[
            "symbol",
            "side",
            "shares",
            "limit_price",
            "ref_close",
            "reason",
            "rank_pct",
            "blocked",
        ],
    ).to_csv(sandbox / "outputs" / "orders_20261106.csv", index=False)
    data.refresh()
    card = overview._orders_card(data.overview())
    classes = []

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            [walk(c) for c in n]
            return
        if getattr(n, "className", None) == "blocked":
            classes.append(text_of(n))
        walk(getattr(n, "children", None))

    walk(card)
    assert len(classes) == 1 and "B.SZ" in classes[0]


# ----------------------------------------------------------------------------- 持仓账本、订单落实与个性化设置
def test_holding_order_states_cover_all_groups():
    holdings = pd.DataFrame(
        {"symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000006.SZ"]}
    )
    orders = pd.DataFrame(
        [
            ["000001.SZ", "buy", 100, False],
            ["000002.SZ", "buy", 100, False],
            ["000002.SZ", "sell", 100, False],
            ["000003.SZ", "buy", 100, False],
            ["000004.SZ", "buy", 100, False],
            ["000006.SZ", "buy", 100, True],
        ],
        columns=["symbol", "side", "shares", "blocked"],
    )
    state = {
        "buy-000001.SZ": "done",
        "buy-000002.SZ": "done",
        "buy-000003.SZ": "rejected",
    }
    assert overview.holding_order_states(holdings, orders, state) == {
        "000001.SZ": "已执行",
        "000002.SZ": "部分落实",
        "000003.SZ": "已拒绝",
        "000004.SZ": "待执行",
        "000005.SZ": "无本批订单",
        "000006.SZ": "已拦截",
    }


def test_holdings_editor_replaces_exact_state_and_appends_history(sandbox):
    ok, msg = actions.save_holdings_editor(
        "symbol,shares,entry_date,ref_close\n600000.SH,100,2026-01-02,4.8\n"
    )
    assert ok, msg
    assert data.holdings().iloc[0]["shares"] == 100
    ok, _ = actions.save_holdings_editor(
        "symbol,shares,entry_date,ref_close\n600000.SH,200,2026-01-03,5.0\n"
    )
    assert ok
    assert data.holdings().iloc[0]["shares"] == 200
    history = data.holdings_history()
    assert len(history) == 2 and set(history["action"]) == {"manual_edit"}
    assert history["snapshot_id"].nunique() == 2
    assert "手动校正" in text_of(overview._history_card())

    before = (sandbox / "outputs" / "holdings.csv").read_text()
    ok, _ = actions.save_holdings_editor("symbol,shares,entry_date,ref_close\nBAD,100,x,1\n")
    assert not ok and (sandbox / "outputs" / "holdings.csv").read_text() == before
    assert len(data.holdings_history()) == 2


def test_overview_forms_are_static_and_refresh_is_signature_gated(sandbox):
    new_body, sig = overview.refresh_decision(None)
    assert new_body is not None
    assert overview.refresh_decision(sig)[0] is None
    ids = ids_of(overview.layout())
    assert {
        "ov-sig",
        "ov-holdings-input",
        "ov-holdings-save",
        "ov-holdings-reload",
        "ov-settings-save",
        "ov-setting-stamp_duty_rate",
        "ov-setting-use-impact",
    } <= ids
    assert not (
        {"ov-holdings-input", "ov-settings-save", "ov-setting-stamp_duty_rate"}
        & ids_of(overview.body())
    )
    assert actions.save_holdings_editor("symbol,shares,entry_date,ref_close\n")[0]
    assert overview.refresh_decision(sig)[0] is not None


def test_execution_status_editor_is_rendered_with_current_orders(sandbox, monkeypatch):
    monkeypatch.setattr(actions, "today_str", lambda: "20261123")
    _orders_file(sandbox, "20261123")
    actions.set_execution_mark("20261123", "buy", "A.SZ", "done")
    data.refresh()
    text = text_of(overview._execution_card(data.overview()))
    assert "再次编辑" in text and "已执行" in text and "已拒绝" not in text


def test_trading_preferences_persist_and_update_current_process(sandbox, monkeypatch):
    import config as C

    names = (
        "COMMISSION_RATE",
        "STAMP_DUTY_RATE",
        "MIN_COMMISSION",
        "IMPACT_K",
        "USE_IMPACT_COST",
        "MAX_PARTICIPATION",
    )
    for name in names:
        monkeypatch.setattr(C, name, getattr(C, name))
        monkeypatch.delenv(name, raising=False)
    values = {
        "commission_rate": 0.0003,
        "stamp_duty_rate": 0.0005,
        "min_commission": 2.0,
        "impact_k": 0.2,
        "use_impact_cost": False,
        "max_participation": 0.05,
    }
    ok, msg, saved = actions.save_trading_preferences(values)
    assert ok, msg
    assert saved == values
    assert actions.trading_preferences() == values
    assert C.STAMP_DUTY_RATE == pytest.approx(0.0005)
    assert C.COMMISSION_RATE == pytest.approx(0.0003)
    assert not C.USE_IMPACT_COST
    assert "新启动" in msg and "进行中的任务不变" in msg

    ok, _, _ = actions.save_trading_preferences({"stamp_duty_rate": 0.2})
    assert not ok
    assert actions.trading_preferences()["stamp_duty_rate"] == pytest.approx(0.0005)


def test_latest_execution_state_keeps_order_status_visible_on_later_days(sandbox, monkeypatch):
    _orders_file(sandbox, "20261123")
    actions.set_execution_mark("20261123", "buy", "A.SZ", "done")
    pd.DataFrame(
        [["A.SZ", 100, "2026-11-23", 2.0]],
        columns=["symbol", "shares", "entry_date", "ref_close"],
    ).to_csv(sandbox / "outputs" / "holdings.csv", index=False)
    monkeypatch.setattr(actions, "today_str", lambda: "20261124")
    data.refresh()

    day, state = data.latest_execution_state()
    assert day == "20261123" and state["buy-A.SZ"] == "done"
    assert "已执行" in text_of(overview._execution_card(data.overview()))
    card_text = text_of(overview._holdings_card(data.overview()))
    assert "已执行" in card_text and "待执行" not in card_text
