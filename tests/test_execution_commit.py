"""成交确认写回与「今日流程」前端能力的单元测试：

· src/execution.py::save_holdings —— CLI（predict_tomorrow --commit）与
  webapp「确认成交」共用的写回实现：entry_date 保留/刷新、卖出减仓到 0 移除、
  blocked 单跳过、文件缺失时新建；
· predict_tomorrow 与 execution 为同一实现（CLI/web 行为逐位一致）；
· dataio 执行勾选状态 JSON 读写、append_account_cash 创建/追加/表头校验；
· 交易台流程卡状态判定纯函数（_step1~4_state / _exec_progress）。
"""

from __future__ import annotations


import pandas as pd
import pytest
from dash.development.base_component import Component
from dash import dcc
import dash_bootstrap_components as dbc

import execution
from webapp import dataio
from webapp.pages import trading_desk as td


def _stub_is_trading_day(day):
    """测试桩：交易日判定不打真实日历/网络（带 cache_clear 兼容 refresh_caches）。"""
    return None


_stub_is_trading_day.cache_clear = lambda: None


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
    """CLI（predict_tomorrow --commit）与 webapp 调同一实现，行为逐位一致。"""
    import predict_tomorrow as pt

    assert pt.save_holdings is execution.save_holdings
    assert pt.load_holdings_table is execution.load_holdings_table


# ---------------------------------------------------------------------------
# dataio：执行勾选状态 / 现金维护
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(dataio, "OUTPUTS", str(out))
    dataio.refresh_caches()
    yield out
    dataio.refresh_caches()


def test_execution_state_roundtrip(tmp_outputs):
    day = "20261027"
    assert dataio.load_execution_state(day) == {}  # 缺失 → {}
    dataio.save_execution_state(day, {"buy-600000.SH": True, "sell-000001.SZ": True})
    assert dataio.load_execution_state(day) == {
        "buy-600000.SH": True,
        "sell-000001.SZ": True,
    }
    # 取消勾选后全量重写
    dataio.save_execution_state(day, {"buy-600000.SH": True})
    assert dataio.load_execution_state(day) == {"buy-600000.SH": True}
    # 损坏 JSON → {}
    (tmp_outputs / f"execution_state_{day}.json").write_text("{bad", encoding="utf-8")
    assert dataio.load_execution_state(day) == {}


def test_append_account_cash_create_and_append(tmp_outputs):
    path = tmp_outputs / "account.csv"
    dataio.append_account_cash("2026-10-27", 30000.0)
    assert path.read_text(encoding="utf-8") == "date,cash\n2026-10-27,30000.00\n"
    dataio.append_account_cash("2026-10-28", 31500.5)
    df = pd.read_csv(path)
    assert list(df["date"]) == ["2026-10-27", "2026-10-28"]
    assert df["cash"].iloc[-1] == pytest.approx(31500.5)
    # account() 的 mtime 缓存自动失效，立即读到新值
    assert dataio.account()["cash"].iloc[-1] == pytest.approx(31500.5)


def test_append_account_cash_bad_header(tmp_outputs):
    (tmp_outputs / "account.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="date/cash"):
        dataio.append_account_cash("2026-10-27", 100.0)


# ---------------------------------------------------------------------------
# 交易台流程卡：状态判定纯函数
# ---------------------------------------------------------------------------
def test_step1_state():
    # 非交易日 → 灰
    assert td._step1_state(None, None, "", "20261025", False, False) == ("disabled", "今日非交易日")
    # 今日订单（当前周期内）→ 绿（带 HH:MM）
    lamp, txt = td._step1_state("20261027", 1793080800.0, "", "20261027", True, True)
    assert lamp == "ok" and "已完成" in txt
    # 昨晚生成、今早执行：订单日期 = 数据日（≠ 今日）但在执行周期内 → 绿
    lamp, txt = td._step1_state("20261026", 1793080800.0, "", "20261027", True, True)
    assert lamp == "ok" and "20261026" in txt and "早盘待执行" in txt
    # 会话来源在文案中标注
    lamp, txt = td._step1_state(
        "20261027", 1793080800.0, "会话 nextday_20261027_135952", "20261027", True, True
    )
    assert lamp == "ok" and "nextday_20261027_135952" in txt
    # 有订单但不在当前执行周期 → 黄（提示重新生成）
    assert td._step1_state("20261020", 1793080800.0, "", "20261027", True, False) == (
        "warn",
        "待生成",
    )
    # 无产物 → 黄；日历未知 → 黄但标注
    assert td._step1_state(None, None, "", "20261027", True, False) == ("warn", "待生成")
    assert td._step1_state(None, None, "", "20261027", None, False)[1] == "待生成（日历未知）"
    # _fmt_hhmm 容错
    assert td._fmt_hhmm(1793080800.0) and td._fmt_hhmm("bad") == ""


def test_step2_state():
    assert td._step2_state(0, 0, False) == ("disabled", "等待订单")
    assert td._step2_state(0, 3, True) == ("warn", "待执行 0/3")
    assert td._step2_state(1, 3, True) == ("warn", "已勾选 1/3")
    assert td._step2_state(3, 3, True) == ("ok", "已全部勾选 3/3")


def test_step3_state():
    # 无今日订单 → 灰（确认按钮禁用）
    assert td._step3_state(False, None, None)[0] == "disabled"
    # holdings 晚于订单 → 绿；早于 → 黄；无 holdings → 黄（空仓首建）
    assert td._step3_state(True, 200.0, 100.0)[0] == "ok"
    assert td._step3_state(True, 100.0, 200.0)[0] == "warn"
    assert td._step3_state(True, None, 100.0)[0] == "warn"


def test_step4_state():
    assert td._step4_state(pd.DataFrame(), "2026-10-27")[0] == "disabled"
    acct = pd.DataFrame({"date": ["2026-10-27"], "cash": [30000.0]})
    assert td._step4_state(acct, "2026-10-27")[0] == "ok"
    old = pd.DataFrame({"date": ["2026-10-20"], "cash": [30000.0]})
    lamp, txt = td._step4_state(old, "2026-10-27")
    assert lamp == "warn" and "2026-10-20" in txt


def test_exec_progress_counts_non_blocked():
    orders = pd.DataFrame(
        [
            ["000001.SZ", "sell", False],
            ["600000.SH", "buy", False],
            ["600001.SH", "buy", True],
        ],
        columns=["symbol", "side", "blocked"],
    )
    assert td._exec_progress(orders, {}) == (0, 2)
    assert td._exec_progress(orders, {"sell-000001.SZ": True, "buy-600001.SH": True}) == (1, 2)
    assert td._exec_progress(pd.DataFrame(), {"x": True}) == (0, 0)


def test_exec_progress_excludes_rejected():
    """拒绝单不计入应执行笔数，也不因残留 true 值被计入已勾选。"""
    orders = pd.DataFrame(
        [
            ["000001.SZ", "sell", False],
            ["600000.SH", "buy", False],
            ["600001.SH", "buy", True],
        ],
        columns=["symbol", "side", "blocked"],
    )
    assert td._exec_progress(orders, {"buy-600000.SH": "rejected"}) == (0, 1)
    assert td._exec_progress(orders, {"buy-600000.SH": "rejected", "sell-000001.SZ": True}) == (
        1,
        1,
    )
    assert td._step2_state(0, 1, True, 1) == ("warn", "待执行 0/1（已拒绝 1 笔）")
    assert td._step2_state(0, 0, True, 2) == ("ok", "应执行单已全部拒绝（2 笔）")


def test_load_rejected_rids(tmp_path):
    assert execution.load_rejected_rids(str(tmp_path), "20261027") == set()
    (tmp_path / "execution_state_20261027.json").write_text(
        '{"buy-600000.SH": "rejected", "sell-000001.SZ": true}', encoding="utf-8"
    )
    assert execution.load_rejected_rids(str(tmp_path), "20261027") == {"buy-600000.SH"}


# ---------------------------------------------------------------------------
# layout：流程卡渲染 + 勾选回读
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_artifacts(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "tracking").mkdir(parents=True)
    monkeypatch.setattr(dataio, "OUTPUTS", str(out))
    monkeypatch.setattr(dataio, "TRACKING", str(out / "tracking"))
    monkeypatch.setattr(dataio, "TUSHARE_CACHE", str(tmp_path / "tushare"))
    monkeypatch.setattr(dataio, "is_trading_day", _stub_is_trading_day)
    monkeypatch.setattr(td, "_iv_scale_now", lambda: ("disabled", None))
    dataio.refresh_caches()
    td._close_maps.cache_clear()
    yield out
    dataio.refresh_caches()
    td._close_maps.cache_clear()


def _ids(comp) -> list:
    out = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, Component):
            cid = getattr(x, "id", None)
            if cid is not None:
                out.append(cid)
            walk(getattr(x, "children", None))

    walk(comp)
    return out


def _components(comp, cls) -> list:
    out = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, Component):
            if isinstance(x, cls):
                out.append(x)
            walk(getattr(x, "children", None))

    walk(comp)
    return out


def test_layout_flow_card_ids(tmp_artifacts):
    ids = {i for i in _ids(td.layout()) if isinstance(i, str)}
    for cid in (
        "td-flow-gen",
        "td-flow-s1-status",
        "td-flow-s2-status",
        "td-flow-s3-status",
        "td-flow-s4-status",
        "td-commit-open",
        "td-commit-modal",
        "td-commit-body",
        "td-commit-run",
        "td-commit-cancel",
        "td-cash-input",
        "td-cash-save",
        "td-cash-msg",
        "td-exec-store",
        "td-reject-store",
        "td-card-c-wrap",
    ):
        assert cid in ids, f"layout 缺少组件 {cid}"


def test_layout_flow_empty_state(tmp_artifacts, monkeypatch):
    """空态：无订单/无持仓/未建账 → ①黄 ②灰 ③灰（按钮禁用）④灰。"""
    monkeypatch.setattr(td, "_today_str", lambda: "20261027")
    true_stub = lambda day: True  # noqa: E731
    true_stub.cache_clear = lambda: None
    monkeypatch.setattr(dataio, "is_trading_day", true_stub)
    flow = td._compute_flow(pd.DataFrame())
    assert flow["s1"] == ("warn", "待生成")
    assert flow["s2"][0] == "disabled"
    assert flow["s3"][0] == "disabled" and not flow["can_commit"]
    assert flow["s4"][0] == "disabled"


def test_layout_exec_state_restored(tmp_artifacts, monkeypatch):
    """今日订单 + 已持久化勾选 → 执行单对应勾选框回读为选中，②进度 1/2。"""
    monkeypatch.setattr(td, "_today_str", lambda: "20261027")
    pd.DataFrame(
        [
            ["000001.SZ", "sell", 400, 11.0, 10.8, False],
            ["600000.SH", "buy", 200, 5.0, 4.9, False],
        ],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked"],
    ).to_csv(tmp_artifacts / "orders_20261027.csv", index=False)
    dataio.save_execution_state("20261027", {"sell-000001.SZ": True})
    dataio.refresh_caches()

    comp = td.layout()
    done_btns = {
        b.id["index"]: b
        for b in _components(comp, dbc.Button)
        if isinstance(b.id, dict) and b.id.get("type") == "td-done"
    }
    assert done_btns["sell-000001.SZ"].children == "✓ 已执行"
    assert done_btns["sell-000001.SZ"].outline is False
    assert done_btns["buy-600000.SH"].children == "执行"
    assert done_btns["buy-600000.SH"].outline is True
    stores = {s.id: s for s in _components(comp, dcc.Store) if isinstance(s.id, str)}
    assert stores["td-exec-store"].data == ["sell-000001.SZ"]
    flow = td._compute_flow(pd.DataFrame())
    assert flow["s2"] == ("warn", "已勾选 1/2")
    assert flow["can_commit"] is True
    # 现金输入框预填最新 cash
    pd.DataFrame({"date": ["2026-10-26"], "cash": [30000.0]}).to_csv(
        tmp_artifacts / "account.csv", index=False
    )
    dataio.refresh_caches()
    comp = td.layout()
    cash_in = next(c for c in _components(comp, dbc.Input) if c.id == "td-cash-input")
    assert cash_in.value == pytest.approx(30000.0)


def test_layout_rejected_state_restored(tmp_artifacts, monkeypatch):
    """拒绝状态回读：拒绝按钮显示「✗ 已拒绝」红实底、执行按钮保持未选、store 含该单。"""
    monkeypatch.setattr(td, "_today_str", lambda: "20261027")
    pd.DataFrame(
        [
            ["000001.SZ", "sell", 400, 11.0, 10.8, False],
            ["600000.SH", "buy", 200, 5.0, 4.9, False],
        ],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked"],
    ).to_csv(tmp_artifacts / "orders_20261027.csv", index=False)
    dataio.save_execution_state("20261027", {"sell-000001.SZ": "rejected"})
    dataio.refresh_caches()

    comp = td.layout()
    rej_btns = {
        b.id["index"]: b
        for b in _components(comp, dbc.Button)
        if isinstance(b.id, dict) and b.id.get("type") == "td-reject"
    }
    assert rej_btns["sell-000001.SZ"].children == "✗ 已拒绝"
    assert rej_btns["sell-000001.SZ"].color == "danger"
    assert rej_btns["sell-000001.SZ"].outline is False
    assert rej_btns["buy-600000.SH"].children == "拒绝"
    assert rej_btns["buy-600000.SH"].outline is True
    done_btns = {
        b.id["index"]: b
        for b in _components(comp, dbc.Button)
        if isinstance(b.id, dict) and b.id.get("type") == "td-done"
    }
    assert done_btns["sell-000001.SZ"].children == "执行"
    stores = {s.id: s for s in _components(comp, dcc.Store) if isinstance(s.id, str)}
    assert stores["td-reject-store"].data == ["sell-000001.SZ"]
    assert stores["td-exec-store"].data == []
    # 拒绝单不计入②应执行笔数
    flow = td._compute_flow(pd.DataFrame())
    assert flow["s2"] == ("warn", "待执行 0/1（已拒绝 1 笔）")


def test_do_commit_writes_holdings(tmp_artifacts, monkeypatch):
    """网页成交确认路径：今日订单 + 当前持仓 → holdings.csv 写回（共用 save_holdings）。"""
    monkeypatch.setattr(td, "_today_str", lambda: "20261027")
    pd.DataFrame(
        [
            ["000001.SZ", "sell", 400, 11.0, False],
            ["600000.SH", "buy", 200, 5.0, False],
            ["600001.SH", "buy", 100, 8.0, True],
        ],
        columns=["symbol", "side", "shares", "ref_close", "blocked"],
    ).to_csv(tmp_artifacts / "orders_20261027.csv", index=False)
    _holdings_file(tmp_artifacts / "holdings.csv")
    dataio.refresh_caches()

    n = td._do_commit()
    assert n == 3  # 000001 减仓 + 000002 保留 + 600000 新仓
    df = execution.load_holdings_table(str(tmp_artifacts / "holdings.csv")).set_index("symbol")
    assert df.loc["000001.SZ", "shares"] == 600
    assert df.loc["000001.SZ", "entry_date"] == "2026-10-01"  # 保留
    assert df.loc["600000.SH", "entry_date"] == "2026-10-27"  # 买入刷新
    assert "600001.SH" not in df.index  # blocked

    # 订单不在当前执行周期（昨日订单 + 今日无日线缓存）→ 拒绝
    monkeypatch.setattr(td, "_today_str", lambda: "20261028")
    with pytest.raises(RuntimeError, match="无当前周期的订单"):
        td._do_commit()


def test_do_commit_skips_rejected(tmp_artifacts, monkeypatch):
    """拒绝单视同拦截单：成交确认跳过；全部拒绝 → 拒绝写回。"""
    monkeypatch.setattr(td, "_today_str", lambda: "20261027")
    pd.DataFrame(
        [
            ["000001.SZ", "sell", 400, 11.0, False],
            ["600000.SH", "buy", 200, 5.0, False],
            ["600001.SH", "buy", 100, 8.0, True],
        ],
        columns=["symbol", "side", "shares", "ref_close", "blocked"],
    ).to_csv(tmp_artifacts / "orders_20261027.csv", index=False)
    _holdings_file(tmp_artifacts / "holdings.csv")
    dataio.save_execution_state("20261027", {"sell-000001.SZ": "rejected", "buy-600000.SH": True})
    dataio.refresh_caches()

    n = td._do_commit()
    assert n == 3  # 000001 拒绝 → 保持 1000；000002 保留；600000 买入；600001 拦截
    df = execution.load_holdings_table(str(tmp_artifacts / "holdings.csv")).set_index("symbol")
    assert df.loc["000001.SZ", "shares"] == 1000  # 拒绝的卖单未生效
    assert df.loc["600000.SH", "shares"] == 200

    # 全部非拦截单拒绝 → 无成交可确认
    dataio.save_execution_state(
        "20261027", {"sell-000001.SZ": "rejected", "buy-600000.SH": "rejected"}
    )
    with pytest.raises(RuntimeError, match="全部拒绝"):
        td._do_commit()


def test_flow_from_session_artifact(tmp_artifacts, monkeypatch):
    """仅 nextday 会话产物（无 CLI orders_*.csv）→ ①绿（标注会话来源）且③可用。

    口径：会话订单 asof_date=数据日（20261026），今日 20261028 早盘，
    日线缓存最新 = 20261026 → 订单在执行周期内（_orders_fresh）。
    """
    import json

    sid = "nextday_20261027_135952"
    sdir = tmp_artifacts / "sessions" / sid
    (sdir / "artifacts").mkdir(parents=True)
    pd.DataFrame(
        [["600000.SH", "buy", 200, 5.0, False, "2026-10-26"]],
        columns=["symbol", "side", "shares", "ref_close", "blocked", "asof_date"],
    ).to_csv(sdir / "artifacts" / "orders.csv", index=False)
    (sdir / "session.json").write_text(
        json.dumps({"id": sid, "mode": "nextday", "status": "done", "finished": 1793000000.0}),
        encoding="utf-8",
    )
    tdir = tmp_artifacts.parent / "tushare"
    tdir.mkdir()
    pd.DataFrame({"symbol": ["600000.SH"], "close": [5.0]}).to_parquet(
        tdir / "daily_20261026.parquet", index=False
    )
    monkeypatch.setattr(td, "_today_str", lambda: "20261028")
    dataio.refresh_caches()
    td._close_maps.cache_clear()

    orders = dataio.latest_orders()
    assert list(orders["symbol"]) == ["600000.SH"]  # 统一来源读到会话产物
    flow = td._compute_flow(pd.DataFrame())
    lamp, txt = flow["s1"]
    assert lamp == "ok" and "20261026" in txt and sid in txt
    assert flow["s2"] == ("warn", "待执行 0/1")
    assert flow["s3"][0] == "warn"  # 未确认成交
    assert flow["can_commit"] is True
