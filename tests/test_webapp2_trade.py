"""交易页：纯逻辑（订单/风控/账户/目标/流程）、订单解释、写操作防呆、页面渲染。

纯逻辑与解释用例移植自旧交易台（tests/test_trading_desk.py）的同一批真实场景，
口径不变：风控拆层复用 risk.compute_risk_scale，订单文案覆盖 decide_orders 的全部 reason。
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

import run_registry as rr
from webapp2 import actions, data, explain, trade_logic as tl
from webapp2.pages import trade

from tests.test_webapp2_pages import ids_of, text_of  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    monkeypatch.setattr(actions, "today_str", lambda: "20261109")
    data.refresh()
    yield tmp_path
    data.refresh()


COLS = ["symbol", "side", "shares", "limit_price", "ref_close", "reason", "blocked", "asof_date"]


def orders_df():
    """1 卖 + 2 买（一笔缺限价回落参考价）+ 1 拦截买。"""
    return pd.DataFrame(
        [
            ["000001.SZ", "sell", 1000, 12.0, 11.8, "再平衡:清仓出池", False, "2026-11-06"],
            ["600000.SH", "buy", 500, 9.0, 8.9, "再平衡:建仓", False, "2026-11-06"],
            ["600001.SH", "buy", 300, np.nan, 10.0, "再平衡:建仓", False, "2026-11-06"],
            ["600002.SH", "buy", 200, 20.0, 19.5, "涨停拦截:再平衡:建仓", True, "2026-11-06"],
        ],
        columns=COLS,
    )


def put_orders(sandbox, df=None, day="20261106"):
    (df if df is not None else orders_df()).to_csv(
        sandbox / "outputs" / f"orders_{day}.csv", index=False
    )
    data.refresh()


def put_daily(sandbox, *days):
    d = sandbox / "data_cache" / "tushare"
    d.mkdir(parents=True, exist_ok=True)
    for day in days:
        pd.DataFrame({"symbol": ["000001.SZ"], "close": [11.0]}).to_parquet(
            d / f"daily_{day}.parquet"
        )
    data.refresh()


# ----------------------------------------------------------------------------- 订单统计
def test_order_amounts_excludes_blocked_and_falls_back_to_ref_price():
    out = tl.order_amounts(orders_df())
    assert (out["n_sell"], out["n_buy"], out["n_blocked"]) == (1, 2, 1)
    assert out["sell_amt"] == pytest.approx(12000.0)
    assert out["buy_amt"] == pytest.approx(500 * 9.0 + 300 * 10.0)  # 缺限价 → 参考价；拦截不计
    assert out["asof"] == "2026-11-06"
    assert tl.order_amounts(pd.DataFrame())["n_buy"] == 0 and tl.order_amounts(None)["n_sell"] == 0


def test_blocked_as_string_from_csv_roundtrip():
    df = orders_df()
    df["blocked"] = df["blocked"].astype(str)
    assert tl.order_amounts(df)["n_blocked"] == 1


# ----------------------------------------------------------------------------- 执行进度
def test_exec_progress_counts_only_actionable_orders():
    state = {"sell-000001.SZ": "done", "buy-600000.SH": "rejected", "buy-600002.SH": "done"}
    # 拦截单（600002）的勾选不计；拒绝单不算应执行
    assert tl.exec_progress(orders_df(), state) == (1, 2, 1)
    assert tl.exec_progress(orders_df(), {}) == (0, 3, 0)
    assert tl.exec_progress(pd.DataFrame(), state) == (0, 0, 0)


def test_exec_progress_accepts_legacy_true_marker():
    assert tl.exec_progress(orders_df(), {"sell-000001.SZ": True})[0] == 1


def test_orders_fresh_matches_overview_rule():
    assert tl.orders_fresh("20261113", "20261113")
    assert not tl.orders_fresh("20261106", "20261113")  # 订单早于行情最新日 → 过期
    assert tl.orders_fresh("20261106", None)  # 无行情缓存无从比较
    assert not tl.orders_fresh(None, "20261113")


# ----------------------------------------------------------------------------- 流程四步
def _acc(**kw):
    base = {"exists": True, "stale": False, "cash": 30000.0, "age_days": 0.5, "max_age_days": 4}
    base.update(kw)
    return base


def _flow(**kw):
    args = dict(
        orders_day="20261113",
        data_day="20261113",
        progress=(0, 3, 0),
        has_orders=True,
        orders_mtime=100.0,
        holdings_mtime=None,
        account=_acc(),
    )
    args.update(kw)
    return {s["key"]: s for s in tl.flow_steps(**args)}


def test_flow_happy_path_and_each_state():
    f = _flow()
    assert f["gen"]["state"] == "ok" and f["exec"]["state"] == "warn" and "0/3" in f["exec"]["text"]
    assert f["fill"]["state"] == "warn" and f["cash"]["state"] == "ok"
    f = _flow(progress=(3, 3, 0), holdings_mtime=200.0)
    assert f["exec"]["state"] == "ok" and f["fill"]["state"] == "ok"
    assert "已确认" in f["fill"]["text"]


def test_flow_stale_orders_block_execution_and_confirmation():
    f = _flow(orders_day="20261106")
    assert f["gen"]["state"] == "warn" and "已过期" in f["gen"]["text"]
    assert f["exec"]["state"] == "off" and f["fill"]["state"] == "off"


def test_flow_no_orders_and_account_states():
    f = _flow(orders_day=None, has_orders=False, account=_acc(exists=False, cash=None))
    assert f["gen"]["state"] == "warn" and f["cash"]["state"] == "off"
    f = _flow(account=_acc(stale=True, age_days=10.0))
    assert f["cash"]["state"] == "warn" and "10 天" in f["cash"]["text"]
    f = _flow(progress=(0, 0, 2))
    assert f["exec"]["state"] == "ok" and "全部拒绝" in f["exec"]["text"]


# ----------------------------------------------------------------------------- 风控五层
def _risk_state(tvs, icq=None):
    return {"total_values": list(tvs), "ic_queue": list(icq or [])}


def test_risk_layers_empty_and_short_history():
    assert tl.risk_layers(None, iv_status="disabled") is None
    assert tl.risk_layers({}, iv_status="disabled") is None
    assert tl.risk_layers(_risk_state([]), iv_status="disabled") is None
    r = tl.risk_layers(_risk_state([1e6, 1.01e6]), iv_status="disabled")
    by = {x["key"]: x for x in r["layers"]}
    assert r["total"] == pytest.approx(1.0)
    assert by["vol"]["status"] == by["dd"]["status"] == by["ic"]["status"] == "unknown"
    assert by["regime"]["status"] == "disabled"


def test_risk_layers_vol_target_math():
    # 日收益 ±2% 交替 → 年化波动≈31.7% 远高于目标 → 波动层降仓
    tvs = 1e6 * np.cumprod(1 + np.tile([0.02, -0.02], 15))
    r = tl.risk_layers(_risk_state(tvs), iv_status="disabled")
    by = {x["key"]: x for x in r["layers"]}
    expect = min(1.0, (0.15 / np.sqrt(252.0)) / 0.02)
    assert by["vol"]["scale"] == pytest.approx(expect, rel=1e-6)
    assert by["vol"]["status"] == "danger"
    assert by["dd"]["scale"] == pytest.approx(1.0)
    assert r["total"] == pytest.approx(expect, rel=1e-6)


def test_risk_layers_drawdown_brake():
    r = tl.risk_layers(_risk_state([1e6] * 10 + [0.75e6] * 5), iv_status="disabled")
    by = {x["key"]: x for x in r["layers"]}
    assert by["dd"]["scale"] == pytest.approx(0.5) and by["dd"]["status"] == "danger"
    assert "-25.0%" in by["dd"]["reason"]


def test_risk_layers_ic_scale_and_insufficient_samples():
    tvs = [1e6] * 10
    neg = tl.risk_layers(_risk_state(tvs, [-0.02] * 20), iv_status="disabled")
    assert {x["key"]: x for x in neg["layers"]}["ic"]["scale"] == pytest.approx(0.0)
    assert neg["total"] == pytest.approx(0.0)
    pos = tl.risk_layers(_risk_state(tvs, [0.03] * 20), iv_status="disabled")
    assert {x["key"]: x for x in pos["layers"]}["ic"]["status"] == "ok"
    few = tl.risk_layers(_risk_state(tvs, [-0.05] * 4), iv_status="disabled")
    assert {x["key"]: x for x in few["layers"]}["ic"]["status"] == "unknown"
    assert few["total"] == pytest.approx(1.0)  # 样本不足的层不参与叠乘


def test_risk_layers_iv():
    tvs = [1e6] * 10
    unk = tl.risk_layers(_risk_state(tvs), iv_scale=None, iv_status="unknown")
    assert {x["key"]: x for x in unk["layers"]}["iv"]["status"] == "unknown"
    ok = tl.risk_layers(_risk_state(tvs), iv_scale=0.75, iv_status="ok")
    assert {x["key"]: x for x in ok["layers"]}["iv"]["status"] == "warn"
    assert ok["total"] == pytest.approx(0.75)


# ----------------------------------------------------------------------------- 账户 / 目标
HOLD = pd.DataFrame(
    {"symbol": ["000001.SZ"], "shares": [1000], "entry_date": ["2026-10-01"], "ref_close": [10.0]}
)


def test_account_snapshot():
    snap = tl.account_snapshot(100000.0, HOLD, {"000001.SZ": 11.0}, {"000001.SZ": 10.5})
    assert snap["mv"] == pytest.approx(11000.0) and snap["nav"] == pytest.approx(111000.0)
    assert snap["pnl"] == pytest.approx(500.0)
    assert tl.account_snapshot(100000.0, HOLD, {"000001.SZ": 11.0}, {})["pnl"] is None
    assert tl.account_snapshot(None, HOLD, {}, {}) is None


def test_target_vs_current_applies_orders_but_not_blocked_or_rejected():
    orders = pd.DataFrame(
        [
            ["000001.SZ", "sell", 400, 11.0, 10.8, False],
            ["600000.SH", "buy", 200, 5.0, 4.9, False],
            ["600001.SH", "buy", 100, 8.0, 7.9, True],  # 拦截
            ["600003.SH", "buy", 100, 6.0, 5.9, False],  # 将被拒绝
        ],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked"],
    )
    tvc = tl.target_vs_current(
        orders, HOLD, {"000001.SZ": 11.0}, 11000.0, skip_keys={"buy-600003.SH"}
    ).set_index("symbol")
    assert tvc.loc["000001.SZ", "tgt_shares"] == 600
    assert tvc.loc["000001.SZ", "cur_w"] == pytest.approx(1000 * 11 / 11000.0)
    assert tvc.loc["600000.SH", "tgt_w"] == pytest.approx(200 * 5 / 11000.0)
    assert "600001.SH" not in tvc.index and "600003.SH" not in tvc.index
    assert tl.target_vs_current(pd.DataFrame(), pd.DataFrame(), {}, None).empty


# ----------------------------------------------------------------------------- 成交确认预览
def test_confirm_plan_explains_every_refusal():
    df = orders_df()
    ok = tl.confirm_plan(df, {"buy-600000.SH": "rejected"}, fresh=True, confirmed=False)
    assert ok["can"] and (ok["n_sell"], ok["n_buy"], ok["n_rej"], ok["n_blocked"]) == (1, 1, 1, 1)
    assert "过期" in tl.confirm_plan(df, {}, fresh=False, confirmed=False)["why"]
    assert "重复加仓" in tl.confirm_plan(df, {}, fresh=True, confirmed=True)["why"]
    allrej = {
        "sell-000001.SZ": "rejected",
        "buy-600000.SH": "rejected",
        "buy-600001.SH": "rejected",
    }
    assert "全部被拦截或拒绝" in tl.confirm_plan(df, allrej, True, False)["why"]
    assert not tl.confirm_plan(pd.DataFrame(), {}, True, False)["can"]


# ----------------------------------------------------------------------------- 写操作防呆
def test_confirm_fills_refuses_stale_orders(sandbox):
    put_orders(sandbox, day="20261106")
    put_daily(sandbox, "20261106", "20261113")
    ok, msg = actions.confirm_fills()
    assert not ok and "已过期" in msg and "20261113" in msg
    assert not (sandbox / "outputs" / "holdings.csv").exists()


def test_confirm_fills_refuses_double_confirmation_unless_forced(sandbox):
    put_orders(sandbox)
    put_daily(sandbox, "20261106")
    assert actions.confirm_fills()[0]
    h1 = pd.read_csv(sandbox / "outputs" / "holdings.csv").set_index("symbol")
    os.utime(sandbox / "outputs" / "holdings.csv", (time.time() + 5, time.time() + 5))
    data.refresh()
    ok, msg = actions.confirm_fills()
    assert not ok and "重复加仓" in msg
    h2 = pd.read_csv(sandbox / "outputs" / "holdings.csv").set_index("symbol")
    assert (h1["shares"] == h2["shares"]).all()  # 被拒绝的二次确认没有改持仓
    assert actions.confirm_fills(force=True)[0]
    h3 = pd.read_csv(sandbox / "outputs" / "holdings.csv").set_index("symbol")
    assert h3.loc["600000.SH", "shares"] == 2 * h1.loc["600000.SH", "shares"]


def test_launch_nightly_is_singleton(sandbox, monkeypatch):
    """夜间任务并发会同时改 holdings / 风控状态 / 订单文件，已有一个在跑就拒绝第二个。"""
    root = str(sandbox / "outputs" / "runs")
    live = rr.RunReporter("nightly", label="夜间", run_id="n_live", root=root)
    live.step("model")
    ok, msg, rid = actions.launch_cli("nightly", ["--capital", "30000"])
    assert not ok and "n_live" in msg and rid == "n_live"
    # 被拒绝的启动不能留下新的运行目录（否则列表里会多一条永远不会结束的空运行）
    assert sorted(p.name for p in (sandbox / "outputs" / "runs").iterdir()) == ["n_live"]
    live.finish()
    # 前一个结束后可以再启动（这里只验证不再被单例拦截，用假入口避免真的跑夜间任务）
    script = sandbox / "fake_cli.py"
    script.write_text("print('ok')")
    monkeypatch.setitem(actions.CLI_ENTRIES, "nightly", [str(script)])
    monkeypatch.setattr(actions, "PY", __import__("sys").executable)
    ok2, msg2, rid2 = actions.launch_cli("nightly", [])
    assert ok2 and rid2 != "n_live", msg2


# ----------------------------------------------------------------------------- 订单解释
def row(**kw):
    base = dict(
        symbol="002611.SZ",
        side="sell",
        shares=1200,
        limit_price=14.889,
        ref_close=13.39,
        reason="跌出ML持有带",
        rank_pct=0.6572,
        bb_pos=0.439,
        rsi=43.4,
        blocked=False,
    )
    base.update(kw)
    return base


def full_ctx():
    return {
        "params": {"topk": 0.15, "order_price_mode": "limit_bb"},
        "n_stocks": 5476,
        "weights": {"gbm_l2": 0.5, "ic_linear": 0.5},
        "pred": {
            "002611.SZ": {
                "score": 0.4584,
                "prev_score": 0.4631,
                "legs": {"gbm_l2": -0.0085, "ic_linear": 0.7271},
                "date": "2026-10-27",
            }
        },
        "holdings": {"002611.SZ": {"shares": 1200, "entry_date": "2026-10-10", "ref_close": 12.0}},
    }


def secs(r, ctx=None):
    return {s["title"]: " ".join(s["lines"]) for s in explain.order_explain(r, ctx)}


def test_explain_sell_out_of_hold_band_full_chain():
    s = secs(row(), full_ctx())
    assert "第 3,599/5,476 名（前 65.7%）" in s["① 触发规则：跌出ML持有带（排名衰减卖出）"]
    assert "非择时卖出" in s["② 择时检查"] and "43.4" in s["② 择时检查"]
    model = s["③ 模型明细（各子模型分别打分）"]
    assert "AI 树模型 -0.0085（偏弱）" in model and "因子加权模型 +0.7271（偏多）" in model
    assert "权重 AI 树模型 50% · 因子加权模型 50% → 融合分 0.458（前一日 0.463，略降）" in model
    assert "gbm_l2" not in model  # 内部键不外露
    param = s["④ 订单参数"]
    assert "卖出 1,200 股（全部持仓）" in param and "布林上轨" in param
    assert "建仓日 2026-10-10" in s["⑤ 持仓信息"] and "+11.6%" in s["⑤ 持仓信息"]


@pytest.mark.parametrize(
    "reason,kw,title,needle",
    [
        ("RSI超买(85)", {"rsi": 85.0}, "① 触发规则：RSI 超买（择时卖出）", "超买线 80"),
        ("触布林上轨", {"bb_pos": 0.97}, "① 触发规则：触布林上轨（择时卖出）", "布林上轨"),
        ("再平衡:清仓出池", {}, "① 触发规则：再平衡清仓（跌出目标池）", "目标池"),
        ("再平衡:减仓", {}, "① 触发规则：再平衡减仓（超配回落）", "减仓"),
        ("再平衡:加仓", {"side": "buy"}, "① 触发规则：再平衡加仓（低配补足）", "加仓"),
    ],
)
def test_explain_trigger_reasons(reason, kw, title, needle):
    assert needle in secs(row(reason=reason, **kw), {})[title]


def test_explain_timing_sell_vs_rank_driven():
    assert "非择时卖出" not in secs(row(reason="RSI超买(85)", rsi=85.0), {})["② 择时检查"]
    assert "非择时卖出" in secs(row(reason="再平衡:清仓出池"), {})["② 择时检查"]


def test_explain_buy_paths():
    buy = row(
        side="buy", shares=200, limit_price=5.0, ref_close=5.0, reason="再平衡:建仓", rank_pct=0.05
    )
    s = secs(buy, {"n_stocks": 5476})
    assert "第 274/5,476 名" in s["① 触发规则：再平衡建仓（进入目标池）"]
    assert "未启用指标择时门控" in s["② 择时检查"] and "市价单" in s["④ 订单参数"]
    pull = row(
        side="buy",
        reason="布林下轨回踩",
        rank_pct=0.03,
        bb_pos=0.05,
        rsi=35.0,
        limit_price=8.5,
        ref_close=8.9,
    )
    s = secs(pull, {"n_stocks": 5476})
    assert "下轨" in s["① 触发规则：布林下轨回踩（排名+择时）"]
    assert "择时门槛通过" in s["② 择时检查"] and "贴近下轨" in s["② 择时检查"]


def test_explain_blocked_orders_lead_with_the_reason():
    up = explain.order_explain(row(side="buy", reason="涨停拦截:布林下轨回踩", blocked="True"), {})
    assert (
        up[0]["title"] == "⛔ 拦截说明"
        and "涨停" in up[0]["lines"][0]
        and "不执行" in up[0]["lines"][0]
    )
    assert up[1]["title"].startswith("① 触发规则：布林下轨回踩")
    down = explain.order_explain(row(reason="跌停拦截:跌出ML持有带", blocked=True), {})
    assert "跌停" in down[0]["lines"][0] and "跌出ML持有带" in down[1]["title"]


def test_explain_degrades_with_empty_context():
    r = row(
        reason="跌出横截面:清退",
        rank_pct=None,
        bb_pos=None,
        rsi=None,
        limit_price=13.39,
        ref_close=13.39,
    )
    s = secs(r, {})
    assert "停牌" in s["① 触发规则：跌出横截面（清退）"] and "市价单" in s["④ 订单参数"]
    assert (
        "② 择时检查" not in s
        and "③ 模型明细（各子模型分别打分）" not in s
        and "⑤ 持仓信息" not in s
    )
    assert "全市场前 65.7%" in secs(row(), {})["① 触发规则：跌出ML持有带（排名衰减卖出）"]
    assert explain.order_explain({}, None)  # 空订单行也不抛


def test_explain_context_reads_pred_hist_and_holdings(sandbox):
    pd.DataFrame(
        [
            ["2026-10-26", "002611.SZ", 0.4631, -0.0074, 0.6323],
            ["2026-10-27", "002611.SZ", 0.4584, -0.0085, 0.7271],
            ["2026-10-27", "600000.SH", 0.9, 0.5, 0.6],
        ],
        columns=["date", "symbol", "score", "pred_gbm_l2", "pred_ic_linear"],
    ).to_parquet(sandbox / "outputs" / "live_risk_state_pred_hist.parquet", index=False)
    HOLD.assign(symbol="002611.SZ", shares=1200).to_csv(
        sandbox / "outputs" / "holdings.csv", index=False
    )
    data.refresh()
    orders = pd.DataFrame([["002611.SZ", "sell"]], columns=["symbol", "side"])
    ctx = explain.explain_context(orders, weights={"gbm_l2": 0.5, "ic_linear": 0.5})
    assert ctx["n_stocks"] == 2 and ctx["weights"] == {"gbm_l2": 0.5, "ic_linear": 0.5}
    p = ctx["pred"]["002611.SZ"]
    assert p["score"] == pytest.approx(0.4584) and p["prev_score"] == pytest.approx(0.4631)
    assert p["legs"]["gbm_l2"] == pytest.approx(-0.0085)
    assert "600000.SH" not in ctx["pred"]  # 只组装订单里出现的股票
    assert ctx["holdings"]["002611.SZ"]["entry_date"] == "2026-10-01"
    assert explain.explain_context(pd.DataFrame())["pred"] == {}


# ----------------------------------------------------------------------------- 页面
def test_trade_page_renders_on_empty_disk(sandbox):
    t = text_of(trade.layout())
    for needle in ("没有订单", "还没有净值历史", "现金未维护", "当前空仓", "生成次日订单"):
        assert needle in t, needle


def test_trade_page_renders_orders_with_marks_and_explanations(sandbox):
    (sandbox / "outputs" / "account.csv").write_text("cash\n30000\n")
    put_orders(sandbox)
    put_daily(sandbox, "20261106")
    actions.set_execution_mark("20261109", "buy", "600000.SH", "rejected")
    actions.set_execution_mark("20261109", "sell", "000001.SZ", "done")
    body = trade.body()
    t = text_of(body)
    for needle in (
        "卖单：先卖腾现金",
        "买单：后买",
        "000001.SZ",
        "✓ 已执行",
        "✗ 已拒绝",
        "拦截 · 不执行",
    ):
        assert needle in t, needle
    assert "① 触发规则：再平衡清仓" in t  # 「分析」面板随订单渲染
    ids = [i for i in ids_of(body)]
    assert "confirm-open" in ids


def test_execution_buttons_are_pattern_ids_and_skip_blocked(sandbox):
    put_orders(sandbox)
    body = trade.body()
    found = []

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            [walk(c) for c in n]
            return
        i = getattr(n, "id", None)
        if isinstance(i, dict) and i.get("type") == "mark":
            found.append((i["key"], i["to"]))
        walk(getattr(n, "children", None))

    walk(body)
    keys = {k for k, _ in found}
    assert keys == {"sell-000001.SZ", "buy-600000.SH", "buy-600001.SH"}  # 拦截单没有按钮
    assert {to for _, to in found} == {"done", "rejected"}


def test_stale_orders_are_labelled_and_cannot_be_confirmed(sandbox):
    put_orders(sandbox, day="20261106")
    put_daily(sandbox, "20261106", "20261113")
    t = text_of(trade.body())
    assert "已过期，勿执行" in t and "当前不可确认" in t


def test_apply_mark_toggles_and_is_exclusive(sandbox):
    assert trade.apply_mark("buy-600000.SH", "done")[0]
    assert data.execution_state("20261109") == {"buy-600000.SH": "done"}
    assert trade.apply_mark("buy-600000.SH", "rejected")[0]  # 互斥：拒绝覆盖执行
    assert data.execution_state("20261109") == {"buy-600000.SH": "rejected"}
    assert trade.apply_mark("buy-600000.SH", "rejected")[0]  # 再点一次撤销
    assert data.execution_state("20261109") == {}
    assert not trade.apply_mark("hold-X", "done")[0]
    assert not trade.apply_mark("buy-X", "weird")[0]
    assert not trade.apply_mark("nodash", "done")[0]


def test_rejected_mark_is_visible_to_cli_commit_reader(sandbox):
    """前端写下的拒绝，必须被 CLI --commit 用的 load_rejected_rids 读到（同一份文件契约）。"""
    from execution import load_rejected_rids

    trade.apply_mark("buy-600000.SH", "rejected")
    assert load_rejected_rids(str(sandbox / "outputs"), "20261109") == {"buy-600000.SH"}


def test_forms_live_outside_the_polled_body(sandbox):
    page = trade.layout()
    ids = ids_of(page)
    assert {"trade-tick", "trade-body", "trade-cash", "trade-cash-save", "trade-gen"} <= ids
    assert {"confirm-run", "confirm-cancel", "confirm-panel"} <= ids
    assert not ({"trade-cash", "trade-cash-save", "trade-gen"} & ids_of(trade.body()))


def test_live_status_shows_progress_from_run_registry(sandbox):
    root = str(sandbox / "outputs" / "runs")
    assert "还没有夜间运行记录" in text_of(trade._live_status())
    r = rr.RunReporter("nightly", label="夜间出单", run_id="n1", root=root)
    r.step("model", "模型")
    r.progress("训练", 2, 4, "增量续训")
    t = text_of(trade._live_status())
    assert "正在生成订单" in t and "训练" in t and "n1" in t
    r.finish()
    assert "最近一次夜间运行" in text_of(trade._live_status())


def test_live_status_lives_outside_polled_body_but_is_refreshed_separately(sandbox):
    ids = ids_of(trade.layout())
    assert {"trade-live", "trade-sig", "trade-gen"} <= ids
    assert "trade-live" not in ids_of(trade.body())


# ----------------------------------------------------------------------------- 刷新门控
def test_refresh_is_skipped_when_nothing_on_disk_changed(sandbox):
    """每 3 秒无条件重绘会把用户刚展开的「分析」面板收回去（浏览器里实测复现过）。
    签名不变就必须不重绘，DOM 原样保留。"""
    put_orders(sandbox)
    new, sig = trade.refresh_decision(None)
    assert new is not None  # 首次（无已存签名）要绘制
    again, sig2 = trade.refresh_decision(sig)
    assert again is None and sig2 == sig


@pytest.mark.parametrize(
    "touch",
    [
        lambda sb: (sb / "outputs" / "account.csv").write_text("cash\n1\n"),
        lambda sb: (sb / "outputs" / "holdings.csv").write_text("symbol,shares\nA,100\n"),
        lambda sb: (sb / "outputs" / "execution_state_20261109.json").write_text("{}"),
        lambda sb: (sb / "outputs" / "live_risk_state.json").write_text("{}"),
        lambda sb: put_orders(sb, day="20261110"),
        lambda sb: put_daily(sb, "20261110"),
    ],
    ids=["account", "holdings", "exec_state", "risk_state", "new_orders", "new_market_day"],
)
def test_refresh_fires_when_a_dependency_changes(sandbox, touch):
    """CLI 在另一个进程里改了任何一个依赖文件，页面都必须跟着更新，不能因为门控而漏刷。"""
    put_orders(sandbox)
    _, sig = trade.refresh_decision(None)
    touch(sandbox)
    data.refresh()
    new, sig2 = trade.refresh_decision(sig)
    assert new is not None and sig2 != sig


def test_refresh_fires_when_a_nightly_run_finishes(sandbox):
    root = str(sandbox / "outputs" / "runs")
    _, sig = trade.refresh_decision(None)
    with rr.RunReporter("nightly", run_id="n3", root=root) as r:
        r.set_param(model_weights={"gbm_l2": 1.0})
    new, _ = trade.refresh_decision(sig)
    assert new is not None


def test_nightly_params_expose_model_weights_for_explanations(sandbox):
    root = str(sandbox / "outputs" / "runs")
    with rr.RunReporter("nightly", run_id="n2", root=root) as r:
        r.set_param(model_weights={"gbm_l2": 0.6, "ic_linear": 0.4})
    assert trade._latest_nightly_params()["model_weights"] == {"gbm_l2": 0.6, "ic_linear": 0.4}


def test_app_routes_trade_and_nav_has_it(sandbox):
    from webapp2 import app as appmod

    assert ("/trade", "交易") in appmod.NAV
    assert "生成次日订单" in text_of(appmod.route("/trade"))
    assert appmod.nav_active("/trade", "/trade")


def _find_by_id(node, wanted):
    if node is None or isinstance(node, (str, int, float)):
        return None
    if isinstance(node, (list, tuple)):
        for c in node:
            r = _find_by_id(c, wanted)
            if r is not None:
                return r
        return None
    if getattr(node, "id", None) == wanted:
        return node
    return _find_by_id(getattr(node, "children", None), wanted)


def test_cash_inputs_accept_any_amount_not_just_multiples_of_100(sandbox):
    """回归：step=100 时浏览器把 31234.5 判为无效，Dash 传 None，保存报「请输入数字」。
    浏览器级复现过（scripts/check_confirm_flow.py）。两个页面的现金输入都必须是 step="any"。"""
    from webapp2.pages import overview

    assert _find_by_id(trade.layout(), "trade-cash").step == "any"
    assert _find_by_id(overview.layout(), "cash-input").step == "any"


# ----------------------------------------------------------------------------- IV 降仓取数
def _put_panel_cache(sandbox, vix):
    d = sandbox / "data_cache"
    d.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2026-01-01", periods=len(vix))
    pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "symbol": ["A", "B"] * len(vix),
            "opt_vix_like": np.repeat(vix, 2),
        }
    ).to_parquet(d / "factor_panel_cache.parquet")
    data.refresh()


def test_iv_layer_reads_the_panel_cache_the_nightly_run_uses(sandbox):
    """回归：旧前端读 wholemkt_panel.parquet（本仓库不存在），IV 层永远「数据不可得」，
    总乘数漏掉这一层。现在读夜间任务实际使用的 factor_panel_cache.parquet。"""
    import config as C

    _put_panel_cache(sandbox, np.linspace(10.0, 30.0, 80))  # 最新值是窗口内最高 → 满额降仓
    status, scale = data.iv_scale_now()
    assert status == "ok" and scale == pytest.approx(C.IV_MIN_SCALE)
    r = tl.risk_layers(_risk_state([1e6] * 10), iv_scale=scale, iv_status=status)
    iv = {x["key"]: x for x in r["layers"]}["iv"]
    assert iv["status"] in ("warn", "danger") and r["total"] == pytest.approx(C.IV_MIN_SCALE)


def test_iv_layer_unknown_when_cache_missing_or_lacks_the_feature(sandbox):
    assert data.iv_scale_now() == ("unknown", None)
    d = sandbox / "data_cache"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=5), "x": range(5)}).to_parquet(
        d / "factor_panel_cache.parquet"
    )
    data.refresh()
    assert data.iv_scale_now() == ("unknown", None)
