"""交易台页面单元测试：

· 纯函数数值正确性：order_amounts（预计成交金额）、compute_risk_layers
  （风控五层乘数）、account_snapshot（净值/当日盈亏）、target_vs_current
  （目标权重推导）；
· 空态容错：无任何实盘产物时 layout() 渲染引导卡、不抛异常；
· 非空态：tmp_path 构造 outputs 产物，layout() 渲染出关键数字。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dash import Dash, dcc, html, dash_table
from dash.development.base_component import Component
import dash_bootstrap_components as dbc

from pipeline import registry
from trading import parse_initial_holdings
from webapp import dataio
from webapp.pages import trading_desk as td


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _texts(comp) -> list[str]:
    """递归收集组件树里的全部字符串。"""
    out = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, Component):
            walk(getattr(x, "children", None))
            props = getattr(x, "props", None) or {}
            for k in ("title",):
                if isinstance(props.get(k), str):
                    out.append(props[k])

    walk(comp)
    return out


def _orders_df() -> pd.DataFrame:
    """2 买（一笔缺 limit_price 回落 ref_close）+ 1 卖 + 1 拦截买。"""
    return pd.DataFrame(
        [
            # symbol, name, side, shares, limit_price, ref_close, reason, blocked, asof_date
            [
                "000001.SZ",
                "平安银行",
                "sell",
                1000,
                12.0,
                11.8,
                "再平衡:清仓出池",
                False,
                "2026-10-22",
            ],
            ["600000.SH", "浦发银行", "buy", 500, 9.0, 8.9, "再平衡:建仓", False, "2026-10-22"],
            ["600001.SH", "测试股A", "buy", 300, np.nan, 10.0, "再平衡:建仓", False, "2026-10-22"],
            [
                "600002.SH",
                "测试股B",
                "buy",
                200,
                20.0,
                19.5,
                "涨停拦截:再平衡:建仓",
                True,
                "2026-10-22",
            ],
        ],
        columns=[
            "symbol",
            "name",
            "side",
            "shares",
            "limit_price",
            "ref_close",
            "reason",
            "blocked",
            "asof_date",
        ],
    )


# ---------------------------------------------------------------------------
# order_amounts：预计成交金额
# ---------------------------------------------------------------------------
def test_order_amounts_empty():
    out = td.order_amounts(pd.DataFrame())
    assert out == {
        "n_buy": 0,
        "n_sell": 0,
        "n_blocked": 0,
        "buy_amt": 0.0,
        "sell_amt": 0.0,
        "asof": None,
    }
    assert td.order_amounts(None)["n_buy"] == 0


def test_order_amounts_excludes_blocked_and_falls_back_price():
    out = td.order_amounts(_orders_df())
    assert out["n_sell"] == 1 and out["n_buy"] == 2 and out["n_blocked"] == 1
    assert out["sell_amt"] == pytest.approx(1000 * 12.0)
    # 600001.SH 无 limit_price → ref_close 10.0；拦截的 600002.SH 不计入
    assert out["buy_amt"] == pytest.approx(500 * 9.0 + 300 * 10.0)
    assert out["asof"] == "2026-10-22"


def test_order_amounts_blocked_as_string():
    df = _orders_df()
    df["blocked"] = df["blocked"].astype(str)  # csv 读回可能是 "True"/"False"
    out = td.order_amounts(df)
    assert out["n_blocked"] == 1 and out["n_buy"] == 2


# ---------------------------------------------------------------------------
# compute_risk_layers：五层乘数（复用 src/risk.py 唯一实现拆解）
# ---------------------------------------------------------------------------
def _state(tvs, icq=None):
    return {
        "dates": [f"2026-09-{i + 1:02d}" for i in range(len(tvs))],
        "total_values": list(tvs),
        "ic_queue": list(icq or []),
    }


def test_risk_layers_empty_state():
    assert td.compute_risk_layers(None, iv_status="disabled") is None
    assert td.compute_risk_layers({}, iv_status="disabled") is None
    assert td.compute_risk_layers(_state([]), iv_status="disabled") is None


def test_risk_layers_short_history_unknown():
    r = td.compute_risk_layers(_state([1e6, 1.01e6]), iv_status="disabled")
    assert r["total"] == pytest.approx(1.0)
    by_key = {x["key"]: x for x in r["layers"]}
    assert by_key["vol"]["status"] == "unknown"  # 净值历史 <6 日
    assert by_key["dd"]["status"] == "unknown"
    assert by_key["ic"]["status"] == "unknown"  # IC 样本不足
    assert by_key["regime"]["status"] == "disabled"


def test_risk_layers_vol_target_math():
    # 日收益 ±2% 交替 → 实现波动年化 ≈ 31.7% > 目标 15% → vol 层降仓
    rets = np.tile([0.02, -0.02], 15)
    tvs = 1e6 * np.cumprod(1 + rets)
    r = td.compute_risk_layers(_state(tvs), iv_status="disabled")
    by_key = {x["key"]: x for x in r["layers"]}
    expect = min(1.0, (0.15 / np.sqrt(252.0)) / 0.02)
    assert by_key["vol"]["scale"] == pytest.approx(expect, rel=1e-6)
    assert by_key["vol"]["status"] == "danger"  # ≈0.47 < 0.5
    assert by_key["dd"]["scale"] == pytest.approx(1.0)  # 回撤远未及 -20%
    assert r["total"] == pytest.approx(expect, rel=1e-6)


def test_risk_layers_dd_brake_math():
    # 平稳后单日 -25% → 回撤超阈值触发 ×0.5；dd 层独立于 vol 层拆出
    tvs = [1e6] * 10 + [0.75e6] * 5
    r = td.compute_risk_layers(_state(tvs), iv_status="disabled")
    by_key = {x["key"]: x for x in r["layers"]}
    assert by_key["dd"]["scale"] == pytest.approx(0.5)
    assert by_key["dd"]["status"] == "danger"
    assert "-25.0%" in by_key["dd"]["reason"]


def test_risk_layers_ic_scale_math():
    tvs = [1e6] * 10  # 无波动 → vol/dd 不触发
    neg = td.compute_risk_layers(_state(tvs, [-0.02] * 20), iv_status="disabled")
    by_key = {x["key"]: x for x in neg["layers"]}
    # IC 均值 -0.02 ≤ 清零线 -0.01 → ic_scale = IC_MIN_SCALE = 0.0
    assert by_key["ic"]["scale"] == pytest.approx(0.0)
    assert by_key["ic"]["status"] == "danger"
    assert neg["total"] == pytest.approx(0.0)

    pos = td.compute_risk_layers(_state(tvs, [0.03] * 20), iv_status="disabled")
    by_key = {x["key"]: x for x in pos["layers"]}
    assert by_key["ic"]["scale"] == pytest.approx(1.0)
    assert by_key["ic"]["status"] == "ok"

    # 样本不足（4 < max(3, 20//4)=5）→ unknown，不参与叠乘
    few = td.compute_risk_layers(_state(tvs, [-0.05] * 4), iv_status="disabled")
    assert {x["key"]: x for x in few["layers"]}["ic"]["status"] == "unknown"
    assert few["total"] == pytest.approx(1.0)


def test_risk_layers_iv_lamp():
    tvs = [1e6] * 10
    unk = td.compute_risk_layers(_state(tvs), iv_scale=None, iv_status="unknown")
    assert {x["key"]: x for x in unk["layers"]}["iv"]["status"] == "unknown"
    ok = td.compute_risk_layers(_state(tvs), iv_scale=0.75, iv_status="ok")
    iv = {x["key"]: x for x in ok["layers"]}["iv"]
    assert iv["scale"] == pytest.approx(0.75) and iv["status"] == "warn"
    assert ok["total"] == pytest.approx(0.75)  # IV 参与总乘数


# ---------------------------------------------------------------------------
# account_snapshot / target_vs_current
# ---------------------------------------------------------------------------
def test_account_snapshot():
    acct = pd.DataFrame({"date": ["2026-10-22"], "cash": [100000.0]})
    holds = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "shares": [1000],
            "entry_date": ["2026-10-01"],
            "ref_close": [10.0],
        }
    )
    snap = td.account_snapshot(acct, holds, {"000001.SZ": 11.0}, {"000001.SZ": 10.5})
    assert snap["cash"] == 100000.0
    assert snap["mv"] == pytest.approx(11000.0)
    assert snap["nav"] == pytest.approx(111000.0)
    assert snap["pnl"] == pytest.approx(500.0)
    # 无前收 → 盈亏未知；无 account → None（空态）
    snap2 = td.account_snapshot(acct, holds, {"000001.SZ": 11.0}, {})
    assert snap2["pnl"] is None
    assert td.account_snapshot(pd.DataFrame(), holds, {}, {}) is None
    assert td.account_snapshot(pd.DataFrame({"x": [1]}), holds, {}, {}) is None


def test_target_vs_current():
    holds = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "shares": [1000],
            "entry_date": ["2026-10-01"],
            "ref_close": [10.0],
        }
    )
    orders = pd.DataFrame(
        [
            ["000001.SZ", "sell", 400, 11.0, 10.8, False],  # blocked? no
            ["600000.SH", "buy", 200, 5.0, 4.9, False],  # 新仓，价取订单价
            ["600001.SH", "buy", 100, 8.0, 7.9, True],  # 拦截 → 不进目标
        ],
        columns=["symbol", "side", "shares", "limit_price", "ref_close", "blocked"],
    )
    nav = 11000.0
    tvc = td.target_vs_current(orders, holds, {"000001.SZ": 11.0}, nav)
    by_sym = tvc.set_index("symbol")
    assert by_sym.loc["000001.SZ", "tgt_shares"] == 600
    assert by_sym.loc["000001.SZ", "cur_w"] == pytest.approx(1000 * 11 / nav)
    assert by_sym.loc["000001.SZ", "tgt_w"] == pytest.approx(600 * 11 / nav)
    assert by_sym.loc["600000.SH", "tgt_w"] == pytest.approx(200 * 5 / nav)
    assert "600001.SH" not in by_sym.index  # 拦截单不产生目标仓位
    # 空态
    assert td.target_vs_current(pd.DataFrame(), pd.DataFrame(), {}, None).empty


# ---------------------------------------------------------------------------
# layout 空态 / 非空态
# ---------------------------------------------------------------------------
def _stub_is_trading_day(day):
    """测试桩：交易日判定不打真实日历/网络（带 cache_clear 兼容 refresh_caches）。"""
    return None


_stub_is_trading_day.cache_clear = lambda: None


@pytest.fixture
def tmp_artifacts(tmp_path, monkeypatch):
    """dataio 产物目录指向 tmp；IV 计算与收盘价 map 隔离真实大文件。"""
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


def test_layout_empty_state_shows_onboarding(tmp_artifacts):
    comp = td.layout()
    assert isinstance(comp, Component)
    txt = " ".join(_texts(comp))
    assert "三步开始实盘跟踪" in txt
    # onboarding：三步全部指向「今日流程」卡的网页操作
    assert "④ 现金维护" in txt
    assert "① 生成订单" in txt
    assert "③ 成交确认" in txt
    assert "风控未初始化" in txt
    assert "account.csv" in txt
    assert "当前空仓" in txt
    # onboarding 卡本身纯网页操作引导，无 CLI 文案（CLI 仅留在执行闭环折叠区）
    ob_txt = " ".join(_texts(td._onboarding_card()))
    assert "daily_run.sh" not in ob_txt and "predict_tomorrow" not in ob_txt
    # 闭环折叠卡内保留一行等价 CLI 小字（逃生通道）
    assert "./daily_run.sh" in " ".join(_texts(td._card_h()))


def test_layout_with_artifacts(tmp_artifacts, monkeypatch):
    out = tmp_artifacts
    monkeypatch.setattr(td, "_today_str", lambda: "20261022")
    _orders_df().to_csv(out / "orders_20261022.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "shares": [1000],
            "entry_date": ["2026-10-01"],
            "ref_close": [10.0],
        }
    ).to_csv(out / "holdings.csv", index=False)
    pd.DataFrame({"date": ["2026-10-22"], "cash": [100000.0]}).to_csv(
        out / "account.csv", index=False
    )
    (out / "live_risk_state.json").write_text(
        json.dumps(_state([1e6 + i * 100 for i in range(10)], [0.03] * 20)),
        encoding="utf-8",
    )
    dataio.refresh_caches()

    comp = td.layout()
    txt = " ".join(_texts(comp))
    assert "三步开始" not in txt  # 已有产物 → 不再显示引导
    assert "20261022" in txt  # 订单日期
    assert "风控总乘数" in txt and "×1.00" in txt  # 平稳净值 + 高 IC → 满仓
    assert "预计成交" in txt
    assert "卖单区" in txt and "买单区" in txt
    assert "涨停拦截" in txt  # blocked 行原因可见
    assert "现金" in txt and "净值" in txt
    assert "执行闭环" in txt and "commit" in txt
    # 今日操作卡分层：订单区小标题 + 出单区小标题
    assert "今日订单" in txt and "生成次日订单" in txt
    # 风控卡术语 tooltip（IC 定权）
    assert "IC 定权" in txt
    # 持仓明细代码 → 行情页深链
    assert any(x.href == "/market?symbol=000001.SZ" for x in _components(comp, dcc.Link))


def test_layout_stale_orders_warns(tmp_artifacts, monkeypatch):
    out = tmp_artifacts
    monkeypatch.setattr(td, "_today_str", lambda: "20261022")
    _orders_df().to_csv(out / "orders_20261020.csv", index=False)  # 前天订单
    dataio.refresh_caches()
    txt = " ".join(_texts(td.layout()))
    assert "已过期" in txt  # 订单不在当前执行周期（tmp 无日线缓存 → 回落日期=今日口径）


def test_close_maps_from_daily_files(tmp_artifacts):
    tdir = tmp_artifacts.parent / "tushare"
    tdir.mkdir()
    pd.DataFrame({"symbol": ["000001.SZ"], "close": [10.0]}).to_parquet(
        tdir / "daily_20261021.parquet", index=False
    )
    pd.DataFrame({"symbol": ["000001.SZ"], "close": [11.0]}).to_parquet(
        tdir / "daily_20261022.parquet", index=False
    )
    dataio.refresh_caches()
    td._close_maps.cache_clear()
    prev, last, d = td._close_maps()
    assert d == "20261022"
    assert prev == {"000001.SZ": 10.0} and last == {"000001.SZ": 11.0}


def test_register_wires_callback(tmp_artifacts):
    app = Dash(__name__, suppress_callback_exceptions=True)
    td.register(app)
    assert app.callback_map  # 勾选划线回调已挂


# ---------------------------------------------------------------------------
# 出单入口：参数组装 / 预填 / 进度渲染
# ---------------------------------------------------------------------------
def _ids(comp) -> list:
    """递归收集组件树里的全部 id（str 或 pattern-matching dict）。"""
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
    """递归收集组件树里指定类型的全部组件。"""
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


def test_layout_has_nextday_entry(tmp_artifacts):
    comp = td.layout()
    ids = {i for i in _ids(comp) if isinstance(i, str)}
    for cid in (
        "td-gen-open",
        "td-gen-modal",
        "td-gen-label",
        "td-gen-capital",
        "td-gen-holdings-table",
        "td-gen-addrow",
        "td-gen-prefill-note",
        "td-gen-submit",
        "td-gen-cancel",
        "td-gen-alert",
        "td-gen-status",
        "td-poll",
        "td-loop-toggle",
        "td-loop-collapse",
    ):
        assert cid in ids, f"layout 缺少组件 {cid}"
    intervals = _components(comp, dcc.Interval)
    assert any(getattr(i, "id", None) == "td-poll" and i.interval == 2500 for i in intervals)
    assert "生成次日订单" in " ".join(_texts(comp))
    # 持仓输入为可编辑表格（可删行）
    tbl = next(
        c for c in _components(comp, dash_table.DataTable) if c.id == "td-gen-holdings-table"
    )
    assert tbl.editable is True and tbl.row_deletable is True
    assert [c["name"] for c in tbl.columns] == ["股票代码", "股数"]


def test_layout_headers_semantic_and_loop_collapsed(tmp_artifacts):
    """卡片标题去字母编号（语义名）；执行闭环默认折叠；页面头带流程面包屑。"""
    comp = td.layout()
    headers = _components(comp, dbc.CardHeader)
    titles = [h.children for h in headers if isinstance(h.children, str)]
    # 语义名标题齐全，且不再有 "A ·" 式字母前缀
    for t in ("今日操作", "风控五层状态", "账户快照", "持仓明细", "执行闭环"):
        assert any(t in x for x in titles), t
    assert not any(len(x) >= 3 and x[1:3] == " ·" for x in titles)
    # 执行闭环：Collapse 默认收起，按钮文案含引导语
    col = next(c for c in _components(comp, dbc.Collapse) if c.id == "td-loop-collapse")
    assert col.is_open is False
    btn = next(b for b in _components(comp, dbc.Button) if b.id == "td-loop-toggle")
    assert btn.children == "第一次用？展开 5 步操作说明"
    # 页面头：标题 + 面包屑当前步「交易执行」
    cur = [
        n
        for n in _components(comp, html.Span)
        if "flow-step-cur" in (getattr(n, "className", "") or "").split()
    ]
    assert cur and cur[0].children == "交易执行"
    assert "交易台 Trading Desk" in " ".join(_texts(comp))


def test_layout_entry_prefills_from_account(tmp_artifacts):
    out = tmp_artifacts
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "shares": [1000],
            "entry_date": ["2026-10-01"],
            "ref_close": [10.0],
        }
    ).to_csv(out / "holdings.csv", index=False)
    pd.DataFrame({"date": ["2026-10-22"], "cash": [100000.0]}).to_csv(
        out / "account.csv", index=False
    )
    # 收盘价 map 走真实 _close_maps：写两个 daily 缓存（避免 monkeypatch 掉
    # lru_cache 包装器导致 fixture 清理时 cache_clear 失效）
    tdir = tmp_artifacts.parent / "tushare"
    tdir.mkdir()
    pd.DataFrame({"symbol": ["000001.SZ"], "close": [10.0]}).to_parquet(
        tdir / "daily_20261021.parquet", index=False
    )
    pd.DataFrame({"symbol": ["000001.SZ"], "close": [11.0]}).to_parquet(
        tdir / "daily_20261022.parquet", index=False
    )
    dataio.refresh_caches()
    td._close_maps.cache_clear()
    comp = td.layout()
    # 持仓预填在弹窗打开时按口径填充；layout 里资金预填 = 现金 + 持仓市值 = 100000 + 11000
    rows, note = td._prefill_holding_rows()
    assert rows == [{"symbol": "000001.SZ", "shares": 1000}]
    assert "当前持仓" in note
    cap = next(c for c in _components(comp, dbc.Input) if c.id == "td-gen-capital")
    assert cap.value == pytest.approx(111000.0)


def test_build_nextday_params_defaults():
    params, label, err = td._build_nextday_params(2_000_000, "600519.SH:200", "  今晚出单 ")
    assert err is None
    assert label == "今晚出单"
    assert params["capital"] == pytest.approx(2_000_000.0)
    assert params["initial_holdings"] == "600519.SH:200"
    # 其余参数 = registry 默认值
    defaults = registry.default_params("nextday")
    for k, v in defaults.items():
        if k not in ("capital", "initial_holdings"):
            assert params[k] == v, k
    assert set(params) == set(defaults)


def test_build_nextday_params_empty_holdings_and_label():
    params, label, err = td._build_nextday_params(100000, "", None)
    assert err is None and label == ""
    assert params["initial_holdings"] == ""
    # 注释行与空行合法
    params, _, err = td._build_nextday_params(100000, "# 备注\n\n000001.SZ:1000\n", "x")
    assert err is None


def test_build_nextday_params_validation():
    # 缺分隔符
    params, _, err = td._build_nextday_params(100000, "600519.SH 200", "")
    assert params is None and "第1行" in err
    # 股数非数字
    _, _, err = td._build_nextday_params(100000, "600519.SH:abc", "")
    assert err and "600519.SH" in err
    # 不足一手会被解析器丢弃 → 视为格式错误
    _, _, err = td._build_nextday_params(100000, "600519.SH:50", "")
    assert err
    # 资金非法 / 低于下限
    _, _, err = td._build_nextday_params("abc", "", "")
    assert err and "资金" in err
    _, _, err = td._build_nextday_params(100, "", "")
    assert err


def test_default_capital():
    # 有快照用净值口径，无快照回落 registry 默认
    assert td._default_capital({"nav": 111000.0}) == pytest.approx(111000.0)
    assert td._default_capital(None) == pytest.approx(
        float(registry.default_params("nextday")["capital"])
    )


# ---------------------------------------------------------------------------
# 生成弹窗：预填持仓（projected_holdings）与表格序列化校验（rows_to_holdings_text）
# ---------------------------------------------------------------------------
def _holds_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "shares": [1000, 500],
            "entry_date": ["2026-10-01", "2026-09-15"],
            "ref_close": [10.0, 20.0],
        }
    )


def _gen_orders_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["000001.SZ", "buy", 200, False],  # 加仓
            ["000002.SZ", "sell", 500, False],  # 清仓 → 删行
            ["600000.SH", "buy", 300, False],  # 新建
            ["600001.SH", "buy", 100, True],  # blocked 跳过
        ],
        columns=["symbol", "side", "shares", "blocked"],
    )


def test_projected_holdings():
    rows = td.projected_holdings(_gen_orders_df(), _holds_df())
    assert rows == [
        {"symbol": "000001.SZ", "shares": 1200},  # 加仓，保序
        {"symbol": "600000.SH", "shares": 300},  # 新建
        # 000002.SZ 清仓删除；600001.SH 拦截不进
    ]
    # 部分减仓
    o = _gen_orders_df()
    o.loc[1, "shares"] = 200
    rows = td.projected_holdings(o, _holds_df())
    assert {r["symbol"]: r["shares"] for r in rows}["000002.SZ"] == 300
    # 无订单 → 当前持仓原样；空持仓 + 订单 → 只含买单
    assert td.projected_holdings(pd.DataFrame(), _holds_df()) == [
        {"symbol": "000001.SZ", "shares": 1000},
        {"symbol": "000002.SZ", "shares": 500},
    ]
    assert td.projected_holdings(_gen_orders_df(), pd.DataFrame()) == [
        {"symbol": "000001.SZ", "shares": 200},
        {"symbol": "600000.SH", "shares": 300},
    ]


def test_rows_to_holdings_text_ok():
    text, err = td.rows_to_holdings_text(
        [
            {"symbol": "000001.sz", "shares": 1000},  # 小写后缀归一
            {"symbol": "600519.SH", "shares": 200},
            {"symbol": "", "shares": None},  # 空白行跳过
        ]
    )
    assert err is None
    assert text == "000001.SZ:1000\n600519.SH:200"
    # 重复代码合并
    text, err = td.rows_to_holdings_text(
        [{"symbol": "000001.SZ", "shares": 300}, {"symbol": "000001.SZ", "shares": 200}]
    )
    assert err is None and text == "000001.SZ:500"
    # 空表 → 空文本（空仓）
    assert td.rows_to_holdings_text([]) == ("", None)
    assert td.rows_to_holdings_text(None) == ("", None)


def test_rows_to_holdings_text_validation():
    # 非法代码
    _, err = td.rows_to_holdings_text([{"symbol": "600519", "shares": 200}])
    assert err and "第1行" in err and "代码" in err
    _, err = td.rows_to_holdings_text([{"symbol": "600519.XX", "shares": 200}])
    assert err
    # 股数非数字 / 0 / 负
    _, err = td.rows_to_holdings_text([{"symbol": "600519.SH", "shares": "abc"}])
    assert err and "数字" in err
    _, err = td.rows_to_holdings_text([{"symbol": "600519.SH", "shares": 0}])
    assert err and "正数" in err
    _, err = td.rows_to_holdings_text([{"symbol": "600519.SH", "shares": -100}])
    assert err
    # 有代码但股数留空 → 报错（不算空白行）
    _, err = td.rows_to_holdings_text([{"symbol": "600519.SH", "shares": None}])
    assert err
    # 序列化结果过下游 parse_initial_holdings 链路（整手向下取整）
    text, err = td.rows_to_holdings_text([{"symbol": "600519.SH", "shares": 250}])
    assert err is None and parse_initial_holdings(text) == {"600519.SH": 200}


def test_prefill_holding_rows_committed_not_double_applied(tmp_artifacts, monkeypatch):
    """已 commit（holdings mtime ≥ 订单 mtime）→ 预填 = holdings 本身，不重复应用订单。"""
    import os

    out = tmp_artifacts
    monkeypatch.setattr(td, "_today_str", lambda: "20261022")
    _gen_orders_df().assign(limit_price=1.0, ref_close=1.0).to_csv(
        out / "orders_20261022.csv", index=False
    )
    _holds_df().to_csv(out / "holdings.csv", index=False)
    dataio.refresh_caches()
    # 订单晚于 holdings → 应用订单
    os.utime(out / "orders_20261022.csv", (1_800_000_100, 1_800_000_100))
    os.utime(out / "holdings.csv", (1_800_000_000, 1_800_000_000))
    rows, note = td._prefill_holding_rows()
    assert {r["symbol"]: r["shares"] for r in rows} == {"000001.SZ": 1200, "600000.SH": 300}
    assert "全部成交" in note
    # holdings 晚于订单（已 commit）→ 不重复应用
    os.utime(out / "holdings.csv", (1_800_000_200, 1_800_000_200))
    rows, note = td._prefill_holding_rows()
    assert {r["symbol"]: r["shares"] for r in rows} == {"000001.SZ": 1000, "000002.SZ": 500}
    assert "不重复应用" in note


def test_prefill_holding_rows_empty_state(tmp_artifacts, monkeypatch):
    monkeypatch.setattr(td, "_today_str", lambda: "20261022")
    dataio.refresh_caches()
    rows, note = td._prefill_holding_rows()
    assert rows == [] and "空表起步" in note


def test_session_overall_pct():
    meta = {
        "steps": [{"status": "done"}, {"status": "running"}, {"status": "pending"}],
        "current_step": 1,
        "current_progress": {"pct": 0.5},
    }
    assert td._session_overall_pct(meta) == pytest.approx(50.0)
    # 缺 current_progress → 只算已完成步
    meta2 = {"steps": meta["steps"], "current_step": 1}
    assert td._session_overall_pct(meta2) == pytest.approx(100 / 3)
    # 无步骤 / 非法 pct 容错为 0
    assert td._session_overall_pct({}) == 0.0
    meta3 = {"steps": meta["steps"], "current_step": 1, "current_progress": {"pct": "bad"}}
    assert td._session_overall_pct(meta3) == pytest.approx(100 / 3)


def test_render_session_status_queued_and_running():
    queued = {
        "id": "nextday_20261023_180000",
        "label": "今晚出单",
        "status": "queued",
        "steps": [{"status": "pending", "title": "数据导入"}],
    }
    comp = td._render_session_status(queued)
    txt = " ".join(_texts(comp))
    assert "排队中" in txt and "nextday_20261023_180000" in txt
    link = _components(comp, dcc.Link)[0]
    assert link.href == "/runs/nextday_20261023_180000"

    running = {
        "id": "nextday_20261023_180000",
        "label": "今晚出单",
        "status": "running",
        "current_step": 1,
        "steps": [
            {"status": "done", "title": "数据导入"},
            {"status": "running", "title": "训练 + 预测"},
            {"status": "pending", "title": "生成次日订单"},
        ],
        "current_progress": {
            "phase": "预测最新日",
            "message": "5000 只",
            "pct": 0.5,
            "eta_sec": 42,
        },
    }
    comp = td._render_session_status(running)
    prog = _components(comp, dbc.Progress)[0]
    assert prog.value == pytest.approx(50.0)
    txt = " ".join(_texts(comp))
    assert "预测最新日" in txt and "5000 只" in txt and "预计余 42s" in txt
    assert "查看详情" in txt


def test_render_idle_status(tmp_artifacts, monkeypatch):
    monkeypatch.setattr(td, "_today_str", lambda: "20261022")
    _orders_df().to_csv(tmp_artifacts / "orders_20261022.csv", index=False)
    done = {
        "id": "nextday_20261021_180000",
        "status": "done",
        "finished": 1780000000.0,
    }
    comp = td._render_idle_status(done, cli_today=td._cli_orders_today())
    txt = " ".join(_texts(comp))
    assert "今日已有订单文件" in txt  # 防双写提示
    assert "最近出单会话" in txt and "完成" in txt
    # 无会话且无 CLI 产物 → 空态
    monkeypatch.setattr(td, "_today_str", lambda: "20261023")
    assert not _components(td._render_idle_status(None, cli_today=False), dbc.Alert)


# ---------------------------------------------------------------------------
# 订单「分析▾」解释面板：_order_explain 决策链翻译 + _explain_context 数据组装
# ---------------------------------------------------------------------------
def _mk_order_row(**kw):
    base = dict(
        symbol="002611.SZ",
        name="测试股",
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
    return SimpleNamespace(**base)


def _full_ctx():
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


def _sec_map(secs):
    return {s["title"]: " ".join(s["lines"]) for s in secs}


def test_order_explain_sell_hold_band_full():
    """真实案例复刻：002611.SZ 卖 1200 股，跌出持有带 + 全数据小节齐全。"""
    secs = _sec_map(td._order_explain(_mk_order_row(), _full_ctx()))
    trig = secs["① 触发规则：跌出ML持有带（排名衰减卖出）"]
    assert "第 3,599/5,476 名（前 65.7%）" in trig and "前 15%" in trig
    timing = secs["② 择时检查"]
    assert "43.4" in timing and "中性" in timing and "0.44" in timing and "带内" in timing
    assert "非择时卖出" in timing
    model = secs["③ 模型明细（各子模型分别打分）"]
    assert "AI 树模型 -0.0085（偏弱）" in model and "因子加权模型 +0.7271（偏多）" in model
    assert "LightGBM" in model  # 子模型一句话解释
    assert "权重 AI 树模型 50% · 因子加权模型 50% → 融合分 0.458（前一日 0.463，略降）" in model
    assert "腿" not in model and "gbm_l2" not in model and "ic_linear" not in model
    param = secs["④ 订单参数"]
    assert "卖出 1,200 股（全部持仓）" in param
    assert "限价 14.889" in param and "布林上轨" in param and "参考收盘 13.39" in param
    pos = secs["⑤ 持仓信息"]
    assert "建仓日 2026-10-10" in pos and "成本 12.00" in pos and "+11.6%" in pos


def test_order_explain_sell_timing_reasons():
    """RSI 超买 / 触布林上轨（择时卖出）：② 不再写「非择时卖出」。"""
    secs = _sec_map(td._order_explain(_mk_order_row(reason="RSI超买(85)", rsi=85.0), {}))
    trig = secs["① 触发规则：RSI 超买（择时卖出）"]
    assert "85" in trig and "超买线 80" in trig
    assert "非择时卖出" not in secs["② 择时检查"]
    secs = _sec_map(td._order_explain(_mk_order_row(reason="触布林上轨", bb_pos=0.97), {}))
    assert "布林上轨" in secs["① 触发规则：触布林上轨（择时卖出）"]
    assert "非择时卖出" not in secs["② 择时检查"]


def test_order_explain_rebalance_reasons():
    """满仓再平衡口径（use_bb 关闭路径）的四种 reason。"""
    secs = _sec_map(td._order_explain(_mk_order_row(reason="再平衡:清仓出池"), {}))
    assert "目标池" in secs["① 触发规则：再平衡清仓（跌出目标池）"]
    assert "非择时卖出" in secs["② 择时检查"]
    secs = _sec_map(td._order_explain(_mk_order_row(reason="再平衡:减仓"), {}))
    assert "减仓" in secs["① 触发规则：再平衡减仓（超配回落）"]
    buy = _mk_order_row(
        side="buy", shares=200, limit_price=5.0, ref_close=5.0, reason="再平衡:建仓", rank_pct=0.05
    )
    secs = _sec_map(td._order_explain(buy, {"n_stocks": 5476}))
    trig = secs["① 触发规则：再平衡建仓（进入目标池）"]
    assert "第 274/5,476 名" in trig
    assert "未启用指标择时门控" in secs["② 择时检查"]
    # 市价单（limit==ref → next_open 占位）
    assert "市价单" in secs["④ 订单参数"]
    secs = _sec_map(td._order_explain(_mk_order_row(side="buy", reason="再平衡:加仓"), {}))
    assert "加仓" in secs["① 触发规则：再平衡加仓（低配补足）"]


def test_order_explain_buy_bb_pullback():
    buy = _mk_order_row(
        side="buy",
        shares=300,
        limit_price=8.5,
        ref_close=8.9,
        reason="布林下轨回踩",
        rank_pct=0.03,
        bb_pos=0.05,
        rsi=35.0,
    )
    secs = _sec_map(td._order_explain(buy, {"n_stocks": 5476}))
    trig = secs["① 触发规则：布林下轨回踩（排名+择时）"]
    assert "第 164/5,476 名" in trig and "下轨" in trig
    assert "择时门槛通过" in secs["② 择时检查"]
    assert "贴近下轨" in secs["② 择时检查"]


def test_order_explain_blocked_orders():
    """blocked 单：置顶拦截说明 + 不执行语义；触发规则剥离拦截前缀后照常翻译。"""
    buy = _mk_order_row(side="buy", reason="涨停拦截:布林下轨回踩", blocked="True")
    secs = td._order_explain(buy, {})
    assert secs[0]["title"] == "⛔ 拦截说明"
    assert "不执行" in secs[0]["lines"][0] and "涨停" in secs[0]["lines"][0]
    assert secs[1]["title"].startswith("① 触发规则：布林下轨回踩")
    sell = _mk_order_row(reason="跌停拦截:跌出ML持有带", blocked=True)
    secs = td._order_explain(sell, {})
    assert secs[0]["title"] == "⛔ 拦截说明" and "跌停" in secs[0]["lines"][0]
    assert "跌出ML持有带" in secs[1]["title"]


def test_order_explain_clearout_and_degradation():
    """跌出横截面清退单（无排名/指标数据）+ 空 ctx 降级：只剩 ①④，不抛异常。"""
    row = _mk_order_row(
        reason="跌出横截面:清退",
        rank_pct=None,
        bb_pos=None,
        rsi=None,
        limit_price=13.39,
        ref_close=13.39,
    )
    secs = _sec_map(td._order_explain(row, {}))
    assert "停牌" in secs["① 触发规则：跌出横截面（清退）"]
    assert "② 择时检查" not in secs and "③ 模型明细（各子模型分别打分）" not in secs
    assert "市价单" in secs["④ 订单参数"]
    # 空 ctx：pred/holdings/weights/n_stocks 全缺 → 小节自动省略
    secs = _sec_map(td._order_explain(_mk_order_row(), {}))
    titles = list(secs)
    assert "③ 模型明细（各子模型分别打分）" not in titles and "⑤ 持仓信息" not in titles
    assert "全市场前 65.7%" in secs["① 触发规则：跌出ML持有带（排名衰减卖出）"]  # 无 N 回落百分比
    assert "（全部持仓）" not in secs["④ 订单参数"]


def test_explain_context_assembles(tmp_artifacts):
    """_explain_context：pred_hist 近两日 + 会话步报告权重/参数 + 持仓，缺源降级。"""
    out = tmp_artifacts
    pd.DataFrame(
        [
            ["2026-10-26", "002611.SZ", 0.4631, -0.0074, 0.6323],
            ["2026-10-27", "002611.SZ", 0.4584, -0.0085, 0.7271],
            ["2026-10-27", "600000.SH", 0.9, 0.5, 0.6],
        ],
        columns=["date", "symbol", "score", "pred_gbm_l2", "pred_ic_linear"],
    ).to_parquet(out / "live_risk_state_pred_hist.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["002611.SZ"],
            "shares": [1200],
            "entry_date": ["2026-10-10"],
            "ref_close": [12.0],
        }
    ).to_csv(out / "holdings.csv", index=False)
    sdir = out / "sessions" / "nextday_20261028_134325"
    (sdir / "steps").mkdir(parents=True)
    (sdir / "session.json").write_text(
        json.dumps(
            {
                "id": sdir.name,
                "status": "done",
                "params": {"topk": 0.2, "order_price_mode": "limit_bb"},
            }
        ),
        encoding="utf-8",
    )
    (sdir / "steps" / "01_train_predict.json").write_text(
        json.dumps({"summary": "已预测 5,476 只 · 权重 gbm_l2=0.50 · ic_linear=0.50"}),
        encoding="utf-8",
    )
    dataio.refresh_caches()

    orders = pd.DataFrame([["002611.SZ", "sell"]], columns=["symbol", "side"])
    meta = {"path": str(sdir / "artifacts" / "orders.csv")}
    ctx = td._explain_context(orders, meta)
    # 当日股票数优先 pred_hist 行数（2），不取步报告的 5,476
    assert ctx["n_stocks"] == 2
    assert ctx["weights"] == {"gbm_l2": 0.5, "ic_linear": 0.5}
    assert ctx["params"]["topk"] == 0.2
    p = ctx["pred"]["002611.SZ"]
    assert p["score"] == pytest.approx(0.4584) and p["prev_score"] == pytest.approx(0.4631)
    assert p["legs"]["gbm_l2"] == pytest.approx(-0.0085)
    assert "600000.SH" not in ctx["pred"]  # 只组装订单内股票
    assert ctx["holdings"]["002611.SZ"]["entry_date"] == "2026-10-10"

    # CLI 来源（无会话目录）→ 权重/参数空，pred_hist/持仓仍可用
    ctx2 = td._explain_context(orders, {"path": str(out / "orders_20261027.csv")})
    assert ctx2["weights"] is None and ctx2["params"] == {}
    assert ctx2["pred"]["002611.SZ"]["score"] == pytest.approx(0.4584)
    # 空订单 → 只保留会话层信息；meta 全空 → 全空态不抛异常
    ctx3 = td._explain_context(pd.DataFrame(), meta)
    assert ctx3["pred"] == {} and ctx3["weights"] == {"gbm_l2": 0.5, "ic_linear": 0.5}
    assert td._explain_context(pd.DataFrame(), {})["weights"] is None

    # 新版管线：步报告带结构化 weights/n_predicted + 人话摘要 → 优先结构化字段，
    # 人话摘要（无 key=value）不会让正则兜底误解析
    (sdir / "steps" / "01_train_predict.json").write_text(
        json.dumps(
            {
                "summary": "已预测 5,476 只 · 权重 AI 树模型 60% · 因子加权模型 40%",
                "weights": {"gbm_l2": 0.6, "ic_linear": 0.4},
                "n_predicted": 5476,
            }
        ),
        encoding="utf-8",
    )
    dataio.refresh_caches()
    ctx4 = td._explain_context(orders, meta)
    assert ctx4["weights"] == {"gbm_l2": 0.6, "ic_linear": 0.4}
    # n_stocks 仍优先 pred_hist 当日行数（2），结构化 n_predicted 只是兜底
    assert ctx4["n_stocks"] == 2
    # 无 pred_hist 时取结构化 n_predicted
    (out / "live_risk_state_pred_hist.parquet").unlink()
    dataio.refresh_caches()
    ctx5 = td._explain_context(pd.DataFrame(), meta)
    assert ctx5["n_stocks"] == 5476 and ctx5["weights"] == {"gbm_l2": 0.6, "ic_linear": 0.4}


def test_layout_order_rows_have_explain_panels(tmp_artifacts, monkeypatch):
    """layout 渲染：每笔订单行末尾带「分析 ▸」Details 面板（4 笔订单 → 4 个）。"""
    monkeypatch.setattr(td, "_today_str", lambda: "20261022")
    _orders_df().to_csv(tmp_artifacts / "orders_20261022.csv", index=False)
    dataio.refresh_caches()
    comp = td.layout()
    assert _texts(comp).count("分析 ▸") == 4
    panels = _components(comp, html.Details)
    assert len(panels) >= 4
