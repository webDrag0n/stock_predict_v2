"""前端重构 P0 冒烟测试：

· 每条新路由对应的页面 layout() 在空产物环境下返回 Dash 组件、不抛异常；
· app 外壳可导入（新路由表 / 状态条容错渲染 / 旧路由重定向）；
· dataio 新 loader 在文件缺失时返回空态（None / 空 DataFrame），不抛异常。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest
from dash.development.base_component import Component

from webapp import dataio
from webapp.pages import (
    factor_workbench,
    market_explorer,
    model_lab,
    session_detail,
    trading_desk,
)

# 新路由表（docs/frontend_redesign.html 第 2 节）：路由 → 页面模块
PAGE_MODULES = {
    "/": trading_desk,
    "/models": model_lab,
    "/factors": factor_workbench,
    "/market": market_explorer,
    "/runs/<id>": session_detail,
}


@pytest.mark.parametrize("route", sorted(PAGE_MODULES))
def test_page_layout_renders(route, monkeypatch):
    # 交易日判定走真实 tushare 日历/网络，测试环境统一打桩为「未知」
    stub = lambda day: None  # noqa: E731
    stub.cache_clear = lambda: None
    monkeypatch.setattr(dataio, "is_trading_day", stub)
    mod = PAGE_MODULES[route]
    lay = mod.layout
    comp = lay() if callable(lay) else lay
    assert isinstance(comp, Component)
    assert hasattr(mod, "register")


def test_app_shell_routes_and_status_bar(monkeypatch):
    # 跳过启动对账/预热（DEBUG=1 且无 WERKZEUG_RUN_MAIN → reloader 父进程分支），
    # 避免测试进程对真实 outputs/sessions 产生副作用
    monkeypatch.setenv("DEBUG", "1")
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    from webapp import app as app_module

    assert set(app_module.PAGES) == {"/", "/models", "/factors", "/market"}

    # 旧路由 /session/<id> → dcc.Location 重定向到 /runs/<id>
    redirect = app_module._route("/session/backtest_20240101_120000")
    assert isinstance(redirect, Component)
    assert redirect.href == "/runs/backtest_20240101_120000"

    # /runs/<id> → 会话详情页；未知路由 → 交易台首页
    assert isinstance(app_module._route("/runs/x"), Component)
    assert isinstance(app_module._route("/nonexistent"), Component)

    # 状态条在无 champion / 无 daily 缓存环境下也要能渲染
    chips = app_module._status_bar()
    assert chips and all(isinstance(c, Component) for c in chips)

    # 导航高亮：会话详情 /runs/、/session/ 归模型实验室；/ 仅精确匹配
    assert app_module._nav_active("/runs/x", "/models")
    assert app_module._nav_active("/session/x", "/models")
    assert not app_module._nav_active("/runs/x", "/")
    assert not app_module._nav_active("/runs/x", "/factors")
    # 侧边栏副行已中文化
    subs = [sub for _, _, sub in app_module.NAV_ITEMS]
    assert subs == ["每日出单与执行", "训练·评估·回测", "因子研发与体检", "个股与全市场行情"]


@pytest.fixture
def empty_artifact_dirs(tmp_path, monkeypatch):
    """把 dataio 的产物目录指向空 tmp 目录，并清掉全部 lru_cache。"""
    monkeypatch.setattr(dataio, "OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setattr(dataio, "TRACKING", str(tmp_path / "outputs" / "tracking"))
    monkeypatch.setattr(dataio, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(dataio, "FACTOR_LIBRARY", str(tmp_path / "factor_library"))
    dataio.refresh_caches()
    yield tmp_path
    dataio.refresh_caches()


def test_dataio_new_loaders_empty_state(empty_artifact_dirs):
    assert dataio.latest_orders().empty
    df, meta = dataio.latest_orders_with_meta()
    assert df.empty and meta == {}
    assert dataio.holdings().empty
    assert dataio.account().empty
    assert dataio.live_risk_state() is None
    assert dataio.tracking_report() is None
    assert dataio.live_daily().empty
    assert dataio.champion_manifest() is None
    assert dataio.staging_manifests() == ()
    lib = dataio.factor_library()
    assert lib.empty and list(lib.columns) == list(dataio._FACTOR_COLS)
    assert dataio.factor_eval_table().empty
    assert dataio.factor_ic_decay().empty
    assert dataio.factor_corr().empty


def test_factor_library_reads_yaml(empty_artifact_dirs):
    """非空态：写一份最小因子 YAML，确认 factor_library() 解析出 schema 字段。"""
    lib_dir = empty_artifact_dirs / "factor_library"
    lib_dir.mkdir(parents=True)
    (lib_dir / "custom.yaml").write_text(
        "factors:\n"
        "  - id: f1\n"
        "    name: 测试因子\n"
        "    category: momentum\n"
        '    formula: "adj_close / ts_delay(adj_close, 5) - 1"\n'
        "    tags: [momentum]\n"
        "    expected_sign: 1\n",
        encoding="utf-8",
    )
    df = dataio.factor_library()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["id"] == "f1"
    assert row["name"] == "测试因子"
    assert row["category"] == "momentum"
    assert bool(row["enabled"]) is True
    assert row["tags"] == ["momentum"]
    assert row["expected_sign"] == 1


def test_champion_and_staging_manifests(empty_artifact_dirs):
    """非空态：构造 champion/staging 目录，确认 manifest loader 读取字段。"""
    import json

    models = empty_artifact_dirs / "models"
    (models / "champion").mkdir(parents=True)
    (models / "champion" / "manifest.json").write_text(
        json.dumps({"model_id": "20261001_abcdef123456", "created_at": "2026-10-01T18:00:00"}),
        encoding="utf-8",
    )
    stg = models / "staging" / "cand_a"
    stg.mkdir(parents=True)
    (stg / "manifest.json").write_text(json.dumps({"model_id": "cand_a"}), encoding="utf-8")
    (stg / "evaluation.json").write_text(json.dumps({"sharpe": 1.5}), encoding="utf-8")

    man = dataio.champion_manifest()
    assert man["model_id"] == "20261001_abcdef123456"
    stags = dataio.staging_manifests()
    assert len(stags) == 1
    assert stags[0]["id"] == "cand_a"
    assert stags[0]["manifest"]["model_id"] == "cand_a"
    assert stags[0]["evaluation"]["sharpe"] == 1.5


def test_artifact_csv_loaders_mtime_invalidation(empty_artifact_dirs):
    """mtime 感知缓存：文件改写（mtime 变化）后无需 refresh_caches() 即读到新内容。"""
    import os

    out_dir = empty_artifact_dirs / "outputs"
    out_dir.mkdir(parents=True)
    path = out_dir / "factor_eval.csv"
    path.write_text("factor,ic\nf1,0.01\n", encoding="utf-8")
    os.utime(path, (1_700_000_000, 1_700_000_000))

    df1 = dataio.factor_eval_table()
    assert list(df1["factor"]) == ["f1"]

    # 模拟 factoreval 会话落盘新 csv：改写内容并推进 mtime，不调 refresh_caches()
    path.write_text("factor,ic\nf1,0.01\nf2,0.02\n", encoding="utf-8")
    os.utime(path, (1_700_000_100, 1_700_000_100))

    df2 = dataio.factor_eval_table()
    assert list(df2["factor"]) == ["f1", "f2"]


def test_latest_orders_mtime_invalidation(empty_artifact_dirs):
    """交易台轮询语义：nextday 落盘新 orders_*.csv 后 latest_orders 立即可见。"""
    import os

    out_dir = empty_artifact_dirs / "outputs"
    out_dir.mkdir(parents=True)
    assert dataio.latest_orders().empty

    p1 = out_dir / "orders_20261024.csv"
    p1.write_text("symbol,side\n000001.SZ,buy\n", encoding="utf-8")
    os.utime(p1, (1_700_000_000, 1_700_000_000))
    df1 = dataio.latest_orders()
    assert list(df1["symbol"]) == ["000001.SZ"]

    # 新一轮 nextday 落盘次日订单（新文件名），不调 refresh_caches()
    p2 = out_dir / "orders_20261025.csv"
    p2.write_text("symbol,side\n000002.SZ,sell\n", encoding="utf-8")
    os.utime(p2, (1_700_000_100, 1_700_000_100))
    df2 = dataio.latest_orders()
    assert list(df2["symbol"]) == ["000002.SZ"]


# ---------------------------------------------------------------------------
# latest_orders_with_meta：CLI 产物 + nextday 会话产物的统一来源
# ---------------------------------------------------------------------------
def _session_artifact(out_dir, sid, symbol, asof, finished, status="done", mtime=None):
    """造一个 nextday 会话目录：session.json + artifacts/orders.csv（asof=None → 缺 asof 列）。"""
    sdir = out_dir / "sessions" / sid
    (sdir / "artifacts").mkdir(parents=True)
    opath = sdir / "artifacts" / "orders.csv"
    if asof is None:
        opath.write_text(f"symbol,side\n{symbol},buy\n", encoding="utf-8")
    else:
        opath.write_text(f"symbol,side,asof_date\n{symbol},buy,{asof}\n", encoding="utf-8")
    if mtime is not None:
        os.utime(opath, (mtime, mtime))
    (sdir / "session.json").write_text(
        json.dumps({"id": sid, "mode": "nextday", "status": status, "finished": finished}),
        encoding="utf-8",
    )
    return opath


def test_latest_orders_unified_source_session_wins_by_date(empty_artifact_dirs):
    """统一来源：CLI 与会话产物取订单日期最新（会话 asof 20261026 > CLI 文件名 20261020）。"""
    out_dir = empty_artifact_dirs / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "orders_20261020.csv").write_text(
        "symbol,side,asof_date\n000001.SZ,buy,2026-10-20\n", encoding="utf-8"
    )
    _session_artifact(out_dir, "nextday_20261027_135952", "600000.SH", "2026-10-26", 1793000000.0)
    df, meta = dataio.latest_orders_with_meta()
    assert list(df["symbol"]) == ["600000.SH"]
    assert meta["date"] == "20261026"
    assert meta["source"] == "会话 nextday_20261027_135952"
    assert meta["path"].endswith("artifacts/orders.csv")
    # latest_orders 与 with_meta 同源
    assert list(dataio.latest_orders()["symbol"]) == ["600000.SH"]


def test_latest_orders_unified_source_same_date_mtime_tiebreak(empty_artifact_dirs):
    """同订单日期：mtime 新的来源胜出（双向），且 mtime 变化无需 refresh_caches。"""
    out_dir = empty_artifact_dirs / "outputs"
    out_dir.mkdir(parents=True)
    cli = out_dir / "orders_20261026.csv"
    cli.write_text("symbol,side,asof_date\n000001.SZ,buy,2026-10-26\n", encoding="utf-8")
    os.utime(cli, (1_800_000_000, 1_800_000_000))
    _session_artifact(
        out_dir,
        "nextday_20261026_220000",
        "600000.SH",
        "2026-10-26",
        1793000000.0,
        mtime=1_800_000_100,
    )
    df, meta = dataio.latest_orders_with_meta()
    assert meta["source"] == "会话 nextday_20261026_220000"
    assert list(df["symbol"]) == ["600000.SH"]
    # 反向：CLI 文件 mtime 更新 → CLI 胜出
    os.utime(cli, (1_800_000_200, 1_800_000_200))
    df2, meta2 = dataio.latest_orders_with_meta()
    assert meta2["source"] == "cli"
    assert list(df2["symbol"]) == ["000001.SZ"]


def test_latest_orders_session_asof_fallback_and_skip_not_done(empty_artifact_dirs):
    """会话订单缺 asof_date 列 → 订单日期回落会话完成时间；非 done 会话不算来源。"""
    out_dir = empty_artifact_dirs / "outputs"
    out_dir.mkdir(parents=True)
    fin = 1793000000.0
    _session_artifact(out_dir, "nextday_20261027_090000", "600001.SH", None, fin, status="running")
    df, meta = dataio.latest_orders_with_meta()
    assert df.empty and meta == {}
    _session_artifact(out_dir, "nextday_20261027_135952", "600002.SH", None, fin)
    df, meta = dataio.latest_orders_with_meta()
    assert list(df["symbol"]) == ["600002.SH"]
    assert meta["date"] == datetime.fromtimestamp(fin).strftime("%Y%m%d")
    assert meta["source"] == "会话 nextday_20261027_135952"


def test_pred_hist_loader(empty_artifact_dirs):
    """pred_hist loader：缺失 → 空表；落盘/改写后 mtime 感知自动重读（无需 refresh_caches）。"""
    import pandas as pd

    out_dir = empty_artifact_dirs / "outputs"
    out_dir.mkdir(parents=True)
    assert dataio.pred_hist().empty
    p = out_dir / "live_risk_state_pred_hist.parquet"
    pd.DataFrame({"date": ["2026-10-27"], "symbol": ["000001.SZ"], "score": [0.5]}).to_parquet(
        p, index=False
    )
    os.utime(p, (1_800_000_000, 1_800_000_000))
    assert list(dataio.pred_hist()["symbol"]) == ["000001.SZ"]
    pd.DataFrame(
        {"date": ["2026-10-27"] * 2, "symbol": ["000001.SZ", "600000.SH"], "score": [0.5, 0.6]}
    ).to_parquet(p, index=False)
    os.utime(p, (1_800_000_100, 1_800_000_100))
    assert len(dataio.pred_hist()) == 2
