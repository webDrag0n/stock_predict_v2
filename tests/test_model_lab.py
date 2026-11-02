"""模型实验室（webapp/pages/model_lab.py）单元测试。

构造临时 outputs/sessions 与 models 目录（tmp_path + monkeypatch），验证：
  · 页面头（page_header 面包屑）、引导条 _guidance 三态与静态组件 id；
  · champion 卡片空态（models/ 为空是常态）与年龄黄灯；
  · 验证历史时间线的数据组装（会话 Sharpe 提取 + canonical 点 + 模式过滤）；
  · staging evaluation.json 的对比结论解析（Sharpe 差/回撤差/晋升门槛）；
  · canonical meta.json 缺 sharpe_pos_ratio/predic_pos_ratio 时从
    window_metrics.parquet 兜底现算；
  · 创建成功 Alert 含 /runs/<sid> 链接；Runs 表默认截断最近 20 条 + 显示全部；
  · 真实目录多为空：所有渲染函数的空态路径不抛异常。
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest
import dash_bootstrap_components as dbc
from dash import dcc
from dash.development.base_component import Component

from pipeline import registry, scheduler
from webapp import dataio
from webapp.pages import model_lab


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """把 dataio / scheduler 的目录指向空 tmp 目录，并清掉 dataio 的 lru_cache。"""
    models = tmp_path / "models"
    outputs = tmp_path / "outputs"
    sessions = outputs / "sessions"
    (outputs / "walkforward").mkdir(parents=True)
    sessions.mkdir(parents=True)
    monkeypatch.setattr(dataio, "MODELS", str(models))
    monkeypatch.setattr(dataio, "OUTPUTS", str(outputs))
    monkeypatch.setattr(scheduler, "SESSIONS_DIR", str(sessions))
    dataio.refresh_caches()
    yield tmp_path
    dataio.refresh_caches()


def _flatten_text(comp) -> str:
    """递归收集组件树里的全部字符串，用于空态/文案断言。"""
    out = []

    def _walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, Component):
            _walk(getattr(x, "children", None))
            # Badge children 之外，title 等 prop 也可能带文案
            for attr in ("title",):
                v = getattr(x, attr, None)
                if isinstance(v, str):
                    out.append(v)
        elif isinstance(x, (list, tuple)):
            for c in x:
                _walk(c)

    _walk(comp)
    return "".join(out)


def _find_all(comp, cls) -> list:
    """递归收集组件树里所有 cls 实例（用于 Tabs/Progress 等组件断言）。"""
    out = []

    def _walk(x):
        if isinstance(x, cls):
            out.append(x)
        if isinstance(x, Component):
            _walk(getattr(x, "children", None))
        elif isinstance(x, (list, tuple)):
            for c in x:
                _walk(c)

    _walk(comp)
    return out


def _collect_ids(comp) -> set:
    """递归收集组件树里的全部字符串 id。"""
    ids = set()

    def _walk(x):
        if isinstance(x, Component):
            cid = getattr(x, "id", None)
            if isinstance(cid, str):
                ids.add(cid)
            _walk(getattr(x, "children", None))
        elif isinstance(x, (list, tuple)):
            for c in x:
                _walk(c)

    _walk(comp)
    return ids


def _write_session(sessions_dir, sid, mode, status, sharpe_kpi="+1.23", finished=None, extra=None):
    """构造一个最小会话目录：session.json + 一个带 Sharpe KPI 的 step report。

    extra 里的键会覆盖默认 meta（用于 running 会话的 current_progress 等）。
    """
    d = sessions_dir / sid
    (d / "steps").mkdir(parents=True)
    meta = {
        "id": sid,
        "mode": mode,
        "label": "测试",
        "params": {},
        "status": status,
        "created": time.time() - 100,
        "started": time.time() - 90,
        "finished": finished or (time.time() - 10),
        "current_step": 0,
        "steps": [{"step": "s1", "title": "步骤一", "status": "done"}],
    }
    meta.update(extra or {})
    (d / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    rep = {
        "step": "s1",
        "title": "步骤一",
        "summary": "",
        "kpis": [{"label": "Sharpe", "value": sharpe_kpi, "hint": ""}],
        "tables": [],
        "figures": [],
        "notes": [],
    }
    (d / "steps" / "01_s1.json").write_text(json.dumps(rep), encoding="utf-8")


# --- 空态（真实目录的常态） ------------------------------------------------------


def test_layout_renders_in_empty_sandbox(sandbox):
    comp = model_lab.layout()
    assert isinstance(comp, Component)
    assert callable(model_lab.register)


def test_layout_uses_page_header(sandbox):
    """页面头走 components.page_header：含工作流面包屑（链 /factors 等）。"""
    comp = model_lab.layout()
    assert "模型实验室 Model Lab" in _flatten_text(comp)
    links = _find_all(comp, dcc.Link)
    assert any(lk.href == "/factors" for lk in links)


def test_layout_has_guidance_and_runs_toggle(sandbox):
    """静态 layout 含引导条占位与 Runs 分页切换组件（按钮 + Store）。"""
    ids = _collect_ids(model_lab.layout())
    assert {"mlab-guidance", "mlab-runs-toggle", "mlab-runs-show-all"} <= ids


def test_guidance_states(sandbox):
    """引导条三态互斥：无 champion / 有 champion 无 staging / 有 staging。"""
    assert model_lab._guidance(None, ()) == "还没有线上模型：先在下方创建「训练候选模型」任务"
    man = {"model_id": "m1"}
    assert model_lab._guidance(man, ({"id": "c1"},)) == (
        "有待晋升候选：在下方「模型版本」区对比并决定晋升"
    )
    assert model_lab._guidance(man, ()) == (
        "线上模型运行中：想验证它，先跑一次「滚动Walk-Forward」"
    )
    # staging 存在但无 champion 时优先训练引导
    assert "训练候选模型" in model_lab._guidance(None, ({"id": "c1"},))


def test_champion_card_empty_state(sandbox):
    """无 champion（models/ 为空）→ 空态引导：交易台 bootstrap / staging 手动晋升。"""
    assert dataio.champion_manifest() is None
    card = model_lab._champion_card(None)
    txt = _flatten_text(card)
    assert "暂无 champion" in txt
    assert "bootstrap" in txt  # 交易台出单会话自动建立首个 champion
    assert "predict_tomorrow.py" not in txt  # 主路径不再给 CLI 命令


def test_staging_list_empty(sandbox):
    assert dataio.staging_manifests() == ()
    assert "暂无 staging" in _flatten_text(model_lab._staging_list(()))


def test_timeline_empty(sandbox):
    assert model_lab._timeline_points() == []
    fig = model_lab._timeline_fig([])
    assert fig.layout.annotations  # empty_fig 的占位提示


def test_runs_table_empty(sandbox):
    assert "暂无任务" in _flatten_text(model_lab._runs_table(None))


def test_runs_table_default_truncates_and_show_all(sandbox):
    """默认只渲染最近 _RUNS_PAGE_SIZE 条会话；show_all=True 渲染全部；
    canonical CLI 置顶行不受截断影响。"""
    sessions_dir = sandbox / "outputs" / "sessions"
    for i in range(25):
        _write_session(sessions_dir, f"factoreval_20261001_{i:06d}", "factoreval", "done")
    meta = {"generated_at": "2026-10-18T15:21:30", "metrics": {"sharpe": 1.5}, "n_windows": 3}
    (sandbox / "outputs" / "walkforward" / "meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )

    default_txt = _flatten_text(model_lab._runs_table(model_lab._canonical_wf()))
    assert default_txt.count("factoreval_20261001_") == model_lab._RUNS_PAGE_SIZE
    assert "outputs/walkforward" in default_txt
    all_txt = _flatten_text(model_lab._runs_table(model_lab._canonical_wf(), show_all=True))
    assert all_txt.count("factoreval_20261001_") == 25


# --- runs 表进度列 -----------------------------------------------------------------


def _running_meta_extra():
    return {
        "steps": [
            {"step": "s1", "title": "数据导入", "status": "done"},
            {"step": "s2", "title": "训练候选模型", "status": "running"},
            {"step": "s3", "title": "门控评估", "status": "pending"},
        ],
        "current_step": 1,
        "finished": None,
        "current_progress": {
            "step_index": 1,
            "step_id": "s2",
            "step_title": "训练候选模型",
            "phase": "训练中",
            "current": 2,
            "total": 5,
            "pct": 0.5,
            "elapsed_sec": 120.0,
            "eta_sec": 540.0,
            "message": "fold 2/5",
            "extras": {},
            "ts": time.time(),
        },
    }


def test_runs_table_running_progress_bar(sandbox):
    """running 会话：进度条 = (done步数 + 当前步pct) / 总步数，下方小字带 phase/ETA。"""
    sessions_dir = sandbox / "outputs" / "sessions"
    _write_session(
        sessions_dir, "train_20261022_100000", "train", "running", extra=_running_meta_extra()
    )
    table = model_lab._runs_table(None)
    bars = _find_all(table, dbc.Progress)
    assert len(bars) == 1
    assert bars[0].value == pytest.approx(50.0)  # (1 done + 0.5) / 3
    txt = _flatten_text(table)
    assert "训练中" in txt and "fold 2/5" in txt and "ETA 9m" in txt


def test_runs_table_running_without_current_progress(sandbox):
    """current_progress 缺省 → 容错为 done步数/总步数，不抛异常。"""
    sessions_dir = sandbox / "outputs" / "sessions"
    extra = _running_meta_extra()
    extra["current_progress"] = None
    _write_session(sessions_dir, "train_20261022_100000", "train", "running", extra=extra)
    table = model_lab._runs_table(None)
    bars = _find_all(table, dbc.Progress)
    assert len(bars) == 1
    assert bars[0].value == pytest.approx(33.3)  # 1 done / 3，渲染前 round 到 0.1
    assert "等待子阶段上报" in _flatten_text(table)


def test_runs_table_queued_and_terminal(sandbox):
    """queued → '排队中' 文本；终态保持 done数/总步数，均无进度条。"""
    sessions_dir = sandbox / "outputs" / "sessions"
    _write_session(sessions_dir, "factoreval_20261022_100000", "factoreval", "queued")
    _write_session(sessions_dir, "walkforward_20261021_100000", "walkforward", "done")
    table = model_lab._runs_table(None)
    assert _find_all(table, dbc.Progress) == []
    txt = _flatten_text(table)
    assert "排队中" in txt
    assert "1/1" in txt


def test_fmt_eta(sandbox):
    assert model_lab._fmt_eta(45) == "45s"
    assert model_lab._fmt_eta(540) == "9m"
    assert model_lab._fmt_eta(5400) == "1.5h"
    assert model_lab._fmt_eta(0) == "" and model_lab._fmt_eta(None) == ""


def test_canonical_missing(sandbox):
    assert model_lab._canonical_wf() is None
    assert "无 canonical" in _flatten_text(model_lab._canonical_modal_body(None))


# --- champion 卡片（非空） -------------------------------------------------------


def _champion_manifest(created_at: str) -> dict:
    return {
        "model_id": "20261001_ab12cd34ef56",
        "created_at": created_at,
        "fingerprint": "ab12cd34ef56",
        "config_snapshot": {
            "feat_cols": [f"f{i}" for i in range(42)],
            "fw": 20,
            "embargo": 20,
            "ensemble_models": "lgb,xgb",
            "n_seeds": 3,
            "incr_trees": 60,
            "rebuild_days": 60,
            "train_window": 500,
            "ic_window": 20,
        },
        "promotion": None,
        "schema_version": 1,
    }


def test_champion_card_fresh(sandbox):
    man = _champion_manifest(pd.Timestamp.now().isoformat(timespec="seconds"))
    card = model_lab._champion_card(man)
    txt = _flatten_text(card)
    assert man["model_id"] in txt
    assert "特征 42 个" in txt
    assert "无晋升评估记录" in txt  # promotion=None 的兜底文案
    badge = model_lab._age_badge(man)
    assert badge.color == "success"


def test_champion_card_age_warning(sandbox):
    """创建时间 100 天前 → 超 84 自然日亮黄灯。"""
    old = (pd.Timestamp.now() - pd.Timedelta(days=100)).isoformat(timespec="seconds")
    man = _champion_manifest(old)
    badge = model_lab._age_badge(man)
    assert badge.color == "warning"
    assert "100" in _flatten_text(badge)


# --- staging 结论解析 -------------------------------------------------------------


def test_staging_summary_with_evaluation(sandbox):
    entry = {
        "id": "20261020_ff00aa11bb22",
        "manifest": {"model_id": "20261020_ff00aa11bb22"},
        "evaluation": {
            "candidate": {"sharpe": 1.50, "max_drawdown": -5.0},
            "champion": {"sharpe": 1.00, "max_drawdown": -8.0},
            "decision": "rejected",
            "reason": "Sharpe +1.50 未超过 champion +1.00 + 门槛 0.60",
            "decided_at": "2026-10-20T18:00:00",
        },
    }
    s = model_lab._staging_summary(entry)
    assert s["id"] == "20261020_ff00aa11bb22"
    assert s["d_sharpe"] == pytest.approx(0.50)
    assert s["d_dd"] == pytest.approx(3.0)
    assert s["eligible"] is False  # rejected → 未达门槛，仅可人工强制晋升
    card = model_lab._staging_list((entry,))
    txt = _flatten_text(card)
    assert "+0.50" in txt and "未达门槛" in txt and s["reason"] in txt


def test_staging_summary_without_evaluation(sandbox):
    """evaluation.json 缺失 → 空态文案，不抛异常。"""
    entry = {"id": "cand_x", "manifest": {"model_id": "cand_x"}, "evaluation": None}
    s = model_lab._staging_summary(entry)
    assert s["has_eval"] is False and s["d_sharpe"] is None
    assert "尚无门控评估" in _flatten_text(model_lab._staging_list((entry,)))


def test_staging_summary_promoted(sandbox):
    entry = {
        "id": "cand_y",
        "manifest": None,
        "evaluation": {
            "candidate": {"sharpe": 2.0, "max_drawdown": -4.0},
            "champion": {"sharpe": 1.0, "max_drawdown": -6.0},
            "decision": "promoted",
            "reason": "达标",
            "decided_at": "2026-10-21T18:00:00",
        },
    }
    s = model_lab._staging_summary(entry)
    assert s["eligible"] is True
    assert "已达晋升门槛" in _flatten_text(model_lab._staging_list((entry,)))


# --- 验证历史时间线 -----------------------------------------------------------------


def test_timeline_points_assembly(sandbox):
    """完成的研究类与历史（backtest/robustness）会话打点；running 与 nextday 排除。"""
    sessions_dir = sandbox / "outputs" / "sessions"
    _write_session(sessions_dir, "walkforward_20261020_100000", "walkforward", "done", "+1.23")
    _write_session(sessions_dir, "backtest_20261021_100000", "backtest", "done", "+0.55")
    _write_session(sessions_dir, "robustness_20261021_110000", "robustness", "done", "+0.90 ± 0.30")
    _write_session(sessions_dir, "train_20261021_120000", "train", "done", "+1.05")
    _write_session(sessions_dir, "walkforward_20261022_100000", "walkforward", "running")
    _write_session(sessions_dir, "nextday_20261022_100000", "nextday", "done")

    points = model_lab._timeline_points()
    assert len(points) == 4
    by_sid = {p["sid"]: p for p in points}
    assert by_sid["walkforward_20261020_100000"]["sharpe"] == pytest.approx(1.23)
    assert by_sid["backtest_20261021_100000"]["sharpe"] == pytest.approx(0.55)
    # robustness 的 '+0.90 ± 0.30' 取首个数字
    assert by_sid["robustness_20261021_110000"]["sharpe"] == pytest.approx(0.90)
    assert by_sid["train_20261021_120000"]["sharpe"] == pytest.approx(1.05)
    assert all(not p["canonical"] for p in points)
    # 按时间升序
    assert [p["t"] for p in points] == sorted(p["t"] for p in points)


def test_timeline_canonical_point(sandbox):
    """canonical outputs/walkforward/meta.json → 大号菱形点，sid 占位不链会话。"""
    wdir = sandbox / "outputs" / "walkforward"
    meta = {
        "generated_at": "2026-10-18T15:21:30",
        "metrics": {"sharpe": 2.48, "total_pnl": 1.2},
        "n_windows": 11,
        "sharpe_pos_ratio": 0.8,
        "predic_pos_ratio": 0.6,
    }
    (wdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    points = model_lab._timeline_points()
    assert len(points) == 1
    p = points[0]
    assert p["canonical"] is True and p["sid"] == model_lab._CANONICAL_SID
    assert p["sharpe"] == pytest.approx(2.48)

    fig = model_lab._timeline_fig(points)
    assert any(tr.marker.symbol == "diamond" for tr in fig.data)
    # ±0.4 随机波动带
    hrects = [sh for sh in fig.layout.shapes if sh.type == "rect"]
    assert hrects and hrects[0].y0 == pytest.approx(2.48 - model_lab._NOISE_BAND)


def test_timeline_session_sharpe_prefers_exact_label(sandbox):
    """kpi 标签匹配：精确 'Sharpe' 优先于含 sharpe 的其他标签。"""
    sessions_dir = sandbox / "outputs" / "sessions"
    sid = "walkforward_20261020_100000"
    d = sessions_dir / sid
    (d / "steps").mkdir(parents=True)
    (d / "session.json").write_text(
        json.dumps(
            {
                "id": sid,
                "mode": "walkforward",
                "status": "done",
                "created": time.time() - 100,
                "finished": time.time() - 10,
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    rep = {
        "kpis": [
            {"label": "窗口 Sharpe 均值", "value": "+9.99", "hint": ""},
            {"label": "Sharpe", "value": "+1.11", "hint": ""},
        ]
    }
    (d / "steps" / "01.json").write_text(json.dumps(rep), encoding="utf-8")
    assert model_lab._session_sharpe(sid) == pytest.approx(1.11)


# --- canonical meta 兜底现算 --------------------------------------------------------


def test_canonical_pos_ratio_fallback(sandbox):
    """meta.json 缺 sharpe_pos_ratio/predic_pos_ratio → 从 window_metrics.parquet 现算。"""
    wdir = sandbox / "outputs" / "walkforward"
    meta = {"generated_at": "2026-10-18T15:21:30", "metrics": {"sharpe": 1.5}, "n_windows": 3}
    (wdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    wm = pd.DataFrame({"Sharpe": [1.0, -0.5, 2.0], "PredIC": [0.1, 0.2, -0.3]})
    wm.to_parquet(wdir / "window_metrics.parquet", index=False)

    out = model_lab._canonical_wf()
    assert out["sharpe_pos_ratio"] == pytest.approx(2 / 3)
    assert out["predic_pos_ratio"] == pytest.approx(2 / 3)


def test_trust_badges_thresholds(sandbox):
    """可信度徽章阈值：Sharpe>0 占比 <60% 红，PredIC>0 占比 <50% 红。"""
    badges = model_lab._trust_badges({"sharpe_pos_ratio": 0.5, "predic_pos_ratio": 0.8})
    assert badges[0].color == "danger" and badges[1].color == "success"
    badges = model_lab._trust_badges({"sharpe_pos_ratio": None, "predic_pos_ratio": None})
    assert badges == []


# --- 新建任务表单 / diff --------------------------------------------------------------


def test_param_form_covers_all_modes(sandbox):
    """PARAM_SCHEMA 全模式都能动态渲染出表单行；ETA 提示只覆盖研究类模式。"""
    for mode in registry.PARAM_SCHEMA:
        form = model_lab._param_inputs(mode)
        assert isinstance(form, Component)
    assert set(model_lab._ETA_HINT) == {"walkforward", "train", "modeleval", "factoreval"}
    assert "≈ 5 小时" in model_lab._ETA_HINT["walkforward"]
    assert "门控双回放" in model_lab._ETA_HINT["train"]


def test_create_card_research_tabs_only(sandbox):
    """「新建任务」面板只渲染研究类四模式 tab，默认激活 walkforward。"""
    card = model_lab._create_card()
    tabs = _find_all(card, dbc.Tabs)
    assert len(tabs) == 1
    tab_ids = [t.tab_id for t in tabs[0].children]
    assert tab_ids == ["walkforward", "train", "modeleval", "factoreval"]
    assert tabs[0].active_tab == "walkforward"
    txt = _flatten_text(card)
    assert "次日交易" not in txt  # production 模式不出现在创建面板


def test_recipe_diff_lines(sandbox):
    man = _champion_manifest(pd.Timestamp.now().isoformat(timespec="seconds"))
    params = {"train_window": 250, "n_seeds": 3, "capital": 1000000}
    lines = model_lab._recipe_diff_lines("walkforward", params, man)
    assert any("train_window: champion=500 → 本次=250" in ln for ln in lines)
    assert not any("n_seeds" in ln for ln in lines)  # 与 champion 相同 → 不列
    assert model_lab._recipe_diff_lines("walkforward", params, None) == []


def test_create_notice_has_runs_link(sandbox):
    """创建成功 Alert 含「查看详情 →」链接指向 /runs/<sid>。"""
    man = _champion_manifest(pd.Timestamp.now().isoformat(timespec="seconds"))
    alert = model_lab._create_notice("walkforward", "walkforward_20261026_100000", {}, man)
    txt = _flatten_text(alert)
    assert "已创建会话 walkforward_20261026_100000" in txt
    assert "查看详情" in txt
    links = _find_all(alert, dcc.Link)
    assert any(lk.href == "/runs/walkforward_20261026_100000" for lk in links)


# --- 审计 / 删除 staging（不动真实 models/） -------------------------------------------


def test_delete_staging_and_audit(sandbox):
    stg = sandbox / "models" / "staging" / "cand_z"
    stg.mkdir(parents=True)
    (stg / "manifest.json").write_text(json.dumps({"model_id": "cand_z"}), encoding="utf-8")
    ok, msg = model_lab._do_delete_staging("cand_z")
    assert ok, msg
    assert not stg.exists()
    log = sandbox / "models" / "promotions.log"
    assert log.exists() and "DELETE_STAGING cand_z" in log.read_text(encoding="utf-8")
    # 路径穿越防护
    ok, _ = model_lab._do_delete_staging("../etc")
    assert not ok
