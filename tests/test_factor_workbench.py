"""因子工作台（webapp/pages/factor_workbench.py）单元测试。

覆盖：主表数据组装（YAML+eval+decay+corr+contrib 多路 join 容错）、类别前缀
推断、冗余聚簇、筛选逻辑、编辑器静态校验与草稿落盘、子路由分派与布局冒烟。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from dash.development.base_component import Component

from webapp.pages import factor_workbench as fw


# ---------------------------------------------------------------------------
# 类别前缀推断
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("a158_kmid", "Alpha158"),
        ("gtja_alpha001", "GTJA191"),
        ("north_flow_net", "北向资金"),
        ("mx_north_flow_5d__return_5d", "北向交叉"),
        ("mx_us_vix_level__ln_mv", "宏观交叉"),
        ("ep", "价值"),
        ("bp_rank", "价值"),
        ("ln_mv", "价值"),
        ("value_composite", "价值"),
        ("opt_vix_like", "期权IV"),
        ("us_10y", "海外宏观"),
        ("cn_10y", "中国宏观"),
        ("momentum_20", "动量"),
        ("return_5d", "反转"),
        ("volume_ma_ratio_5", "量价"),
        ("volatility_20", "波动率"),
        ("rsi_14", "技术指标"),
        ("bb_position_20", "技术指标"),
        ("something_else", "其他原生"),
    ],
)
def test_infer_category(name, expected):
    assert fw.infer_category(name) == expected


# ---------------------------------------------------------------------------
# sparkline / 冗余聚簇
# ---------------------------------------------------------------------------
def test_sparkline():
    assert fw.sparkline([]) == ""
    assert fw.sparkline([np.nan, None]) == ""
    s = fw.sparkline([0.0, 1.0, 2.0])
    assert s == "▁▄█"
    assert len(fw.sparkline([1, 1, 1])) == 3  # 恒定序列


def _corr_df(pairs: dict) -> pd.DataFrame:
    """pairs: {(a,b): corr} → 方阵 CSV 形态（首列 unnamed 索引列）。"""
    labels = sorted({x for p in pairs for x in p})
    m = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for (a, b), v in pairs.items():
        m.loc[a, b] = v
        m.loc[b, a] = v
    out = m.reset_index().rename(columns={"index": "Unnamed: 0"})
    return out


def test_corr_clusters():
    corr = _corr_df({("a", "b"): 0.9, ("c", "d"): 0.85, ("d", "e"): 0.9, ("a", "c"): 0.1})
    cl = fw.corr_clusters(corr, threshold=0.8)
    # {a,b} 一簇，{c,d,e} 一簇
    assert cl["a"] == cl["b"]
    assert cl["c"] == cl["d"] == cl["e"]
    assert cl["a"] != cl["c"]
    assert "3因子" in cl["c"]


def test_corr_clusters_empty_and_singleton():
    assert fw.corr_clusters(pd.DataFrame()) == {}
    assert fw.corr_clusters(None) == {}
    # 全部低相关 → 无簇
    assert fw.corr_clusters(_corr_df({("a", "b"): 0.3})) == {}


# ---------------------------------------------------------------------------
# 主表组装（多路 join 容错）
# ---------------------------------------------------------------------------
def _lib_df():
    return pd.DataFrame(
        [
            {
                "id": "my_mom",
                "name": "我的动量",
                "description": "",
                "category": "momentum",
                "source": "custom",
                "formula": "ts_delta(log(close), 5)",
                "tags": ["mom"],
                "expected_sign": 1,
                "enabled": True,
                "version": 1,
                "author": "t",
                "created": "2026-01-01",
                "explicit_deps": [],
            }
        ]
    )


def _eval_df():
    return pd.DataFrame(
        {
            "factor": ["my_mom", "a158_kmid"],
            "RankIC": [0.031, -0.012],
            "RankICIR": [0.42, -0.2],
            "IC_t": [2.5, -1.1],
            "多空收益": [0.08, -0.03],
            "单调性": [0.9, 0.4],
            "换手": [0.3, 0.5],
            "准入": ["✓", ""],
        }
    )


def _decay_df():
    return pd.DataFrame(
        {
            "factor": ["my_mom"],
            "IC_mean": [0.03],
            "w0": [0.04],
            "w1": [0.03],
            "w2": [0.01],
        }
    )


def test_build_overview_full_join():
    corr = _corr_df({("my_mom", "a158_kmid"): 0.92})
    scores = pd.Series({"my_mom": 0.8, "a158_kmid": 0.2})
    df = fw.build_overview(
        _lib_df(), _eval_df(), _decay_df(), corr, scores, ["a158_kmid", "orphan_feat"]
    )
    assert list(df.columns) == fw.OVERVIEW_COLUMNS
    # 并集：YAML 1 + 原生 2 = 3 行
    assert len(df) == 3
    r = df.set_index("因子")
    assert r.loc["my_mom", "来源"] == "公式库"
    assert r.loc["my_mom", "RankIC"] == 0.031
    assert r.loc["my_mom", "ICIR"] == 0.42  # RankICIR 映射到展示列 ICIR
    assert r.loc["my_mom", "准入"] == "✓"
    assert r.loc["my_mom", "近期衰减"] != ""
    assert r.loc["my_mom", "冗余簇"] == r.loc["a158_kmid", "冗余簇"] != ""
    assert r.loc["my_mom", "贡献分位"] == 100.0
    assert r.loc["a158_kmid", "类别"] == "Alpha158"
    assert r.loc["orphan_feat", "来源"] == "原生"
    assert pd.isna(r.loc["orphan_feat", "RankIC"])  # 未评估 → NaN
    assert r.loc["orphan_feat", "准入"] == ""


def test_build_overview_all_empty():
    df = fw.build_overview(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, pd.Series(dtype=float), []
    )
    assert df.empty
    assert list(df.columns) == fw.OVERVIEW_COLUMNS


def test_build_overview_lib_only_and_dedupe():
    # eval/decay/corr/contrib 全缺；panel 列与 YAML id 重叠 → 去重
    df = fw.build_overview(_lib_df(), None, None, None, None, ["my_mom", "ep"])
    assert len(df) == 2
    r = df.set_index("因子")
    assert r.loc["my_mom", "来源"] == "公式库"  # 不重复出现
    assert r.loc["ep", "类别"] == "价值"
    assert df["RankIC"].isna().all()
    assert (df["近期衰减"] == "").all()


def test_apply_filters():
    corr = _corr_df({})
    df = fw.build_overview(
        _lib_df(), _eval_df(), _decay_df(), corr, pd.Series(dtype=float), ["a158_kmid"]
    )
    assert len(fw.apply_filters(df, gate="已通过")) == 1
    assert len(fw.apply_filters(df, gate="未通过")) == 1
    assert len(fw.apply_filters(df, gate="未评估")) == 0
    assert len(fw.apply_filters(df, category="Alpha158")) == 1
    assert len(fw.apply_filters(df, enabled="原生因子")) == 1
    assert len(fw.apply_filters(df, enabled="已启用")) == 1
    assert len(fw.apply_filters(df, enabled="已停用")) == 0
    assert len(fw.apply_filters(df, text="动量")) == 1
    assert len(fw.apply_filters(df, text="A158")) == 1
    assert len(fw.apply_filters(df, text="zzz")) == 0


# ---------------------------------------------------------------------------
# 编辑器校验
# ---------------------------------------------------------------------------
def test_check_formula_valid_and_deps():
    errors, deps = fw.check_formula("-ts_corr(cs_rank(close), cs_rank(volume), 6)")
    assert errors == []
    assert deps == {"close", "volume"}


def test_check_formula_forbidden():
    errors, _ = fw.check_formula("close.dropna()")
    assert any("禁止" in e or "属性" in e or "Attribute" in e for e in errors)
    errors, _ = fw.check_formula("__import__('os')")
    assert errors
    errors, _ = fw.check_formula("unknown_fn(close)")
    assert any("未知函数" in e for e in errors)
    errors, _ = fw.check_formula("close +* 2")
    assert any("语法错误" in e for e in errors)


def test_validate_factor_draft_ok():
    res = fw.validate_factor_draft(
        "my_rev",
        "反转",
        "reversal",
        "-ts_delta(log(close), 5)",
        tags="rev, short",
        expected_sign="-1",
        existing_ids={"other"},
        known_columns={"close"},
    )
    assert res["errors"] == []
    rec = res["record"]
    assert rec["enabled"] is False
    assert rec["expected_sign"] == -1
    assert rec["tags"] == ["rev", "short"]
    assert rec["source"] == "custom"


def test_validate_factor_draft_errors():
    # 空 id / 非法 id / 重复 id / 保留列 / 空公式
    for bad_id in ["", "1abc", "has space", "date", "responder_6"]:
        res = fw.validate_factor_draft(bad_id, "n", "c", "close", existing_ids=set())
        assert res["errors"], bad_id
    res = fw.validate_factor_draft("dup", "n", "c", "close", existing_ids={"dup"})
    assert any("已存在" in e for e in res["errors"])
    res = fw.validate_factor_draft("ok_id", "", "c", "")
    assert any("name" in e for e in res["errors"])
    assert any("公式" in e for e in res["errors"])
    # 未知依赖 → warning 而非 error
    res = fw.validate_factor_draft(
        "ok_id", "n", "c", "ts_mean(mystery_col, 5)", known_columns=set()
    )
    assert res["errors"] == []
    assert any("mystery_col" in w for w in res["warnings"])
    # 非法 expected_sign
    res = fw.validate_factor_draft("ok_id", "n", "c", "close", expected_sign="up")
    assert any("expected_sign" in e for e in res["errors"])


def test_save_draft_roundtrip(tmp_path):
    path = str(tmp_path / "custom.yaml")
    rec = fw.validate_factor_draft("d1", "草稿1", "custom", "close")["record"]
    fw.save_draft(rec, path)
    data = yaml.safe_load(open(path, encoding="utf-8"))
    assert [f["id"] for f in data["factors"]] == ["d1"]
    assert data["factors"][0]["enabled"] is False

    rec2 = fw.validate_factor_draft("d2", "草稿2", "custom", "open")["record"]
    fw.save_draft(rec2, path)
    data = yaml.safe_load(open(path, encoding="utf-8"))
    assert [f["id"] for f in data["factors"]] == ["d1", "d2"]

    with pytest.raises(ValueError, match="重复"):
        fw.save_draft(rec, path)


# ---------------------------------------------------------------------------
# 保存后「立即评估」（factoreval 会话）
# ---------------------------------------------------------------------------
def _find_by_id(comp, cid):
    """递归找组件树里 id == cid 的第一个组件。"""
    if isinstance(comp, Component):
        if getattr(comp, "id", None) == cid:
            return comp
        return _find_by_id(getattr(comp, "children", None), cid)
    if isinstance(comp, (list, tuple)):
        for c in comp:
            hit = _find_by_id(c, cid)
            if hit is not None:
                return hit
    return None


def test_build_factoreval_params():
    """参数组装：factoreval 默认参数 + 单因子子集 + 730 天回看，其余键不动。"""
    from pipeline import registry

    params = fw.build_factoreval_params("my_factor")
    defaults = registry.default_params("factoreval")
    assert params["factor_subset"] == "my_factor"
    assert params["lookback_days"] == 730
    assert params["lookback_days"] != defaults["lookback_days"]  # 确实被覆盖（默认 1825）
    for k, v in defaults.items():
        if k not in ("factor_subset", "lookback_days"):
            assert params[k] == v


def test_save_success_block_has_eval_button_and_cli_fallback():
    """保存成功块：含评估按钮 + 因子 id Store + 折叠的 CLI 逃生通道 + 详情链接。"""
    record = fw.validate_factor_draft("my_rev", "反转", "reversal", "close")["record"]
    block = fw._save_success_block(record)
    btn = _find_by_id(block, "fe-eval-btn")
    assert btn is not None and "立即评估" in str(btn.children)
    store = _find_by_id(block, "fe-eval-fid")
    assert store is not None and store.data == "my_rev"
    assert _find_by_id(block, "fe-eval-msg") is not None
    txt = str(block)
    assert "eval_factors.py" in txt  # CLI 逃生通道仍在
    assert "查看因子详情" in txt and "/factors/my_rev" in txt


def test_editor_layout_has_redirect_location():
    """编辑器布局挂载 dcc.Location(id='fe-redirect') 供评估后跳转 /runs/<sid>。"""
    loc = _find_by_id(fw.editor_layout(), "fe-redirect")
    assert loc is not None


def test_launch_factoreval_session(tmp_path, monkeypatch):
    """创建会话落盘：mode/label/params 正确；tick 打桩不起子进程。"""
    from pipeline import scheduler

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(scheduler, "SESSIONS_DIR", str(sessions))
    monkeypatch.setattr(scheduler, "tick", lambda: None)  # 不在测试里起子进程
    sid = fw.launch_factoreval_session("my_factor")
    assert sid.startswith("factoreval_")
    meta = json.loads((sessions / sid / "session.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "factoreval"
    assert meta["label"] == "单因子评估 my_factor"
    assert meta["status"] == "queued"
    assert meta["params"]["factor_subset"] == "my_factor"
    assert meta["params"]["lookback_days"] == 730


# ---------------------------------------------------------------------------
# 全库评估入口（总览页 B 处）与主表列组
# ---------------------------------------------------------------------------
def test_overview_has_full_eval_entry():
    """总览页：全库评估按钮 + 确认 Modal（含取消/确认/报错位）+ 跳转 Location。"""
    lay = fw.layout()
    btn = _find_by_id(lay, "fw-eval-btn")
    assert btn is not None and "运行全库评估" in str(btn.children)
    assert _find_by_id(lay, "fw-eval-modal") is not None
    assert _find_by_id(lay, "fw-eval-confirm") is not None
    assert _find_by_id(lay, "fw-eval-cancel") is not None
    assert _find_by_id(lay, "fw-eval-msg") is not None
    assert _find_by_id(lay, "fw-redirect") is not None


def test_launch_full_eval_session(tmp_path, monkeypatch):
    """全库评估会话：mode/label 正确，params 恰为 registry 默认参数；tick 打桩。"""
    from pipeline import registry, scheduler

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(scheduler, "SESSIONS_DIR", str(sessions))
    monkeypatch.setattr(scheduler, "tick", lambda: None)
    sid = fw.launch_full_eval_session()
    assert sid.startswith("factoreval_")
    meta = json.loads((sessions / sid / "session.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "factoreval"
    assert meta["label"] == "全库因子评估"
    assert meta["params"] == registry.default_params("factoreval")


def test_column_groups():
    """列组：核心组是 OVERVIEW_COLUMNS 的子集且默认启用；all 组出全量 15 列。"""
    assert set(fw.CORE_COLUMNS).issubset(set(fw.OVERVIEW_COLUMNS))
    assert len(fw.CORE_COLUMNS) == 9
    core = fw._table_columns("core")
    assert [c["id"] for c in core] == fw.CORE_COLUMNS
    assert [c["id"] for c in fw._table_columns("all")] == fw.OVERVIEW_COLUMNS
    assert fw._table_columns() == core  # 默认核心组
    # 总览布局默认核心组，并挂列组切换组件
    lay = fw.layout()
    toggle = _find_by_id(lay, "fw-col-group")
    assert toggle is not None and toggle.value == "core"
    table = _find_by_id(lay, "fw-table")
    assert [c["id"] for c in table.columns] == fw.CORE_COLUMNS
    assert set(table.tooltip_header) == set(fw.OVERVIEW_COLUMNS)


def test_detail_layout_has_reeval_button():
    """详情页头部：「重新评估该因子」按钮 + 因子 id Store + 跳转 Location。"""
    lay = fw.detail_layout("rsi_14")
    btn = _find_by_id(lay, "fwd-reeval-btn")
    assert btn is not None and "重新评估" in str(btn.children)
    store = _find_by_id(lay, "fwd-fid")
    assert store is not None and store.data == "rsi_14"
    assert _find_by_id(lay, "fwd-redirect") is not None
    assert _find_by_id(lay, "fwd-reeval-msg") is not None


# ---------------------------------------------------------------------------
# 布局与子路由冒烟
# ---------------------------------------------------------------------------
def test_route_content_dispatch():
    assert fw.route_content("/factors") is None
    assert fw.route_content("/models") is None
    assert fw.route_content(None) is None
    assert isinstance(fw.route_content("/factors/new"), Component)
    assert isinstance(fw.route_content("/factors/rsi_14"), Component)


def test_layouts_render():
    assert isinstance(fw.layout(), Component)
    # 存在的原生因子 → 详情；不存在的 → 告警页；均不抛异常
    assert isinstance(fw.detail_layout("rsi_14"), Component)
    assert isinstance(fw.detail_layout("no_such_factor_xyz"), Component)
    assert isinstance(fw.editor_layout(), Component)


def test_register_smoke():
    import dash
    from dash import dcc, html

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page-content")])
    fw.register(app)
    # 本页的过滤/编辑等回调已注册（page-content 路由回调只属于 app.py，
    # 本模块不得注册 page-content 输出——重复输出分派曾导致随机路由错误）
    specs = [str(v) for v in app.callback_map.values()]
    assert any("fw-table" in s for s in specs)
    assert not any("page-content" in k for k in app.callback_map)


def test_subroute_dispatch_end_to_end(monkeypatch):
    """真实 app + test_client 验证唯一路由回调的前缀分派（回归：曾因子工作台
    子路由与会话详情被 allow_duplicate 重复回调的随机分派送进交易台兜底页）：
    /factors/<id> → 详情、/factors/new → 编辑器、/models → 模型实验室、
    /runs/<id> → 会话详情、/factors → 总览。"""
    import json

    monkeypatch.setenv("DEBUG", "1")
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    from webapp import app as app_module

    app = app_module.app
    dup_ids = [k for k in app.callback_map if k.startswith("page-content.children@")]
    assert not dup_ids, "page-content 不应存在 allow_duplicate 重复回调"
    assert "page-content.children" in app.callback_map
    client = app.server.test_client()

    def post(pathname):
        payload = {
            "output": "page-content.children",
            "outputs": {"id": "page-content", "property": "children"},
            "inputs": [{"id": "url", "property": "pathname", "value": pathname}],
            "changedPropIds": ["url.pathname"],
            "state": [],
        }
        r = client.post("/_dash-update-component", json=payload)
        assert r.status_code == 200
        return json.dumps(r.get_json(), ensure_ascii=False)

    assert "rsi_14" in post("/factors/rsi_14")  # 单因子详情
    assert "公式因子编辑器" in post("/factors/new")  # 编辑器
    assert "模型实验室" in post("/models")
    assert "sd-body" in post("/runs/some_session")  # 会话详情骨架
    assert "因子工作台" in post("/factors")  # 本页总览
