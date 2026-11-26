"""因子页：主表组装/筛选（自旧前端迁入，口径不变）、评估计划、草稿保存、页面渲染与路由。

关键回归点：
  · 库里的公式因子默认不在因子面板里（USE_FORMULA_FACTORS=0），评估它必须走 --draft 现算，
    且 --draft 不读写共享面板缓存；原生因子直接取缓存面板。
  · 全库评估的回看天数不能越过面板缓存的起点（越过会让夜间任务反复读一份更大的面板）。
  · 保存草稿要与整个库（不只是 custom.yaml）去重，并且只写 custom.yaml。
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import pytest
import yaml

import run_registry as rr
from webapp2 import actions, data, factor_logic as fl
from webapp2.pages import factors

from tests.test_webapp2_pages import ids_of, text_of  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(data, "ROOT", str(tmp_path))
    monkeypatch.setattr(data, "OUTPUTS", str(out))
    monkeypatch.setattr(data, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(data, "CACHE", str(tmp_path / "data_cache"))
    monkeypatch.setattr(data, "FACTOR_LIBRARY", str(tmp_path / "factor_library"))
    data.refresh()
    yield tmp_path
    data.refresh()


def put_library(sandbox, **files):
    d = sandbox / "factor_library"
    d.mkdir(exist_ok=True)
    for name, factors_ in files.items():
        (d / f"{name}.yaml").write_text(yaml.safe_dump({"factors": factors_}, allow_unicode=True))
    data.refresh()


def fdef(fid, formula="close / ts_delay(close, 5) - 1", **kw):
    return {
        "id": fid,
        "name": kw.pop("name", fid),
        "formula": formula,
        "enabled": kw.pop("enabled", True),
        **kw,
    }


def put_panel_cache(sandbox, cols=("ep", "bp", "mom_5"), req_start="2024-04-01"):
    d = sandbox / "data_cache"
    d.mkdir(exist_ok=True)
    n = 6
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=n),
            "symbol": ["A"] * n,
            "close": 1.0,
            "responder_6": 0.0,
        }
    )
    for c in cols:
        df[c] = np.arange(n, dtype=float)
    df.to_parquet(d / "factor_panel_cache.parquet")
    (d / "factor_panel_cache.sig.json").write_text(
        json.dumps({"sig": "x", "req_start": req_start, "data_start": req_start})
    )
    data.refresh()


def put_eval(sandbox, rows):
    pd.DataFrame(rows).to_csv(sandbox / "outputs" / "factor_eval.csv", index=False)
    data.refresh()


def eval_row(factor, rankic=0.05, icir=0.5, gate="✓"):
    return {
        "factor": factor,
        "RankIC": rankic,
        "RankICIR": icir,
        "IC_t": 5.0,
        "多空收益": 0.01,
        "单调性": 0.8,
        "换手": 0.3,
        "准入": gate,
    }


# ----------------------------------------------------------------------------- 迁入的纯逻辑
@pytest.mark.parametrize(
    "name,expected",
    [
        ("ep", "价值"),
        ("a158_kmid", "Alpha158"),
        ("gtja_alpha001", "GTJA191"),
        ("north_ratio", "北向资金"),
        ("mx_north_flow", "北向交叉"),
        ("return_5d", "反转"),
        ("volatility_20", "波动率"),
        ("zzz_unknown", "其他原生"),
    ],
)
def test_infer_category(name, expected):
    assert fl.infer_category(name) == expected


def test_sparkline_and_clusters():
    assert fl.sparkline([]) == "" and fl.sparkline([float("nan")]) == ""
    assert (
        fl.sparkline([1, 1, 1]) == "▄▄▄"
        and fl.sparkline([0, 1])[0] == "▁"
        and fl.sparkline([0, 1])[-1] == "█"
    )
    corr = pd.DataFrame(
        [[1, 0.9, 0.1], [0.9, 1, 0.2], [0.1, 0.2, 1]], columns=list("abc"), index=list("abc")
    )
    cl = fl.corr_clusters(corr.reset_index().rename(columns={"index": ""}))
    assert cl["a"] == cl["b"] and "c" not in cl and "2因子" in cl["a"]
    assert fl.corr_clusters(pd.DataFrame()) == {}


def test_build_overview_joins_library_native_eval_and_contrib():
    lib = pd.DataFrame([{"id": "f1", "name": "F1", "category": "momentum", "enabled": False}])
    ev = pd.DataFrame([eval_row("f1"), eval_row("ep", 0.02, 0.2, "")])
    df = fl.build_overview(
        lib, ev, pd.DataFrame(), pd.DataFrame(), pd.Series({"ep": 3.0, "f1": 1.0}), ["ep", "bp"]
    )
    by = df.set_index("因子")
    assert list(df["因子"]) == ["f1", "ep", "bp"]  # 库因子在前，原生因子并入
    assert by.loc["f1", "来源"] == "公式库" and not by.loc["f1", "启用"]
    assert by.loc["f1", "类别"] == "动量" and by.loc["ep", "类别"] == "价值"
    assert by.loc["f1", "准入"] == "✓" and np.isnan(by.loc["bp", "RankIC"])
    assert by.loc["ep", "贡献分位"] == 100.0  # 贡献分最高 → 分位 100
    assert fl.build_overview(None, None, None, None, None, None).empty


def test_apply_filters():
    lib = pd.DataFrame(
        [{"id": "f1", "name": "Alpha one", "category": "momentum", "enabled": False}]
    )
    df = fl.build_overview(
        lib,
        pd.DataFrame([eval_row("f1"), eval_row("ep", 0.0, 0.0, "")]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Series(dtype=float),
        ["ep", "bp"],
    )
    assert list(fl.apply_filters(df, gate="已通过")["因子"]) == ["f1"]
    assert list(fl.apply_filters(df, gate="未通过")["因子"]) == ["ep"]  # 已评估但没通过
    assert list(fl.apply_filters(df, gate="未评估")["因子"]) == ["bp"]
    assert list(fl.apply_filters(df, enabled="已停用")["因子"]) == ["f1"]
    assert set(fl.apply_filters(df, enabled="原生因子")["因子"]) == {"ep", "bp"}
    assert list(fl.apply_filters(df, text="ALPHA")["因子"]) == ["f1"]  # 名称、大小写不敏感
    assert fl.apply_filters(pd.DataFrame(), gate="已通过").empty


# ----------------------------------------------------------------------------- 数据层
def test_factor_library_reader_matches_loader_rules(sandbox):
    put_library(
        sandbox,
        a=[fdef("one"), fdef("two", enabled=False)],
        b=[fdef("one", name="重复，应被忽略"), fdef("three")],
    )
    (sandbox / "factor_library" / "_manifest.yaml").write_text("exclude_ids: [three]\n")
    data.refresh()
    lib = data.factor_library()
    assert list(lib["id"]) == ["one", "two"]  # 重复与排除名单都不进入
    assert not lib.set_index("id").loc["two", "enabled"]
    assert lib.set_index("id").loc["one", "file"] == "a.yaml"


def test_factor_library_corrupt_and_missing(sandbox):
    assert data.factor_library().empty  # 目录不存在
    put_library(sandbox, ok=[fdef("a")])
    (sandbox / "factor_library" / "broken.yaml").write_text("factors: [")
    data.refresh()
    assert list(data.factor_library()["id"]) == ["a"]  # 坏文件跳过，不拖垮整库


def test_panel_columns_exclude_meta_and_labels(sandbox):
    put_panel_cache(sandbox, cols=("ep", "mom_5"))
    cols = set(data.panel_factor_columns())
    assert cols == {"ep", "mom_5"}
    assert {"close", "date", "responder_6"} <= data.base_columns()
    assert data.panel_cache_meta()["req_start"] == "2024-04-01"
    os.remove(sandbox / "data_cache" / "factor_panel_cache.parquet")
    data.refresh()
    assert data.panel_factor_columns() == ()  # 缓存删掉后读不到


def test_readers_return_empty_state_not_exceptions(sandbox):
    assert (
        data.factor_ic_decay().empty and data.factor_corr().empty and data.model_vs_factor().empty
    )
    assert data.factor_layers("x").empty and data.factor_daily_ic("x").empty
    contrib, scores = data.factor_contrib()
    assert contrib.empty and scores.empty
    assert data.library_signature() == [] and data.panel_cache_meta() is None


def test_layer_filenames_follow_the_producers_sanitising_rule(sandbox):
    """落盘端 src/factor_layers.safe_factor_id 把非法字符折叠为 _；读取端必须同一规则，否则读不到。"""
    from factor_layers import safe_factor_id

    for fid in ("a158_cntd5", "weird id!", "x/y", "ep.v2"):
        pd.DataFrame({"date": ["2026-01-01"], "rank_ic": [0.1]}).to_parquet(
            sandbox / "outputs" / f"factor_ic_daily_{safe_factor_id(fid)}.parquet"
        )
    data.refresh()
    for fid in ("a158_cntd5", "weird id!", "x/y", "ep.v2"):
        assert len(data.factor_daily_ic(fid)) == 1, fid


# ----------------------------------------------------------------------------- 评估计划
def test_library_factor_is_evaluated_with_draft_flag_and_native_with_subset(sandbox):
    put_panel_cache(sandbox)
    assert actions.plan_factor_eval("my_f", {"my_f"}) == [
        "--draft",
        "my_f",
        "--set",
        "lookback_days=730",
    ]
    plan = actions.plan_factor_eval("ep", {"my_f"}, today="2026-01-01")
    assert plan[:2] == ["--set", "factor_subset=ep"] and plan[2:] == ["--set", plan[3]]
    assert "--draft" not in plan


@pytest.mark.parametrize("bad", ["", "a b", "a;b", "../x", "a-b", "1abc", "x" * 65, None])
def test_plan_rejects_unsafe_factor_ids(bad):
    with pytest.raises(ValueError):
        actions.plan_factor_eval(bad, set())


def test_full_eval_lookback_never_crosses_the_panel_cache_start(sandbox):
    """默认 1825 天越过缓存起点会让 load_factor_panel 重算并用更长区间覆盖缓存。"""
    put_panel_cache(sandbox, req_start="2024-04-01")
    days = actions.full_eval_lookback_days(today="2026-11-17")
    assert days == (pd.Timestamp("2026-11-17") - pd.Timestamp("2024-04-01")).days  # 恰好到缓存起点
    assert days < 1825
    # 夹在合法区间 [120, 3650]
    assert actions.full_eval_lookback_days(today="2024-04-05") == 120
    assert actions.full_eval_lookback_days(today="2050-01-01") == 3650


def test_full_eval_lookback_falls_back_without_cache_meta(sandbox):
    assert actions.full_eval_lookback_days() == actions.DEFAULT_PANEL_LOOKBACK_DAYS


def test_launch_uses_the_wrapper_and_blocks_while_another_heavy_job_runs(sandbox, monkeypatch):
    put_library(sandbox, a=[fdef("my_f")])
    live = rr.RunReporter(
        "walkforward", label="WF", run_id="w_live", root=str(sandbox / "outputs" / "runs")
    )
    live.step("x")
    ok, msg, rid = actions.launch_factor_eval("my_f")
    assert (
        not ok and "w_live" in msg and rid == "w_live"
    )  # 回测在跑时不能再起评估（共用面板缓存与内存）
    live.finish()
    captured = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            captured["cmd"], captured["env"] = cmd, kw.get("env", {})

    monkeypatch.setattr(actions.subprocess, "Popen", FakePopen)
    ok, msg, rid = actions.launch_factor_eval("my_f")
    assert ok, msg
    cmd = captured["cmd"]
    assert (
        cmd[2:6] == ["scripts/run_pipeline.py", "--mode", "factoreval", "--draft"] and "my_f" in cmd
    )
    assert captured["env"]["RUN_ID"] == rid


def test_every_pipeline_mode_is_launchable_and_whitelisted():
    for entry in ("train", "modeleval", "factoreval"):
        assert actions.CLI_ENTRIES[entry][:2] == ["scripts/run_pipeline.py", "--mode"]
    assert not actions.launch_cli("factoreval", ["--set", "x=1;rm"])[0]  # 参数白名单字符集


# ----------------------------------------------------------------------------- 保存草稿
def test_save_draft_writes_only_custom_yaml_disabled_and_audits(sandbox):
    put_library(sandbox, gtja=[fdef("existing_one")])
    put_panel_cache(sandbox)
    before = (sandbox / "factor_library" / "gtja.yaml").read_text()
    ok, msg, res = actions.save_factor_draft(
        "new_one", "新因子", "momentum", "close / ts_delay(close, 5) - 1"
    )
    assert ok, msg
    assert "全量重建" in msg  # 告知用户保存会让面板缓存失效一次
    saved = yaml.safe_load((sandbox / "factor_library" / "custom.yaml").read_text())["factors"]
    assert [f["id"] for f in saved] == ["new_one"] and saved[0]["enabled"] is False
    assert (sandbox / "factor_library" / "gtja.yaml").read_text() == before  # 不碰其他库文件
    assert "FACTOR_DRAFT new_one" in (sandbox / "models" / "promotions.log").read_text()
    assert "new_one" in set(data.factor_library()["id"])  # 缓存已刷新，页面立刻看得到


def test_save_draft_dedupes_against_the_whole_library_and_native_columns(sandbox):
    put_library(sandbox, gtja=[fdef("taken")])
    put_panel_cache(sandbox, cols=("ep",))
    for dup in ("taken", "ep"):  # 其他库文件里的 id / 面板原生因子，加载器遇到重复只会静默跳过
        ok, msg, _ = actions.save_factor_draft(dup, "x", "c", "close")
        assert not ok and "已存在" in msg
    assert not (sandbox / "factor_library" / "custom.yaml").exists()


def test_save_draft_rejects_unsafe_formulas_without_writing(sandbox):
    put_library(sandbox, gtja=[fdef("a")])
    ok, msg, res = actions.save_factor_draft("evil", "x", "c", "__import__('os').system('x')")
    assert not ok and "校验未通过" in msg and res["errors"]
    assert not (sandbox / "factor_library" / "custom.yaml").exists()


def test_saved_draft_reaches_the_overview_as_a_disabled_library_factor(sandbox):
    put_library(sandbox, gtja=[fdef("a")])
    put_panel_cache(sandbox)
    assert actions.save_factor_draft("draft_1", "草稿", "custom", "close / ts_delay(close, 5) - 1")[
        0
    ]
    df = factors.overview_frame().set_index("因子")
    assert df.loc["draft_1", "来源"] == "公式库" and not df.loc["draft_1", "启用"]
    assert "未评估" in text_of(factors._table(df.reset_index()))


# ----------------------------------------------------------------------------- 页面
def test_overview_on_empty_disk(sandbox):
    t = text_of(factors.layout("/factors"))
    for needle in (
        "没有符合筛选条件的因子",
        "没有 diag_factor_corr.csv",
        "没有 diag_model_vs_factor.csv",
        "还没有评估运行记录",
        "运行全库评估",
    ):
        assert needle in t, needle


def test_overview_renders_table_with_gate_badges(sandbox):
    put_library(sandbox, a=[fdef("lib_f")])
    put_panel_cache(sandbox, cols=("ep", "bp"))
    put_eval(sandbox, [eval_row("ep"), eval_row("bp", 0.0, 0.0, "")])
    t = text_of(factors.layout("/factors"))
    for needle in ("lib_f", "ep", "bp", "通过", "未通过", "未评估", "因子总数"):
        assert needle in t, needle


def test_filters_and_controls_live_outside_polled_regions(sandbox):
    ids = ids_of(factors.layout("/factors"))
    assert {
        "fw-tick",
        "fw-sig",
        "fw-kpis",
        "fw-charts",
        "fw-table",
        "fw-cat",
        "fw-gate",
        "fw-src",
        "fw-text",
        "fw-eval-all",
        "fw-eval-diag",
    } <= ids
    assert not (
        {"fw-cat", "fw-text", "fw-eval-all"} & ids_of(factors.overview_kpis())
        | ids_of(factors.overview_charts())
    )
    assert "没有符合" in text_of(factors.filtered_table("全部", "已通过", "全部", "没有这个因子"))


def test_status_shows_live_progress_from_run_registry(sandbox):
    r = rr.RunReporter(
        "factoreval", label="因子评估", run_id="fe1", root=str(sandbox / "outputs" / "runs")
    )
    r.step("eval_factors", "因子评估")
    r.progress("factors", 3, 10, "mom_5")
    t = text_of(factors._eval_status())
    assert "正在评估" in t and "fe1" in t and "factors" in t
    r.finish()
    assert "最近一次评估" in text_of(factors._eval_status())


def test_detail_for_library_factor_with_eval_and_artifacts(sandbox):
    put_library(sandbox, a=[fdef("lib_f", expected_sign=1, description="五日动量")])
    put_panel_cache(sandbox)
    put_eval(sandbox, [eval_row("lib_f")])
    pd.DataFrame(
        {"date": pd.bdate_range("2026-01-01", periods=30), "rank_ic": np.linspace(-0.05, 0.1, 30)}
    ).to_parquet(sandbox / "outputs" / "factor_ic_daily_lib_f.parquet")
    t = text_of(factors.layout("/factors/lib_f"))
    for needle in (
        "lib_f",
        "close / ts_delay(close, 5) - 1",
        "五日动量",
        "预期方向 +",
        "评估该因子",
        "RankIC",
    ):
        assert needle in t, needle
    assert "fd-eval" in ids_of(factors.layout("/factors/lib_f"))


def test_detail_for_native_factor_and_for_unknown_factor(sandbox):
    put_panel_cache(sandbox, cols=("ep",))
    assert "面板原生因子" in text_of(factors.layout("/factors/ep"))
    t = text_of(factors.layout("/factors/ghost"))
    assert "不在因子库" in t and "ghost" in t


def test_detail_without_any_artifacts_still_renders(sandbox):
    put_library(sandbox, a=[fdef("lib_f")])
    t = text_of(factors.layout("/factors/lib_f"))
    assert (
        "还没有评估结果" in t and "没有 IC 衰减数据" not in t
    )  # 空图以注释形式出现在 figure 里，不是文字节点


def test_editor_layout_and_validation_view(sandbox):
    ids = ids_of(factors.layout("/factors/new"))
    assert {"fe-id", "fe-name", "fe-formula", "fe-validate", "fe-save", "fe-result"} <= ids
    ok = factors._validation_view(
        {"errors": [], "warnings": ["依赖列 'x' 不在已知列中"], "deps": ["close"]}
    )
    t = text_of(ok)
    assert "校验通过" in t and "依赖列 'x'" in t and "close" in t
    bad = text_of(factors._validation_view({"errors": ["id 不能为空"], "warnings": [], "deps": []}))
    assert "校验未通过" in bad and "id 不能为空" in bad


def test_routing_dispatch_and_nav(sandbox):
    from webapp2 import app as appmod

    assert ("/factors", "因子") in appmod.NAV
    assert "运行全库评估" in text_of(appmod.route("/factors"))
    assert "新建公式因子" in text_of(appmod.route("/factors/new"))
    assert "因子不存在" in text_of(appmod.route("/factors/ghost"))
    assert appmod.nav_active("/factors/new", "/factors") and appmod.nav_active(
        "/factors/ep", "/factors"
    )


def test_signature_changes_with_each_dependency(sandbox):
    sig = factors.signature()
    assert factors.signature() == sig
    put_eval(sandbox, [eval_row("ep")])
    s1 = factors.signature()
    assert s1 != sig
    put_library(sandbox, a=[fdef("x")])
    s2 = factors.signature()
    assert s2 != s1
    with rr.RunReporter("factoreval", run_id="f9", root=str(sandbox / "outputs" / "runs")):
        pass
    assert factors.signature() != s2


def test_callbacks_register_without_duplicate_ids(sandbox):
    from webapp2 import app as appmod

    assert len(appmod.app.callback_map) >= 20
    # 路由回调只有 app.py 一处输出 page-content（重复输出会让路由随机失效）
    outs = [k for k in appmod.app.callback_map if "page-content" in k]
    assert len(outs) == 1
    assert time.time() > 0


def test_table_is_capped_and_says_so(sandbox):
    """真实数据 179 个因子时，不封顶的主表让页面高达 7600px，图表被埋在底部。"""
    n = factors.ROW_CAP + 30
    df = fl.build_overview(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.Series(dtype=float),
        [f"f{i:03d}" for i in range(n)],
    )
    t = text_of(factors._table(df))
    assert f"显示 {factors.ROW_CAP} / {n} 个" in t and "用上方筛选缩小范围" in t
    small = text_of(factors._table(df.head(5)))
    assert "显示 5 / 5 个" in small and "缩小范围" not in small


def test_heavy_charts_come_after_the_table(sandbox):
    """相关热图与模型对比图很高，必须排在主表之后，不能把主表挤到首屏之外。"""
    ids = []

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            [walk(c) for c in n]
            return
        if isinstance(getattr(n, "id", None), str):
            ids.append(n.id)
        walk(getattr(n, "children", None))

    walk(factors.layout("/factors"))
    assert ids.index("fw-table") < ids.index("fw-charts")
