"""webapp/components.py 共享组件契约测试：

· page_header 渲染标题/副标题/面包屑，flow_step=None 时不渲染面包屑；
· flow_breadcrumb 当前步实心高亮、其余步为可点击 dcc.Link；
· term 产出 .gloss-term Span + dbc.Tooltip，Tooltip target id 全局唯一；
· next_step_alert 含说明文字与跳转链接；
· GLOSSARY 覆盖 14 个核心术语（含两个子模型词条）；
· model_label / model_blurb / humanize_model_text（re-export 自 src/model_display）：
  已知键换显示名、未知键原样、「键=小数」转百分比、pred_ 列名不误伤。
"""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from webapp import components as wc


def _walk(node):
    """深度遍历组件树，产出所有节点（含嵌套 children）。"""
    yield node
    for child in getattr(node, "children", None) or []:
        if isinstance(child, (list, tuple)):
            for c in child:
                yield from _walk(c)
        elif hasattr(child, "children") or isinstance(child, html.Div):
            yield from _walk(child)


def test_flow_steps_contract():
    assert [label for label, _ in wc.FLOW_STEPS] == [
        "因子研发",
        "模型训练与评估",
        "回测验证",
        "交易执行",
    ]
    # 「模型训练与评估」与「回测验证」都指向 /models 是有意的
    assert wc.FLOW_STEPS[1][1] == wc.FLOW_STEPS[2][1] == "/models"
    assert wc.FLOW_STEPS[0][1] == "/factors"
    assert wc.FLOW_STEPS[3][1] == "/"


def test_page_header_with_flow_step():
    hdr = wc.page_header("因子工作台", "副标题文案", flow_step=0)
    nodes = list(_walk(hdr))
    title = next(n for n in nodes if getattr(n, "className", None) == "page-title")
    sub = next(n for n in nodes if getattr(n, "className", None) == "page-sub")
    assert title.children == "因子工作台"
    assert sub.children == "副标题文案"

    steps = [n for n in nodes if "flow-step" in (getattr(n, "className", "") or "").split()]
    # 4 步：当前步是 Span（实心高亮），其余 3 步是可点击 Link
    cur = [n for n in steps if "flow-step-cur" in n.className.split()]
    links = [n for n in steps if isinstance(n, dcc.Link)]
    assert len(steps) == 4
    assert len(cur) == 1 and cur[0].children == "因子研发"
    assert len(links) == 3
    assert {link.href for link in links} == {"/models", "/"}


def test_page_header_without_flow_step():
    hdr = wc.page_header("会话详情", "子标题", flow_step=None)
    nodes = list(_walk(hdr))
    assert not any("flow-steps" in (getattr(n, "className", "") or "") for n in nodes)


def test_flow_breadcrumb_highlights_current():
    bc = wc.flow_breadcrumb(2)
    nodes = list(_walk(bc))
    cur = [n for n in nodes if "flow-step-cur" in (getattr(n, "className", "") or "").split()]
    assert len(cur) == 1 and cur[0].children == "回测验证"


def test_term_span_plus_tooltip_unique_ids():
    t = wc.term("RankIC", id_prefix="factors-table")
    assert isinstance(t, html.Span)
    span = next(n for n in t.children if isinstance(n, html.Span))
    tip = next(n for n in t.children if isinstance(n, dbc.Tooltip))
    assert span.className == "gloss-term"
    assert span.children == "RankIC"
    assert tip.children == wc.GLOSSARY["RankIC"]
    assert tip.target == span.id

    # id 全局唯一：不同 id_prefix / 不同术语产出不同 target
    t2 = wc.term("RankIC", id_prefix="models-header")
    t3 = wc.term("ICIR", id_prefix="factors-table")
    ids = {next(n for n in x.children if isinstance(n, html.Span)).id for x in (t, t2, t3)}
    assert len(ids) == 3


def test_term_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        wc.term("不存在的术语", id_prefix="x")


def test_terms_convenience_form():
    t = wc.terms("RankIC", "ICIR", id_prefix="hdr")
    assert isinstance(t, html.Span)
    spans = [n for n in t.children if isinstance(n, html.Span)]
    # 两个 term 各产出一个 .gloss-term 内层 Span
    gloss = [n for n in _walk(t) if getattr(n, "className", None) == "gloss-term"]
    assert [g.children for g in gloss] == ["RankIC", "ICIR"]
    assert spans  # 结构非空


def test_next_step_alert_contains_link():
    alert = wc.next_step_alert("去模型实验室看评估结果", "/models")
    assert isinstance(alert, dbc.Alert)
    assert alert.color == "info"
    assert "next-step-alert" in alert.className.split()
    nodes = list(_walk(alert))
    link = next(n for n in nodes if isinstance(n, dcc.Link))
    assert link.href == "/models"
    assert link.children == "前往 →"
    assert any(getattr(n, "children", None) == "去模型实验室看评估结果" for n in nodes)


def test_glossary_covers_core_terms():
    expected = {
        "RankIC",
        "ICIR",
        "PredIC",
        "IC_t",
        "champion",
        "staging",
        "门控",
        "噪声带",
        "IC 定权",
        "walk-forward",
        "embargo",
        "多空收益",
        "AI 树模型",
        "因子加权模型",
    }
    assert expected <= set(wc.GLOSSARY)
    for k in expected:
        assert wc.GLOSSARY[k].strip()


# --- 子模型显示名（src/model_display，经 components re-export）--------------------
def test_model_label_known_and_unknown():
    assert wc.model_label("gbm_l2") == "AI 树模型"
    assert wc.model_label("ic_linear") == "因子加权模型"
    assert wc.model_label("attn") == "注意力模型"
    # 未知键原样返回，不丢信息
    assert wc.model_label("some_future_model") == "some_future_model"
    # 一句话解释：已知键非空、未知键空串
    assert "LightGBM" in wc.model_blurb("gbm_l2")
    assert wc.model_blurb("ic_linear").strip()
    assert wc.model_blurb("unknown") == ""


def test_humanize_model_text_weight_and_bare_key():
    # 「键=小数」→「显示名 百分比」
    assert (
        wc.humanize_model_text("已预测 5,476 只 · 权重 gbm_l2=0.50 · ic_linear=0.50")
        == "已预测 5,476 只 · 权重 AI 树模型 50% · 因子加权模型 50%"
    )
    # 裸键整词替换
    assert (
        wc.humanize_model_text("ensemble [gbm_l2,ic_linear]") == "ensemble [AI 树模型,因子加权模型]"
    )
    # pred_ 列名（内部标识符）不误伤
    assert wc.humanize_model_text("pred_gbm_l2") == "pred_gbm_l2"
    # 无键文本透传；非字符串/空值原样
    assert wc.humanize_model_text("没有模型键的文本") == "没有模型键的文本"
    assert wc.humanize_model_text("") == ""
    assert wc.humanize_model_text(None) is None
