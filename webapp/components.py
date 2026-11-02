"""共享 UI 组件：页面头、流程面包屑、术语 glossary tooltip、「下一步」指引条。

所有页面（webapp/pages/*）统一从这里取外壳组件，保证视觉与交互一致。
样式类定义在 webapp/assets/custom.css（.flow-steps/.gloss-term/.next-step-alert 等）。

公开接口（其他页面只用这些，不要直接拼 className）：

- FLOW_STEPS                工作流 4 步定义 [(label, href), ...]
- flow_breadcrumb(current)  流程面包屑小条；current 是 FLOW_STEPS 索引
- page_header(title, sub, flow_step=None)  统一页面头（标题+副标题+面包屑）
- GLOSSARY                  核心术语表 dict[str, str]
- term(text, *, id_prefix)  单个术语：虚线下划线 + 悬停 Tooltip
- terms(*texts, id_prefix)  多个术语并排（顿号分隔）
- next_step_alert(text, href, link_text="前往 →")  「下一步去哪」指引条
- model_label(key)          模型键 → 中文显示名（re-export 自 src/model_display）
- model_blurb(key)          模型键 → 一句话解释
- humanize_model_text(text) 文本里的裸模型键 → 显示名（历史会话摘要也能洗干净）
"""

from __future__ import annotations

import hashlib

from dash import dcc, html
import dash_bootstrap_components as dbc

# 模型键 → 显示名的唯一权威定义在 src/model_display（pipeline 步骤摘要也用它）；
# 这里 re-export，前端各页统一 from ..components import，不直接依赖 src 路径。
from model_display import MODEL_DISPLAY, humanize_model_text, model_blurb, model_label

__all__ = [
    "FLOW_STEPS",
    "flow_breadcrumb",
    "page_header",
    "GLOSSARY",
    "term",
    "terms",
    "next_step_alert",
    "MODEL_DISPLAY",
    "model_label",
    "model_blurb",
    "humanize_model_text",
]

# --- 工作流面包屑 --------------------------------------------------------------
# 主研究工作流：因子研发 → 模型训练与评估 → 回测验证 → 交易执行。
# 「模型训练与评估」与「回测验证」都指向 /models 是有意的（回测在模型实验室内完成）。
FLOW_STEPS: list[tuple[str, str]] = [
    ("因子研发", "/factors"),
    ("模型训练与评估", "/models"),
    ("回测验证", "/models"),
    ("交易执行", "/"),
]


def flow_breadcrumb(current: int) -> html.Div:
    """流程面包屑小条：4 步，当前步实心高亮，其余灰字可点击跳转。

    current: FLOW_STEPS 的索引（0..3）。
    """
    children: list = []
    for i, (label, href) in enumerate(FLOW_STEPS):
        if i:
            children.append(html.Span("›", className="flow-step-sep"))
        if i == current:
            children.append(html.Span(label, className="flow-step flow-step-cur"))
        else:
            children.append(dcc.Link(label, href=href, className="flow-step"))
    return html.Div(children, className="flow-steps")


def page_header(title: str, sub: str, flow_step: int | None = None) -> html.Div:
    """统一页面头：.page-title 标题 + .page-sub 副标题 + 可选流程面包屑。

    flow_step 传 FLOW_STEPS 的索引显示面包屑并高亮当前步；传 None 不渲染面包屑
    （如会话详情等不属于主工作流的页面）。
    """
    children: list = [
        html.H4(title, className="page-title"),
        html.P(sub, className="page-sub"),
    ]
    if flow_step is not None:
        children.append(flow_breadcrumb(flow_step))
    return html.Div(children)


# --- 术语表 -------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    "RankIC": "预测排序与真实收益排序的截面相关系数（Spearman）。越接近 1 说明选股排序越准，0 附近就是瞎猜。",
    "ICIR": "RankIC 的均值除以标准差，相当于 IC 的「夏普」。衡量信号稳不稳，而不只是准不准。",
    "PredIC": "预测值与未来实际收益的逐日相关系数，回测/评估报告里衡量预测质量的主指标。",
    "IC_t": "第 t 天算出的单日 IC 值，是 IC 时间序列的一项；平均后才是报告里的 IC。",
    "champion": "当前线上真正出信号的模型，放在 models/champion/。订单永远只出自 champion。",
    "staging": "候选模型的「候审区」models/staging/。新训练的模型先落这里，通过门控对比才晋升 champion。",
    "门控": "候选模型晋升 champion 前的闸门：回测 Sharpe 更高且回撤不更差才放行，否则继续用老模型。",
    "噪声带": "IC 等指标在纯噪声下也会随机波动的区间。指标没跑出噪声带，就不能算有信号。",
    "IC 定权": "集成模型按各子模型最近的滚动 IC 分配权重：最近谁准谁话语权大。",
    "AI 树模型": "LightGBM 梯度提升树：非线性 AI 模型，从 300+ 因子自动学习规律。内部键名 gbm_l2。",
    "因子加权模型": "按各因子近期预测准确率（IC）加权的线性打分。结构简单、抗过拟合，是 AI 树模型的稳定搭档。内部键名 ic_linear。",
    "walk-forward": "滚动回测：用过去一段窗口训练、紧接着的窗口测试，然后整体向前滚动，全程不偷看未来。",
    "embargo": "训练段和测试段之间的隔离带。因为标签要看未来 N 天收益，不留隔离带就等于偷看答案。",
    "多空收益": "做多最看好的一组、做空最不看好的组，两头收益之差。衡量排序能力，不代表实盘可执行。",
}


def _tip_id(id_prefix: str, text: str) -> str:
    """Tooltip target 的全局唯一 id：id_prefix + 术语文本的短 hash。

    约定：id_prefix 由调用方保证在页面内唯一（建议带页面名与位置，如
    "factors-table"）；同一 id_prefix + 同一术语在同一页面渲染两次会产生
    重复 id，dbc.Tooltip 会挂不上。
    """
    h = hashlib.md5(f"{id_prefix}:{text}".encode("utf-8")).hexdigest()[:8]
    return f"gloss-{id_prefix}-{h}"


def term(text: str, *, id_prefix: str) -> html.Span:
    """单个术语：虚线下划线（.gloss-term）+ 悬停显示 GLOSSARY 释义的 Tooltip。

    text 必须在 GLOSSARY 里，否则 KeyError——故意不放行，防止页面上出现
    没有释义的「假术语」。Tooltip 由 dbc 挂到 body，无需手动管理位置。
    """
    tip_id = _tip_id(id_prefix, text)
    return html.Span(
        [
            html.Span(text, className="gloss-term", id=tip_id),
            dbc.Tooltip(GLOSSARY[text], target=tip_id, placement="top"),
        ]
    )


def terms(*texts: str, id_prefix: str) -> html.Span:
    """多个术语并排的便捷形式（顿号分隔），等价于逐个调用 term()。"""
    children: list = []
    for i, t in enumerate(texts):
        if i:
            children.append(html.Span("、", className="gloss-sep"))
        children.append(term(t, id_prefix=f"{id_prefix}-{i}"))
    return html.Span(children)


# --- 下一步指引条 ---------------------------------------------------------------
def next_step_alert(text: str, href: str, link_text: str = "前往 →") -> dbc.Alert:
    """浅蓝 info 底的「下一步去哪」指引条：左侧图标 + 说明文字 + 右侧跳转链接。"""
    return dbc.Alert(
        [
            html.Span("➜", className="next-step-icon"),
            html.Span(text, className="next-step-text"),
            dcc.Link(link_text, href=href, className="next-step-link"),
        ],
        color="info",
        className="next-step-alert",
    )
