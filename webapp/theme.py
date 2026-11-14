"""Shared light-theme Plotly template, color palette, and chart helpers.

Every figure in the app pulls its styling from here so the look is consistent and
professionally clean — a single light template, one palette, common layout knobs.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- palette（与 assets/custom.css 设计令牌同源）----------------------------
PALETTE = {
    "strat": "#067647",  # strategy green
    "bench": "#ea7a08",  # benchmark orange
    "alpha": "#6941c6",  # alpha purple
    "loss": "#d92d20",  # drawdown / negative red
    "impact": "#b54708",  # market-impact cost
    "commission": "#175cd3",  # commission cost
    "up": "#12b76a",  # candle up
    "down": "#f04438",  # candle down
    "faint": "#98a2b3",  # per-seed faint
    "grid": "#eef0f3",
    "ink": "#101828",  # primary text
    "muted": "#667085",
    "panel": "#ffffff",
    "bg": "#f5f6f8",
}

# diverging scale for heatmaps (red→white→green)
DIVERGING = [[0.0, "#d92d20"], [0.5, "#ffffff"], [1.0, "#067647"]]


def _make_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12.5, color=PALETTE["ink"]),
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        colorway=[
            PALETTE["strat"],
            PALETTE["bench"],
            PALETTE["alpha"],
            PALETTE["commission"],
            PALETTE["impact"],
            PALETTE["loss"],
        ],
        xaxis=dict(
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            linecolor="#d0d5dd",
            ticks="outside",
            tickcolor="#d0d5dd",
            tickfont=dict(size=11.5, color=PALETTE["muted"]),
        ),
        yaxis=dict(
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            linecolor="#d0d5dd",
            ticks="outside",
            tickcolor="#d0d5dd",
            tickfont=dict(size=11.5, color=PALETTE["muted"]),
        ),
        margin=dict(l=56, r=20, t=44, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#101828", font_size=12, font_color="#ffffff", bordercolor="#101828"
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor=PALETTE["grid"],
            borderwidth=1,
            font=dict(size=11),
        ),
        title=dict(font=dict(size=13.5, color=PALETTE["ink"]), x=0.01, xanchor="left"),
    )
    return t


pio.templates["quantlight"] = _make_template()
pio.templates.default = "quantlight"

TEMPLATE = "quantlight"


def base_fig(title: str = "", height: int | None = None) -> go.Figure:
    """A blank figure pre-styled with the light template."""
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title=title)
    if height:
        fig.update_layout(height=height)
    return fig


def empty_fig(msg: str = "No data", height: int = 300) -> go.Figure:
    """Placeholder figure with a centered message (for empty/error states)."""
    fig = base_fig(height=height)
    fig.add_annotation(
        text=msg,
        showarrow=False,
        font=dict(size=14, color=PALETTE["muted"]),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def money(v: float) -> str:
    a = abs(v)
    s = "-" if v < 0 else ""
    if a >= 1e8:
        return f"{s}¥{a/1e8:.2f}亿"
    if a >= 1e4:
        return f"{s}¥{a/1e4:.1f}万"
    return f"{s}¥{a:.0f}"
