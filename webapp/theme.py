"""Shared light-theme Plotly template, color palette, and chart helpers.

Every figure in the app pulls its styling from here so the look is consistent and
professionally clean — a single light template, one palette, common layout knobs.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- palette ---------------------------------------------------------------
PALETTE = {
    "strat":      "#1a7f37",   # strategy green
    "bench":      "#ff7f0e",   # benchmark orange
    "alpha":      "#7b3fb5",   # alpha purple
    "loss":       "#d62728",   # drawdown / negative red
    "impact":     "#c2410c",   # market-impact cost
    "commission": "#2563eb",   # commission cost
    "up":         "#16a34a",   # candle up
    "down":       "#dc2626",   # candle down
    "faint":      "#9ca3af",   # per-seed faint
    "grid":       "#eceff3",
    "ink":        "#1f2937",   # primary text
    "muted":      "#6b7280",
    "panel":      "#ffffff",
    "bg":         "#f7f8fa",
}

# diverging scale for heatmaps (red→white→green)
DIVERGING = [[0.0, "#d62728"], [0.5, "#ffffff"], [1.0, "#1a7f37"]]


def _make_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13,
                  color=PALETTE["ink"]),
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        colorway=[PALETTE["strat"], PALETTE["bench"], PALETTE["alpha"],
                  PALETTE["commission"], PALETTE["impact"], PALETTE["loss"]],
        xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"],
                   linecolor="#d1d5db", ticks="outside", tickcolor="#d1d5db"),
        yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"],
                   linecolor="#d1d5db", ticks="outside", tickcolor="#d1d5db"),
        margin=dict(l=60, r=24, t=48, b=44),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12,
                        bordercolor=PALETTE["grid"]),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=PALETTE["grid"],
                    borderwidth=1, font=dict(size=11)),
        title=dict(font=dict(size=15, color=PALETTE["ink"]), x=0.01, xanchor="left"),
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
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=PALETTE["muted"]),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def money(v: float) -> str:
    a = abs(v); s = "-" if v < 0 else ""
    if a >= 1e8:
        return f"{s}¥{a/1e8:.2f}亿"
    if a >= 1e4:
        return f"{s}¥{a/1e4:.1f}万"
    return f"{s}¥{a:.0f}"
