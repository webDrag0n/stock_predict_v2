"""figure_from_spec 结构断言：band 置信带 / 逐系列 opacity / 分组柱状。

纯 Plotly 结构测试，不启动 Dash。
"""
import json

import plotly.graph_objects as go

from pipeline.context import fig_area_stacked, fig_bar, fig_line
from webapp.render import figure_from_spec


def test_line_with_band():
    spec = fig_line("t", [1, 2, 3], [
        {"name": "mid", "y": [1.0, 2.0, 3.0], "color": "#3b82f6",
         "band": {"upper": [2.0, 3.0, 4.0], "lower": [0.0, 1.0, 2.0]}},
    ])
    json.dumps(spec)  # band 也要可 JSON 序列化（落盘要求）
    fig = figure_from_spec(spec)
    assert len(fig.data) == 3
    upper, lower, main = fig.data
    assert upper.line.width == 0
    assert upper.hoverinfo == "skip"
    assert upper.showlegend is False
    assert lower.fill == "tonexty"
    assert lower.hoverinfo == "skip"
    assert lower.showlegend is False
    assert lower.fillcolor == "rgba(59,130,246,0.15)"
    assert main.name == "mid"
    assert main.mode == "lines"


def test_line_band_without_color_falls_back():
    spec = fig_line("t", [1, 2], [
        {"name": "m", "y": [1, 2],
         "band": {"upper": [2, 3], "lower": [0, 1]}},
    ])
    fig = figure_from_spec(spec)
    assert len(fig.data) == 3
    assert fig.data[1].fillcolor == "rgba(100,116,139,0.15)"


def test_line_without_band_unchanged():
    spec = fig_line("t", [1, 2], [{"name": "a", "y": [1, 2]}])
    fig = figure_from_spec(spec)
    assert len(fig.data) == 1
    assert fig.data[0].showlegend is None  # 默认进 legend
    assert fig.data[0].opacity is None     # 默认不设 opacity


def test_line_opacity():
    spec = fig_line("t", [1, 2], [{"name": "a", "y": [1, 2], "opacity": 0.4}])
    fig = figure_from_spec(spec)
    assert fig.data[0].opacity == 0.4


def test_area_stacked_opacity():
    spec = fig_area_stacked("t", [1, 2], [
        {"name": "a", "y": [1, 2], "opacity": 0.5},
        {"name": "b", "y": [3, 4]},
    ])
    fig = figure_from_spec(spec)
    assert fig.data[0].opacity == 0.5
    assert fig.data[1].opacity is None


def test_bar_grouped_multi_series():
    spec = fig_bar("t", ["a", "b"], series=[
        {"name": "s1", "y": [1, 2], "color": "#111111", "opacity": 0.9},
        {"name": "s2", "y": [3, 4]},
    ])
    fig = figure_from_spec(spec)
    assert len(fig.data) == 2
    assert all(isinstance(tr, go.Bar) for tr in fig.data)
    assert fig.layout.barmode == "group"
    assert fig.data[0].name == "s1"
    assert fig.data[0].opacity == 0.9
    assert fig.data[1].opacity is None


def test_bar_legacy_single_series_unchanged():
    spec = fig_bar("t", ["a", "b"], [1, -2], yaxis="y", diverging=True)
    fig = figure_from_spec(spec)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)
    assert fig.layout.barmode != "group"
    assert fig.layout.yaxis.title.text == "y"
