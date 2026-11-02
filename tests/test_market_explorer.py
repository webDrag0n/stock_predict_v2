"""行情页单元测试：

· ?symbol= 深链解析（_symbol_from_search 纯函数）；
· 页面头统一组件（page_header，flow_step=None 无面包屑）；
· 无缓存空态 Alert 含数据导入引导；
· register 挂出 me-symbol 初始化回调（url search → 下拉 value）。
"""

from __future__ import annotations

from dash import Dash
from dash.development.base_component import Component

from webapp.pages import market_explorer as me


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


def _walk(node):
    yield node
    for child in getattr(node, "children", None) or []:
        if isinstance(child, (list, tuple)):
            for c in child:
                yield from _walk(c)
        elif hasattr(child, "children"):
            yield from _walk(child)


def test_symbol_from_search(monkeypatch):
    monkeypatch.setattr(me, "_SYMBOLS", ["000001.SZ", "600519.SH"])
    assert me._symbol_from_search("?symbol=000001.SZ") == "000001.SZ"
    assert me._symbol_from_search("?symbol=600519.SH&x=1") == "600519.SH"
    assert me._symbol_from_search("symbol=000001.SZ") == "000001.SZ"
    # 无参数 / 空串 / 不在股票池 → None（保持下拉默认）
    assert me._symbol_from_search(None) is None
    assert me._symbol_from_search("") is None
    assert me._symbol_from_search("?x=1") is None
    assert me._symbol_from_search("?symbol=999999.XX") is None


def test_layout_uses_page_header():
    comp = me.layout()
    assert isinstance(comp, Component)
    nodes = list(_walk(comp))
    title = [n for n in nodes if getattr(n, "className", None) == "page-title"]
    assert title and title[0].children == "行情浏览器 Market Explorer"
    # flow_step=None → 不渲染流程面包屑
    assert not any("flow-steps" in (getattr(n, "className", "") or "") for n in nodes)


def test_empty_state_guides_data_import(monkeypatch):
    monkeypatch.setattr(me, "_SYMBOLS", [])
    txt = " ".join(_texts(me.layout()))
    assert "未找到行情缓存" in txt
    assert "数据导入" in txt
    assert "daily_run.sh" not in txt  # 主路径纯网页引导，无 CLI 文案


def test_register_wires_symbol_init_callback():
    app = Dash(__name__, suppress_callback_exceptions=True)
    me.register(app)
    # url search → me-symbol value 的初始化回调已挂
    assert "me-symbol.value" in app.callback_map
    assert "me-candle.figure" in app.callback_map
