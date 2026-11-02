"""会话详情页（webapp/pages/session_detail.py）单元测试。

覆盖：_back_target 全模式映射（含历史模式容错）、_sid_from_path 双前缀解析、
failed 布局的重试按钮（历史模式禁用 + 下线提示）、done 布局按模式的
next_step_alert 指引、retry_session 以相同 mode/label/params 重建会话
（monkeypatch scheduler/sessions，不起子进程）。
"""

from __future__ import annotations

import pytest
from dash.development.base_component import Component

from pipeline import registry
from webapp.pages import session_detail as sd


# ---------------------------------------------------------------------------
# 组件树工具
# ---------------------------------------------------------------------------
def _walk(comp):
    """递归展开组件树，yield 每个 Component。"""
    if isinstance(comp, Component):
        yield comp
        yield from _walk(getattr(comp, "children", None))
    elif isinstance(comp, (list, tuple)):
        for c in comp:
            yield from _walk(c)


def _find_by_id(comp, cid):
    for c in _walk(comp):
        if getattr(c, "id", None) == cid:
            return c
    return None


def _find_by_class(comp, cls):
    return [c for c in _walk(comp) if cls in str(getattr(c, "className", "") or "")]


# ---------------------------------------------------------------------------
# _back_target / _sid_from_path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode,expected",
    [
        ("nextday", ("/", "交易台")),
        ("factoreval", ("/factors", "因子工作台")),
        ("walkforward", ("/models", "模型实验室")),
        ("train", ("/models", "模型实验室")),
        ("modeleval", ("/models", "模型实验室")),
        # 历史/未知模式容错 → 模型实验室
        ("backtest", ("/models", "模型实验室")),
        ("robustness", ("/models", "模型实验室")),
        ("no_such_mode", ("/models", "模型实验室")),
        (None, ("/models", "模型实验室")),
    ],
)
def test_back_target(mode, expected):
    assert sd._back_target(mode) == expected


def test_sid_from_path():
    assert sd._sid_from_path("/runs/walkforward_20261026_120000") == "walkforward_20261026_120000"
    assert sd._sid_from_path("/session/backtest_20240101_120000") == "backtest_20240101_120000"
    assert sd._sid_from_path("/runs/abc/") == "abc"
    assert sd._sid_from_path("/models") is None
    assert sd._sid_from_path(None) is None


# ---------------------------------------------------------------------------
# 布局渲染（打桩 sessions 读写）
# ---------------------------------------------------------------------------
def _meta(mode: str, status: str, **over) -> dict:
    m = {
        "id": f"{mode}_20261026_120000",
        "mode": mode,
        "label": "测试会话",
        "params": {"topk": 0.15, "train_window": 500},
        "status": status,
        "steps": [],
    }
    m.update(over)
    return m


@pytest.fixture
def stub_sessions(monkeypatch):
    """把 sessions 的读侧打桩成内存 meta；tick/step_reports/run_log 无副作用。"""
    store = {"meta": None}
    monkeypatch.setattr(sd.sessions, "tick", lambda: None)
    monkeypatch.setattr(sd.sessions, "get", lambda sid: store["meta"])
    monkeypatch.setattr(sd.sessions, "step_reports", lambda sid: [])
    monkeypatch.setattr(sd.sessions, "run_log", lambda sid, n=120: "")
    return store


def _render(store, meta):
    store["meta"] = meta
    return sd._render_detail(meta["id"])


def test_failed_layout_has_retry_button(stub_sessions):
    back, header, body, stop = _render(
        stub_sessions,
        _meta(
            "modeleval",
            "failed",
            failure={"exc_type": "RuntimeError", "exc_msg": "boom", "traceback": "tb"},
        ),
    )
    assert stop is True
    btn = _find_by_id(body, "sd-retry")
    assert btn is not None and btn.disabled is False
    assert "以相同参数重试" in str(btn.children)
    # 返回链接按模式指向来源页
    assert back.href == "/models" and "模型实验室" in str(back.children)


def test_failed_layout_legacy_mode_disables_retry(stub_sessions):
    back, header, body, stop = _render(
        stub_sessions,
        _meta(
            "backtest",
            "failed",
            failure={"exc_type": "RuntimeError", "exc_msg": "boom"},
        ),
    )
    btn = _find_by_id(body, "sd-retry")
    assert btn is not None and btn.disabled is True
    assert "该模式已下线" in btn.title
    assert "模式已下线" in str(body)
    # 历史模式容错：返回模型实验室
    assert back.href == "/models"


@pytest.mark.parametrize(
    "mode,href,text_frag",
    [
        ("factoreval", "/factors", "RankIC/ICIR"),
        ("train", "/models", "staging"),
        ("walkforward", "/models", "验证时间线"),
        ("modeleval", "/models", "Walk-Forward"),
        ("nextday", "/", "回交易台执行"),
    ],
)
def test_done_layout_has_next_step_alert(stub_sessions, mode, href, text_frag):
    back, header, body, stop = _render(stub_sessions, _meta(mode, "done"))
    assert stop is True
    alerts = _find_by_class(body, "next-step-alert")
    assert len(alerts) == 1
    alert = alerts[0]
    assert text_frag in str(alert)
    links = [c for c in _walk(alert) if getattr(c, "href", None) == href]
    assert links, f"缺少指向 {href} 的跳转链接"


def test_done_layout_legacy_mode_no_next_step(stub_sessions):
    for mode in ("backtest", "robustness"):
        _, _, body, _ = _render(stub_sessions, _meta(mode, "done"))
        assert _find_by_class(body, "next-step-alert") == []


# ---------------------------------------------------------------------------
# retry_session
# ---------------------------------------------------------------------------
def test_retry_session_recreates_with_same_params(monkeypatch):
    calls = {}
    meta = _meta("train", "failed", label="我的训练")
    monkeypatch.setattr(sd.sessions, "get", lambda sid: meta)
    monkeypatch.setattr(
        sd.sessions,
        "create",
        lambda mode, label, params: calls.update(mode=mode, label=label, params=params)
        or "train_20261026_130000",
    )
    monkeypatch.setattr(sd.sessions, "tick", lambda: calls.setdefault("ticked", True))

    new_sid = sd.retry_session(meta["id"])
    assert new_sid == "train_20261026_130000"
    assert calls["mode"] == "train"
    assert calls["label"] == "我的训练"
    assert calls["params"] == {"topk": 0.15, "train_window": 500}
    assert calls["ticked"] is True


def test_retry_session_legacy_mode_refused(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("历史模式不应创建会话")

    monkeypatch.setattr(sd.sessions, "get", lambda sid: _meta("robustness", "failed"))
    monkeypatch.setattr(sd.sessions, "create", _boom)
    assert sd.retry_session("robustness_20240101_000000") is None


def test_retry_session_bad_inputs(monkeypatch):
    monkeypatch.setattr(sd.sessions, "get", lambda sid: None)
    assert sd.retry_session("ghost_session") is None  # 会话不存在
    assert sd.retry_session("../etc") is None  # 非法 sid
    assert sd.retry_session(None) is None


# ---------------------------------------------------------------------------
# 布局与回调注册冒烟
# ---------------------------------------------------------------------------
def test_layout_has_back_and_redirect_mounts():
    lay = sd.layout()
    assert _find_by_id(lay, "sd-back") is not None
    loc = _find_by_id(lay, "sd-redirect")
    assert loc is not None and loc.refresh is True
    # 写死的首页链接已移除（改由 sd-back 按模式渲染）
    assert _find_by_id(lay, "sd-header") is not None


def test_register_smoke():
    import dash
    from dash import dcc, html

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page-content")])
    sd.register(app)
    keys = list(
        app.callback_map
    )  # 多输出回调的 key 形如 "..sd-back.children...sd-header.children.."
    assert any("sd-back.children" in k for k in keys)
    assert any("sd-redirect.pathname" in k for k in keys)
    assert any("sd-poll.disabled" in k for k in keys)


def test_registered_modes_consistent():
    """_NEXT_STEP 覆盖的正是当前全部 5 个线上模式（历史模式故意不在内）。"""
    assert set(sd._NEXT_STEP) == set(registry.STEPS)
    assert "backtest" not in sd._NEXT_STEP and "robustness" not in sd._NEXT_STEP
