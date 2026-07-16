"""会话流水线的执行上下文与 step report 构造辅助。

每个模块(步骤)是一个函数 `fn(ctx) -> dict`，返回一个**自描述的 step report**：
前端用一个通用渲染器(webapp/render.py)即可展示，无需为每种步骤写专门页面。

report 结构（落盘为 sessions/<id>/steps/NN_<step>.json）::

    {
      "step": "data_import", "title": "数据导入", "status": "done",
      "elapsed": 12.3, "summary": "一句话摘要",
      "kpis":   [{"label","value","hint"}...],
      "tables": [{"title","columns":[...],"rows":[[...]],"max_rows":50}...],
      "figures":[<figure spec>...],   # 见下方 fig_* 构造器
      "notes":  ["⚠️ ..."]
    }

figure spec 的 type ∈ line|bar|scatter|histogram|heatmap|area_stacked，
所有键都是 JSON 可序列化的纯 Python（list/float/str），便于落盘与回读。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def _jsonable(x: Any) -> Any:
    """把 numpy / pandas 标量与数组转成纯 Python，保证可 json.dump。"""
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, float):
        return None if (x != x or x in (float("inf"), float("-inf"))) else x
    return x


@dataclass
class StepContext:
    """跨步骤共享的执行上下文。

    session_dir : 本会话目录（artifacts/ 落盘在此）
    params      : 会话参数 dict（已在进程启动前写入 os.environ → config）
    shared      : 内存对象在步骤间传递（如 panel / factors / oos），不落盘
    on_progress : 可选 callback，签名 cb(dict)。run_session 会装上 ProgressBridge，
                  步骤函数可通过 ctx.report_progress() 主动汇报阶段内进度。
    step_started_at : 当前步骤起始 wall time（run_session 在进入步骤时填充）
    """
    session_dir: str
    params: dict = field(default_factory=dict)
    shared: dict = field(default_factory=dict)
    on_progress: Callable[[dict], None] | None = None
    step_started_at: float = 0.0

    @property
    def artifacts_dir(self) -> str:
        d = os.path.join(self.session_dir, "artifacts")
        os.makedirs(d, exist_ok=True)
        return d

    def artifact_path(self, name: str) -> str:
        return os.path.join(self.artifacts_dir, name)

    def report_progress(self, phase: str, current: int, total: int,
                        message: str = "", **extras) -> None:
        """上报阶段内进度。无 callback 时 no-op（保持单元测试兼容）。

        payload: {phase, current, total, pct, elapsed_sec, eta_sec, extras, message}
        """
        if self.on_progress is None:
            return
        total = max(int(total), 1)
        current = max(0, min(int(current), total))
        pct = current / total
        elapsed = max(time.time() - (self.step_started_at or time.time()), 0.0)
        eta = (elapsed / current * (total - current)) if current > 0 else 0.0
        try:
            self.on_progress({
                "phase": str(phase),
                "current": current,
                "total": total,
                "pct": round(pct, 4),
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1),
                "message": str(message),
                "extras": {k: _jsonable(v) for k, v in extras.items()},
            })
        except Exception:
            # progress 永远不能影响主流程
            pass


# --- report builders -------------------------------------------------------
def kpi(label: str, value: Any, hint: str = "") -> dict:
    """一个指标卡。value 任意（会被格式化为字符串展示）。"""
    return {"label": label, "value": value, "hint": hint}


def table(title: str, df, max_rows: int = 50) -> dict:
    """把 DataFrame 转成表格 spec（截断到 max_rows 行）。"""
    cols = [str(c) for c in df.columns]
    rows = _jsonable(df.head(max_rows).to_numpy())
    return {"title": title, "columns": cols, "rows": rows,
            "n_total": int(len(df)), "max_rows": max_rows}


def fig_line(title: str, x, series: list[dict], yaxis: str = "",
             xaxis: str = "", x_is_date: bool = False) -> dict:
    """折线图。series=[{"name","y","dash"?,"width"?,"color"?}...]。"""
    return {"type": "line", "title": title, "x": _jsonable(x),
            "series": [{**s, "y": _jsonable(s["y"])} for s in series],
            "yaxis": yaxis, "xaxis": xaxis, "x_is_date": x_is_date}


def fig_bar(title: str, x, y, yaxis: str = "", xaxis: str = "",
            color: str | None = None, diverging: bool = False) -> dict:
    """柱状图（单序列）。diverging=True 时按正负着色。"""
    return {"type": "bar", "title": title, "x": _jsonable(x), "y": _jsonable(y),
            "yaxis": yaxis, "xaxis": xaxis, "color": color, "diverging": diverging}


def fig_scatter(title: str, x, y, yaxis: str = "", xaxis: str = "",
                text=None, trendline: bool = False) -> dict:
    """散点图。trendline=True 叠加一阶拟合线。"""
    return {"type": "scatter", "title": title, "x": _jsonable(x), "y": _jsonable(y),
            "yaxis": yaxis, "xaxis": xaxis,
            "text": _jsonable(text) if text is not None else None,
            "trendline": trendline}


def fig_hist(title: str, values, nbins: int = 60, xaxis: str = "") -> dict:
    """直方图。**服务端预分箱**：只存 bin 中心 + 计数，绝不把原始数组塞进 spec。

    全市场标签有数百万行，直接 _jsonable(values) 会把整列写进 JSON（曾导致单个
    step report 162MB、前端静默卡死）。np.histogram 在服务端算好，spec 只剩 ~nbins
    个点，渲染端画成柱状即可。
    """
    import numpy as np
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"type": "histogram", "title": title, "centers": [], "counts": [],
                "xaxis": xaxis}
    # 1/99 分位裁剪，避免极端离群值把所有质量挤进一个箱
    lo, hi = np.quantile(v, [0.01, 0.99])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max() + 1e-9)
    counts, edges = np.histogram(np.clip(v, lo, hi), bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"type": "histogram", "title": title,
            "centers": _jsonable(centers), "counts": _jsonable(counts),
            "xaxis": xaxis}


def fig_area_stacked(title: str, x, series: list[dict], yaxis: str = "",
                     x_is_date: bool = False) -> dict:
    """堆叠面积图（如成本拆解）。series=[{"name","y","color"?}...]。"""
    return {"type": "area_stacked", "title": title, "x": _jsonable(x),
            "series": [{**s, "y": _jsonable(s["y"])} for s in series],
            "yaxis": yaxis, "x_is_date": x_is_date}


def fig_heatmap(title: str, z, x=None, y=None, xaxis: str = "",
                yaxis: str = "") -> dict:
    """热力图（红-白-绿发散）。"""
    return {"type": "heatmap", "title": title, "z": _jsonable(z),
            "x": _jsonable(x) if x is not None else None,
            "y": _jsonable(y) if y is not None else None,
            "xaxis": xaxis, "yaxis": yaxis}


def report(step: str, title: str, *, summary: str = "", kpis=None,
           tables=None, figures=None, notes=None) -> dict:
    """组装一个完整 step report（status/elapsed 由 runner 填充）。"""
    return {
        "step": step, "title": title, "summary": summary,
        "kpis": _jsonable(kpis or []),
        "tables": tables or [],
        "figures": figures or [],
        "notes": notes or [],
    }
