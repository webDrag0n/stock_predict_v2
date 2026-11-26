"""以 run registry 登记的方式运行 pipeline 模式（train / modeleval / factoreval）。

这三种任务原先跑在「会话」子进程里（已移除）。现在统一登记到 outputs/runs/：
同一批步骤函数、同一份 StepContext，但没有会话目录：

    uv run python scripts/run_pipeline.py --mode factoreval --set factor_subset=ep,bp --set lookback_days=730

  · 步骤、进度、产物、失败堆栈全部写进 outputs/runs/<run_id>/（RunReporter），
    没有前端也能完整运行，前端后启动照样看到。
  · 每步的完整报告（KPI/表格/图规格）落盘到 outputs/runs/<run_id>/reports/NN_<step>.json。
  · 参数按 pipeline.registry.PARAM_SCHEMA 校验与转型，未知参数/越界/非法选项一律拒绝（退出码 2），
    不会静默忽略拼写错误。

时序：config 在 import 时读取环境变量，所以必须先把参数映射的 env 写好，再 import pipeline.registry。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

MODES = ("train", "modeleval", "factoreval")
_TRUE = {"1", "true", "yes", "on", "是"}
_FALSE = {"0", "false", "no", "off", "否", ""}


class ParamError(ValueError):
    """参数校验失败（对应退出码 2）。"""


def split_sets(items: list[str]) -> dict[str, str]:
    """['a=1', 'b=x=y'] → {'a': '1', 'b': 'x=y'}；缺等号或空 key 报错。"""
    raw: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ParamError(f"参数需要 KEY=VALUE 形式：{it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if not k:
            raise ParamError(f"参数名为空：{it!r}")
        raw[k] = v
    return raw


_FACTOR_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def apply_draft(mode: str, factor_id: str, raw: dict[str, str]) -> dict[str, str]:
    """--draft：让公式因子进入本次面板，且不碰共享缓存。必须在 import config 之前调用。

    为什么需要：USE_FORMULA_FACTORS 默认 0，库里的公式因子不在因子面板里（本仓库实测 0/20），
    直接 factor_subset=<id> 只会得到「面板中不存在该因子列」。这里在本进程内临时开启公式族，
    并把子集限定为该因子；同时关闭面板磁盘缓存（FACTOR_PANEL_CACHE=0）：缓存签名含公式库哈希与
    FORMULA_FACTOR_SUBSET，不关的话会用「只含这一个公式因子」的面板覆盖生产夜间任务共用的缓存。
    代价：全量重建面板（几分钟），换来对生产缓存零影响。
    """
    if mode != "factoreval":
        raise ParamError("--draft 只能用于 --mode factoreval")
    if not _FACTOR_ID.match(factor_id):
        raise ParamError(f"--draft 因子 id 非法：{factor_id!r}")
    if "factor_subset" in raw and raw["factor_subset"] != factor_id:
        raise ParamError("--draft 与 --set factor_subset 指向不同因子")
    os.environ["USE_FORMULA_FACTORS"] = "1"
    os.environ["FORMULA_FACTOR_SUBSET"] = factor_id
    os.environ["FACTOR_PANEL_CACHE"] = "0"
    return {**raw, "factor_subset": factor_id}


def coerce_params(mode: str, raw: dict[str, str], schema: list[dict]) -> dict:
    """按 PARAM_SCHEMA[mode] 校验并转型；返回完整参数（未给的取默认值）。"""
    by_key = {p["key"]: p for p in schema}
    unknown = sorted(set(raw) - set(by_key))
    if unknown:
        raise ParamError(f"{mode} 不接受参数：{', '.join(unknown)}（可用：{', '.join(by_key)}）")
    out = {p["key"]: p.get("default") for p in schema}
    for k, v in raw.items():
        p = by_key[k]
        t = p.get("type", "text")
        if t == "number":
            try:
                num = float(v)
            except ValueError as exc:
                raise ParamError(f"{k} 必须是数字：{v!r}") from exc
            if num != num or num in (float("inf"), float("-inf")):
                raise ParamError(f"{k} 必须是有限数字：{v!r}")
            lo, hi = p.get("min"), p.get("max")
            if lo is not None and num < lo:
                raise ParamError(f"{k}={num:g} 小于下限 {lo}")
            if hi is not None and num > hi:
                raise ParamError(f"{k}={num:g} 大于上限 {hi}")
            default = p.get("default")
            out[k] = int(num) if isinstance(default, int) and float(num).is_integer() else num
        elif t == "bool":
            s = str(v).strip().lower()
            if s not in _TRUE | _FALSE:
                raise ParamError(f"{k} 必须是布尔值（1/0/true/false）：{v!r}")
            out[k] = s in _TRUE
        elif t == "select":
            options = [o["value"] if isinstance(o, dict) else o for o in p.get("options", [])]
            if options and v not in options:
                raise ParamError(f"{k}={v!r} 不在可选项 {options}")
            out[k] = v
        else:
            out[k] = v
    return out


def _bool_env_keys_precoerced(raw: dict[str, str]) -> dict:
    """env 映射里有 4 个布尔参数（use_impact 等）：env_overrides 用 truthiness 判断，
    字符串 "0" 会被当成真。在拿到 schema 之前先把它们转成真正的 bool。"""
    out: dict = dict(raw)
    for k in ("use_impact", "use_bb_timing", "use_dynamic_topk", "use_beta_constraint"):
        if k in out:
            s = str(out[k]).strip().lower()
            if s not in _TRUE | _FALSE:
                raise ParamError(f"{k} 必须是布尔值（1/0/true/false）：{out[k]!r}")
            out[k] = s in _TRUE
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="运行 pipeline 研究模式并登记到 outputs/runs/")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--draft",
        default="",
        metavar="FACTOR_ID",
        help="只评估 factor_library 里的这个公式因子（仅 factoreval）：用生产同一条因子构建链"
        "现算该因子，不读不写因子面板缓存。",
    )
    args = ap.parse_args(argv)

    try:
        raw = split_sets(args.sets)
        if args.draft:
            raw = apply_draft(args.mode, args.draft, raw)
        # 1) 先写 env（config 在 import registry 时读取）
        from pipeline.param_env import env_overrides

        for k, v in env_overrides(_bool_env_keys_precoerced(raw)).items():
            os.environ[k] = v
        # 2) 再 import registry 取 schema，完整校验
        from pipeline import registry

        params = coerce_params(args.mode, raw, registry.PARAM_SCHEMA[args.mode])
    except ParamError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    steps = registry.STEPS[args.mode]
    label = args.label or registry.MODE_LABELS.get(args.mode, args.mode)
    return run_steps(args.mode, label, steps, params)


def run_steps(mode: str, label: str, steps: list, params: dict) -> int:
    """逐步执行并登记。异常经 RunReporter 记录堆栈后重新抛出（退出码 1）。"""
    from pipeline.context import StepContext, _jsonable

    if os.environ.get("RUN_REGISTRY", "1") == "0":
        print("RUN_REGISTRY=0：不登记，仅执行。", file=sys.stderr)
        return _run_unregistered(steps, params)

    from run_registry import RunReporter

    with RunReporter(
        mode,
        label=label,
        steps=[sid for sid, _, _ in steps],
        params={k: v for k, v in params.items() if v not in (None, "")},
        run_id=os.environ.get("RUN_ID") or None,
        parent=os.environ.get("RUN_PARENT") or None,
    ) as run:
        reports_dir = os.path.join(run.dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        ctx = StepContext(session_dir=run.dir, params=params, shared={})

        def bridge(payload: dict) -> None:
            extras = payload.get("extras") or {}
            run.progress(
                payload.get("phase", ""),
                payload.get("current", 0),
                payload.get("total", 0),
                payload.get("message", ""),
                **extras,
            )

        for i, (sid, title, fn) in enumerate(steps):
            run.step(sid, title)
            ctx.step_started_at = _now()
            ctx.on_progress = bridge
            try:
                rep = fn(ctx)
            finally:
                ctx.on_progress = None
            rep = _jsonable(rep)
            rep["status"] = "done"
            path = os.path.join(reports_dir, f"{i:02d}_{sid}.json")
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            run.artifact(f"报告 · {title}", path, kind="report")
            run.step_done(sid, str(rep.get("summary", "")))
        run.finish("done", exit_code=0)
    return 0


def _now() -> float:
    import time

    return time.time()


def _run_unregistered(steps: list, params: dict) -> int:
    from pipeline.context import StepContext

    ctx = StepContext(session_dir=os.getcwd(), params=params, shared={})
    for sid, title, fn in steps:
        print(f"== {title} ({sid})", flush=True)
        try:
            rep = fn(ctx)
        except Exception:
            traceback.print_exc()
            return 1
        print(rep.get("summary", ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
