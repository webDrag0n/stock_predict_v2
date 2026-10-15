"""会话执行器 —— 独立进程跑一条流水线，每步落盘 step report。

由 webapp/sessions.py 作为 detached 子进程启动：
    python run_session.py --session-dir outputs/sessions/<id>

读取 <session-dir>/session.json 的 mode/params → 按 registry 顺序执行每个步骤：
  · 执行前把该步标 running、更新 session.json
  · 步骤函数返回 step report → 落盘 steps/NN_<step>.json，标 done
  · 任一步异常 → 该步与整个 session 标 failed，traceback 写进 step report，退出 1

关键时序：**先写 os.environ，再 import pipeline**（config 在 import 时读 env）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings

import pandas as _pd

# pandas 碎片化告警（PerformanceWarning）属构建期噪声；入口脚本显式屏蔽
# （库模块不再在 import 期动全局 warnings 状态）。
warnings.filterwarnings("ignore", category=_pd.errors.PerformanceWarning)


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="运行一个会话流水线")
    ap.add_argument("--session-dir", required=True)
    args = ap.parse_args()

    sdir = os.path.abspath(args.session_dir)
    meta_path = os.path.join(sdir, "session.json")
    meta = _load(meta_path)
    mode = meta["mode"]
    params = meta.get("params", {})

    # 1) 先写 os.environ（config 在 import pipeline.registry/steps_* 时读取 env）。
    #    param_env 是轻量模块，import 它不触发 config，因此映射可先拿到。
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root, "src"))
    sys.path.insert(0, root)
    from pipeline.param_env import env_overrides
    for k, v in env_overrides(params).items():
        os.environ[k] = v

    # 延迟到此处 import：registry 触发 pipeline.steps_* → config，读 env
    from pipeline import registry

    steps = registry.STEPS[mode]
    steps_dir = os.path.join(sdir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    from pipeline.context import StepContext, _jsonable
    ctx = StepContext(session_dir=sdir, params=params, shared={})

    meta["status"] = "running"
    meta["started"] = time.time()
    meta["current_progress"] = None
    meta["failure"] = None
    # 步骤骨架（若未初始化）
    if not meta.get("steps"):
        meta["steps"] = [{"step": sid, "title": title, "status": "pending"}
                         for sid, title, _ in steps]
    _save(meta_path, meta)

    print(f"[run_session] mode={mode} params={params}", flush=True)

    def _make_bridge(i: int, sid: str, title: str):
        """步骤进度闭包：每帧把 current_progress 写进 meta（也是 session.json）。
        写盘节流：同 phase 文本 / 同 pct 在 1.5s 内的连续帧只写一次。
        """
        last_key = [None]
        last_ts = [0.0]

        def cb(payload: dict) -> None:
            now = time.time()
            key = (payload.get("phase", ""), payload.get("pct", 0),
                   payload.get("message", ""))
            if last_key[0] == key and (now - last_ts[0]) < 1.5:
                return
            last_key[0] = key
            last_ts[0] = now
            meta["current_progress"] = {
                "step_index": i,
                "step_id": sid,
                "step_title": title,
                "ts": now,
                **payload,
            }
            _save(meta_path, meta)

        return cb

    for i, (sid, title, fn) in enumerate(steps):
        meta["steps"][i]["status"] = "running"
        meta["current_step"] = i
        meta["current_progress"] = {
            "step_index": i, "step_id": sid, "step_title": title,
            "phase": "步骤启动", "current": 0, "total": 1, "pct": 0.0,
            "elapsed_sec": 0.0, "eta_sec": 0.0, "message": "", "extras": {},
            "ts": time.time(),
        }
        _save(meta_path, meta)
        print(f"\n===== 步骤 {i+1}/{len(steps)}: {title} ({sid}) =====", flush=True)
        t0 = time.time()
        ctx.step_started_at = t0
        ctx.on_progress = _make_bridge(i, sid, title)
        try:
            rep = fn(ctx)
            rep["status"] = "done"
            rep["elapsed"] = round(time.time() - t0, 2)
            rep = _jsonable(rep)
            _save(os.path.join(steps_dir, f"{i:02d}_{sid}.json"), rep)
            meta["steps"][i]["status"] = "done"
            meta["steps"][i]["elapsed"] = rep["elapsed"]
            meta["steps"][i]["summary"] = rep.get("summary", "")
            _save(meta_path, meta)
            print(f"[ok] {title} 完成 ({rep['elapsed']}s) — {rep.get('summary','')}", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[FAILED] {title}\n{tb}", flush=True)
            cur_prog = meta.get("current_progress") or {}
            failure = {
                "step_index": i,
                "step_id": sid,
                "step_title": title,
                "exc_type": type(e).__name__,
                "exc_msg": str(e)[:500],
                "traceback": tb,
                "progress_position": cur_prog.get("phase", ""),
                "progress_detail": (f"{cur_prog.get('current',0)}/{cur_prog.get('total',0)}"
                                    f" ({cur_prog.get('pct',0):.0%})"),
                "ts": time.time(),
            }
            rep = {"step": sid, "title": title, "status": "failed",
                   "elapsed": round(time.time() - t0, 2), "summary": "步骤失败",
                   "kpis": [], "tables": [], "figures": [],
                   "failure": failure,
                   "notes": [f"❌ 步骤执行失败：\n{tb}"]}
            _save(os.path.join(steps_dir, f"{i:02d}_{sid}.json"), rep)
            meta["steps"][i]["status"] = "failed"
            meta["steps"][i]["elapsed"] = rep["elapsed"]
            meta["steps"][i]["summary"] = "步骤失败"
            meta["status"] = "failed"
            meta["finished"] = time.time()
            meta["failure"] = failure
            _save(meta_path, meta)
            return 1
        finally:
            # 下一步开始时清掉上一步 progress
            ctx.on_progress = None

    meta["status"] = "done"
    meta["finished"] = time.time()
    meta["current_step"] = len(steps)
    meta["current_progress"] = None
    _save(meta_path, meta)
    print("\n[run_session] 全部步骤完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
