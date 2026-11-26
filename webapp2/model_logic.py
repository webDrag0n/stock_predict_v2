"""模型页的纯逻辑：champion/staging 指标口径、门控结论、两模型对比、配方 diff。

不依赖 dash，便于单测。

指标口径（踩过的坑）：
  champion 目录里的 evaluation.json 是它【作为候选入围时】的门控评估，其中 "champion" 侧是
  当时被它取代的旧模型，不是它自己；而晋升后又不会更新。所以 champion 自己的表现取
  manifest.promotion.candidate_metrics（晋升时它在门控重放里的指标）。旧前端直接读
  evaluation.json["champion"]，会把【旧模型】的指标展示成现任 champion 的。
"""

from __future__ import annotations

from . import data

# 与 model_gate / evaluation.json 公共子集：(键, 标签, 越大越好)
METRIC_KEYS: tuple[tuple[str, str, bool], ...] = (
    ("sharpe", "Sharpe", True),
    ("sortino", "Sortino", True),
    ("max_drawdown", "最大回撤 %", True),
    ("total_return", "总收益 %", True),
    ("alpha_ann", "年化 Alpha", True),
    ("beta", "Beta", False),
    ("hit_rate", "胜率", True),
    ("t_nw", "t(NW)", False),
    ("n_days", "评估天数", False),
    ("n_features", "特征数", False),
)
_RECIPE_SKIP = ("feat_cols",)


def number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _metric_dict(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if k != "params"}


def identical_replay(ev: dict | None) -> bool:
    """候选与 champion 两侧指标逐项相同 ⇒ 这次门控两边跑的是同一套列（F13 的症状），
    结论不可信。两侧都没有数值指标时不判定。"""
    c, h = _metric_dict((ev or {}).get("candidate")), _metric_dict((ev or {}).get("champion"))
    nums = {k for k, v in c.items() if number(v)}
    return bool(nums) and all(c.get(k) == h.get(k) for k in nums)


def champion_metrics(man: dict | None, ev: dict | None) -> tuple[dict, str]:
    """(指标, 来源说明)。优先晋升记录里它自己的指标，其次它入围时 evaluation 的候选侧
    （仅当那次评估结论为 promoted 且两侧不相同）。拿不到就返回空，不拿别的模型凑数。"""
    promo = (man or {}).get("promotion") or {}
    own = promo.get("candidate_metrics")
    if isinstance(own, dict) and any(number(v) for v in own.values()):
        window = promo.get("eval_window")
        tail = (
            f"（{window[0]} ~ {window[1]}）"
            if isinstance(window, list) and len(window) == 2
            else ""
        )
        return own, f"晋升时门控重放{tail}"
    if (ev or {}).get("decision") == "promoted" and not identical_replay(ev):
        return _metric_dict(ev.get("candidate")), "入围时门控评估"
    return {}, "无可用评估（bootstrap 晋升或评估记录不可信）"


def champion_id() -> str | None:
    man = data.champion_manifest()
    return (man or {}).get("model_id")


def is_champion_copy(entry: dict) -> bool:
    """staging 里 model_id 等于现任 champion 的条目：晋升是「复制」不是「移动」，
    所以已晋升的候选会一直留在 staging。它不是待决策的候选，不能再晋升，也不该参与对比。"""
    cid = champion_id()
    man = entry.get("manifest") or {}
    return bool(cid) and cid in (entry.get("id"), man.get("model_id"))


def pending_candidates(entries: list[dict]) -> list[dict]:
    return [e for e in entries if not is_champion_copy(e)]


def staging_summary(entry: dict) -> dict:
    """staging 条目 → 对比结论。evaluation 契约见 src/model_gate.py::run_gate。"""
    ev = entry.get("evaluation") or {}
    cand, champ = ev.get("candidate") or {}, ev.get("champion") or {}
    cs, hs = cand.get("sharpe"), champ.get("sharpe")
    cd, hd = cand.get("max_drawdown"), champ.get("max_drawdown")
    return {
        "id": entry.get("id", "?"),
        "has_eval": bool(ev),
        "d_sharpe": (cs - hs) if number(cs) and number(hs) else None,
        "d_dd": (cd - hd) if number(cd) and number(hd) else None,
        "decision": ev.get("decision"),
        "eligible": ev.get("decision") == "promoted",
        "reason": ev.get("reason", ""),
        "decided_at": ev.get("decided_at"),
        "eval_days": ev.get("eval_days"),
        "window": ev.get("window"),
        "suspect": identical_replay(ev),
        "missing_cols": ev.get("champion_missing_cols") or [],
        "is_champion": is_champion_copy(entry),
    }


def model_record(ref: str) -> dict | None:
    """ref = 'champion' | staging 目录名 | 'archive:<id>' → {ref,id,role,manifest,recipe,metrics,src}。"""
    if ref == "champion":
        man = data.champion_manifest()
        if not man:
            return None
        metrics, src = champion_metrics(man, data.champion_evaluation())
        return {
            "ref": ref,
            "id": man.get("model_id", "champion"),
            "role": "champion",
            "manifest": man,
            "evaluation": {},
            "recipe": man.get("config_snapshot") or {},
            "metrics": metrics,
            "src": src,
        }
    if ref.startswith("archive:"):
        rid = ref[len("archive:") :]
        for e in data.archived_models():
            if e["id"] == rid and e["manifest"]:
                man = e["manifest"]
                m, src = champion_metrics(man, e["evaluation"])
                return {
                    "ref": ref,
                    "id": man.get("model_id", rid),
                    "role": "archive",
                    "manifest": man,
                    "evaluation": e["evaluation"] or {},
                    "recipe": man.get("config_snapshot") or {},
                    "metrics": m,
                    "src": src,
                }
        return None
    for e in data.staging_models():
        if e["id"] != ref or not e["manifest"]:
            continue
        ev = e["evaluation"] or {}
        return {
            "ref": ref,
            "id": e["manifest"].get("model_id", ref),
            "role": "staging",
            "manifest": e["manifest"],
            "evaluation": ev,
            "recipe": e["manifest"].get("config_snapshot") or {},
            "metrics": _metric_dict(ev.get("candidate")),
            "src": "门控重放（候选侧）"
            + ("；两侧指标相同，不可信" if identical_replay(ev) else ""),
        }
    return None


def model_options() -> list[dict]:
    opts = []
    if data.champion_manifest():
        opts.append({"label": "Champion（线上）", "value": "champion"})
    opts += [
        {"label": f"Staging · {e['id']}", "value": e["id"]}
        for e in pending_candidates(data.staging_models())
    ]
    opts += [
        {"label": f"Archive · {e['id']}", "value": f"archive:{e['id']}"}
        for e in data.archived_models()
        if e["manifest"]
    ]
    return opts


def metric_rows(ra: dict | None, rb: dict | None) -> list[dict]:
    """[{key,label,a,b,delta,better}]；delta=b-a。回撤越接近 0 越好（值为负，越大越好）。"""
    ma, mb = (ra or {}).get("metrics") or {}, (rb or {}).get("metrics") or {}
    rows = []
    for key, label, higher in METRIC_KEYS:
        va, vb = ma.get(key), mb.get(key)
        if va is None and vb is None:
            continue
        delta = better = None
        if number(va) and number(vb):
            delta = vb - va
            if higher and delta != 0:
                better = "b" if delta > 0 else "a"
        rows.append(
            {"key": key, "label": label, "a": va, "b": vb, "delta": delta, "better": better}
        )
    return rows


def recipe_diff(a: dict, b: dict) -> list[dict]:
    keys = sorted((set(a) | set(b)) - set(_RECIPE_SKIP))
    return [{"key": k, "a": a.get(k), "b": b.get(k), "same": a.get(k) == b.get(k)} for k in keys]


def feat_diff(a: dict, b: dict) -> dict:
    fa, fb = set(a.get("feat_cols") or []), set(b.get("feat_cols") or [])
    return {"n_a": len(fa), "n_b": len(fb), "only_a": sorted(fa - fb), "only_b": sorted(fb - fa)}


def recipe_summary(snapshot: dict) -> str:
    if not snapshot:
        return "无配方快照"
    from model_display import model_label

    models = "、".join(
        model_label(m) for m in str(snapshot.get("ensemble_models", "?")).split(",") if m
    )
    return (
        f"特征 {len(snapshot.get('feat_cols') or [])} 个 · 前瞻 {snapshot.get('fw', '?')}d · "
        f"子模型 [{models}] × {snapshot.get('n_seeds', '?')} seeds · "
        f"训练窗 {snapshot.get('train_window', '?')} · 重建 {snapshot.get('rebuild_days', '?')}d · "
        f"增量树 {snapshot.get('incr_trees', '?')} · IC 窗 {snapshot.get('ic_window', '?')}"
    )


def guidance(champion: dict | None, stagings: list) -> str:
    """顶部唯一建议动作（三态互斥）。"""
    if not champion:
        return "还没有线上模型：先训练候选，或跑一次夜间任务让首个 champion 自动 bootstrap。"
    stagings = pending_candidates(stagings)
    eligible = [s for s in stagings if staging_summary(s)["eligible"]]
    if eligible:
        return f"有 {len(eligible)} 个候选已达晋升门槛：在下方对比后决定是否晋升。"
    if stagings:
        return "有候选留档但都未达门槛；可对比配方差异，确有理由再人工晋升。"
    return "线上模型运行中。想验证它，先跑一次 walk-forward。"


def promote_plan(model_id: str) -> dict:
    """晋升前的二次确认信息。纯函数（读 data 层）。

    can=False → why 说明原因；need_reason=True → 候选未达门槛或评估可疑，必须填写理由
    （理由会写入新 champion 的 manifest.promotion 与 models/promotions.log）。
    """
    entry = next((e for e in data.staging_models() if e["id"] == model_id), None)
    out = {"can": False, "why": "", "need_reason": False, "warnings": [], "summary": None}
    if entry is None:
        out["why"] = f"staging 候选不存在：{model_id}"
        return out
    s = staging_summary(entry)
    out["summary"] = s
    if s["is_champion"]:
        out["why"] = (
            "该候选就是当前 champion（晋升后留档的副本），无需再晋升；不需要时可以删除这份副本"
        )
        return out
    if not entry["manifest"]:
        out["why"] = "候选缺少 manifest.json，不能晋升"
        return out
    if not s["has_eval"]:
        out["warnings"].append("没有门控评估记录（evaluation.json 缺失）")
        out["need_reason"] = True
    elif not s["eligible"]:
        out["warnings"].append(f"门控结论为未达门槛：{s['reason'] or '无说明'}")
        out["need_reason"] = True
    if s["suspect"]:
        out["warnings"].append(
            "候选与 champion 两侧指标逐项相同：这次评估两边重放的是同一套特征，结论不可信"
        )
        out["need_reason"] = True
    if s["missing_cols"]:
        out["warnings"].append(
            f"champion 有 {len(s['missing_cols'])} 个特征列在面板里缺失，门控对比不公平"
        )
        out["need_reason"] = True
    out["can"] = True
    return out
