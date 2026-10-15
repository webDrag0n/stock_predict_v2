"""模型工件存取与晋升（champion/staging/archive）。

本系统是"每晚按配置重训"架构（见 src/live.py），因此 champion 的本质是
【配置快照 + 该配置训出的最新工件 + 晋升记录】：

  models/
    champion/            当前生产模型：manifest.json + ensemble_state.pkl
    staging/<model_id>/  候选留档：manifest.json + ensemble_state.pkl + evaluation.json
    archive/<model_id>/  退役冠军（保留最近 C.ARCHIVE_KEEP 个）

model_id = 日期+配置指纹（如 20261001_ab12cd34）。指纹（fingerprint_of）覆盖
会改变分数的一切（特征列/前瞻窗/ensemble 腿与训练超参），与 backtest_daily
score_cache 指纹同一函数（单一来源）；指纹相同 = 同一模型配方，配置不同 =
触发门控评估。

防损坏：manifest/工件均先写临时文件再 os.replace（原子）；读取损坏时备份为
.corrupt-<时间戳> 目录/文件后按"无 champion"处理（同 live_state 惯例）。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime

import config as C

SCHEMA_VERSION = 1

_STATE_FILE = "ensemble_state.pkl"
_MANIFEST_FILE = "manifest.json"
_EVAL_FILE = "evaluation.json"


# ---------------------------------------------------------------------------
# 配置指纹（与 score_cache 指纹同源：backtest_daily 从此处导入）
# ---------------------------------------------------------------------------
def _canon_models(v) -> str:
    """ensemble_models 归一化为逗号连接字符串（str/list/tuple 输入等价），
    保证同一配方指纹一致（旧实现 str 与 list 输入会产生不同指纹）。"""
    if not v:
        v = getattr(C, "ENSEMBLE_MODELS", "")
    parts = v.split(",") if isinstance(v, str) else list(v)
    return ",".join(str(m).strip() for m in parts if m and str(m).strip())


def fingerprint_of(feat_cols: list[str], p: dict) -> str:
    """模型配方指纹：缓存分/模型工件只在产生它的信号配置下有效。

    指纹覆盖会改变分数本身的一切：特征列、前瞻窗/embargo、ensemble 腿与
    训练超参。决策/撮合参数（topk、成本、限价模式等）不在内——它们与分数
    解耦。指纹不同 = 候选与 champion 是不同配方，需门控评估；相同 = 同一
    配方的滚动重训，直接刷新工件即可。
    """
    payload = {
        "feat_cols": [str(c) for c in feat_cols],
        "fw": int(getattr(C, "FORWARD_WINDOW", 20)),
        "embargo": int(getattr(C, "EMBARGO_DAYS", 0)),
        "ensemble_models": _canon_models(p.get("ensemble_models")),
        "n_seeds": p.get("n_seeds") or getattr(C, "ENSEMBLE_N_SEEDS", 1),
        "incr_trees": p.get("incr_trees") or getattr(C, "INCR_TREES", 0),
        "rebuild_days": p.get("rebuild_days") or getattr(C, "REBUILD_DAYS", 0),
        "train_window": p.get("train_window") or getattr(C, "TRAIN_WINDOW", 500),
        "ic_window": p.get("ic_window") or getattr(C, "IC_WINDOW", 20),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True,
                                  default=str).encode()).hexdigest()[:12]


def config_snapshot(feat_cols: list[str], p: dict) -> dict:
    """manifest 里的配置快照：重建同一配方的 IncrementalEnsemble 所需的一切。"""
    return {
        "feat_cols": [str(c) for c in feat_cols],
        "fw": int(getattr(C, "FORWARD_WINDOW", 20)),
        "embargo": int(getattr(C, "EMBARGO_DAYS", 0)),
        "ensemble_models": _canon_models(p.get("ensemble_models")),
        "n_seeds": int(p.get("n_seeds") or getattr(C, "ENSEMBLE_N_SEEDS", 1)),
        "incr_trees": int(p.get("incr_trees") or getattr(C, "INCR_TREES", 0)),
        "rebuild_days": int(p.get("rebuild_days") or getattr(C, "REBUILD_DAYS", 0)),
        "train_window": int(p.get("train_window") or getattr(C, "TRAIN_WINDOW", 500)),
        "ic_window": int(p.get("ic_window") or getattr(C, "IC_WINDOW", 20)),
    }


# ---------------------------------------------------------------------------
# 目录约定
# ---------------------------------------------------------------------------
def champion_dir() -> str:
    return os.path.join(C.MODEL_DIR, "champion")


def staging_root() -> str:
    return os.path.join(C.MODEL_DIR, "staging")


def archive_root() -> str:
    return os.path.join(C.MODEL_DIR, "archive")


def staging_dir_for(model_id: str) -> str:
    return os.path.join(staging_root(), model_id)


def _new_model_id(fingerprint: str) -> str:
    base = datetime.now().strftime("%Y%m%d") + "_" + fingerprint
    mid = base
    n = 2
    while os.path.exists(staging_dir_for(mid)):
        mid = f"{base}-{n}"
        n += 1
    return mid


# ---------------------------------------------------------------------------
# 低层读写（原子：临时文件 → os.replace）
# ---------------------------------------------------------------------------
def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _write_manifest(model_dir: str, manifest: dict) -> None:
    _atomic_write_bytes(
        os.path.join(model_dir, _MANIFEST_FILE),
        json.dumps(manifest, ensure_ascii=False, indent=2,
                   default=str).encode("utf-8"))


def _read_manifest(model_dir: str) -> dict:
    with open(os.path.join(model_dir, _MANIFEST_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def save_ensemble(ens, model_dir: str, manifest: dict) -> None:
    """把 ensemble state_dict + manifest 落盘到 model_dir（不存在则创建）。"""
    os.makedirs(model_dir, exist_ok=True)
    manifest = dict(manifest)
    manifest["schema_version"] = SCHEMA_VERSION
    _atomic_write_bytes(os.path.join(model_dir, _STATE_FILE),
                        pickle.dumps(ens.state_dict()))
    _write_manifest(model_dir, manifest)


def load_ensemble(model_dir: str):
    """读取 (ensemble, manifest)。manifest 损坏/缺失抛异常（调用方按惯例备份）。"""
    from ensemble import IncrementalEnsemble
    manifest = _read_manifest(model_dir)
    with open(os.path.join(model_dir, _STATE_FILE), "rb") as f:
        state = pickle.load(f)
    feat_cols = (manifest.get("config_snapshot") or {}).get("feat_cols") or []
    return IncrementalEnsemble.from_state_dict(state, feat_cols), manifest


def _backup_corrupt(path: str, err: Exception) -> str:
    bak = f"{path}.corrupt-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        os.replace(path, bak)
    except OSError:
        bak = ""
    print(f"[警告] 模型目录/文件 {path} 损坏（{type(err).__name__}: {err}），"
          + (f"已备份到 {bak}。" if bak else "备份失败，请人工核查。"))
    return bak


# ---------------------------------------------------------------------------
# champion / staging / promote
# ---------------------------------------------------------------------------
def load_champion():
    """读取当前 champion。不存在或损坏 → None（调用方按首次引导处理）。"""
    d = champion_dir()
    if not os.path.isdir(d):
        return None
    try:
        return load_ensemble(d)
    except Exception as e:
        _backup_corrupt(d, e)
        return None


def load_champion_manifest() -> dict | None:
    """只读 champion manifest（不反序列化工件，轻量）。损坏 → 备份并 None。"""
    d = champion_dir()
    if not os.path.isdir(d):
        return None
    try:
        return _read_manifest(d)
    except Exception as e:
        _backup_corrupt(os.path.join(d, _MANIFEST_FILE), e)
        return None


def stage_candidate(ens, feat_cols: list[str], params: dict,
                    extra: dict | None = None) -> tuple[str, dict]:
    """候选落盘到 staging/<model_id>/。返回 (staging_dir, manifest)。"""
    fp = fingerprint_of(feat_cols, params)
    mid = _new_model_id(fp)
    manifest = {
        "model_id": mid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fp,
        "config_snapshot": config_snapshot(feat_cols, params),
        "train_end_date": str(ens._last_trained_date)[:10] if ens._last_trained_date is not None else None,
        "weights": {m: round(float(w), 6) for m, w in ens.weights.items()},
        "promotion": None,
    }
    if extra:
        manifest.update(extra)
    d = staging_dir_for(mid)
    save_ensemble(ens, d, manifest)
    return d, manifest


def refresh_champion(ens, feat_cols: list[str], params: dict) -> None:
    """同一配方的滚动重训：原位刷新 champion 工件，保留既有 promotion 记录。"""
    d = champion_dir()
    old = load_champion_manifest() or {}
    manifest = {
        "model_id": old.get("model_id") or "champion_" + fingerprint_of(feat_cols, params),
        "created_at": old.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint_of(feat_cols, params),
        "config_snapshot": config_snapshot(feat_cols, params),
        "train_end_date": str(ens._last_trained_date)[:10] if ens._last_trained_date is not None else None,
        "weights": {m: round(float(w), 6) for m, w in ens.weights.items()},
        "promotion": old.get("promotion"),
    }
    save_ensemble(ens, d, manifest)


def write_evaluation(staging_dir: str, evaluation: dict) -> None:
    """把门控评估报告写进候选目录（双方指标 + 判定 + 原因）。"""
    _atomic_write_bytes(
        os.path.join(staging_dir, _EVAL_FILE),
        json.dumps(evaluation, ensure_ascii=False, indent=2,
                   default=str).encode("utf-8"))


def promote(staging_id: str, promotion_record: dict | None = None) -> dict:
    """把 staging/<staging_id> 晋升为 champion；旧 champion 归档 archive/。

    promotion_record 写入新 champion manifest 的 promotion 字段（评估双方
    指标/判定原因/时间）；None 时保留候选 manifest 里已有的记录（人工晋升
    场景：留档里已有 evaluation，记录注明 manual）。
    """
    src = staging_dir_for(staging_id)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"staging 候选不存在: {src}")
    manifest = _read_manifest(src)
    record = promotion_record or manifest.get("promotion") or {
        "reason": "manual", "decided_at": datetime.now().isoformat(timespec="seconds")}
    manifest["promotion"] = record

    champ = champion_dir()
    if os.path.isdir(champ):
        old_manifest = None
        try:
            old_manifest = _read_manifest(champ)
        except Exception:
            pass
        old_id = (old_manifest or {}).get("model_id") or \
            "champion_" + datetime.now().strftime("%Y%m%d%H%M%S")
        dst_archive = os.path.join(archive_root(), old_id)
        os.makedirs(archive_root(), exist_ok=True)
        n = 2
        while os.path.exists(dst_archive):
            dst_archive = os.path.join(archive_root(), f"{old_id}-{n}")
            n += 1
        os.replace(champ, dst_archive)
    save_ensemble_from_dir(src, champ, manifest)
    _prune_archive()
    return manifest


def save_ensemble_from_dir(src_dir: str, dst_dir: str, manifest: dict) -> None:
    """目录级复制晋升：工件 + manifest 从 src 复制到 dst（src 保留作留档）。"""
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(os.path.join(src_dir, _STATE_FILE),
                 os.path.join(dst_dir, _STATE_FILE))
    manifest = dict(manifest)
    manifest["schema_version"] = SCHEMA_VERSION
    _write_manifest(dst_dir, manifest)


def _prune_archive() -> None:
    """只保留最近 C.ARCHIVE_KEEP 个退役冠军（按目录 mtime）。"""
    root = archive_root()
    if not os.path.isdir(root):
        return
    keep = int(getattr(C, "ARCHIVE_KEEP", 5))
    entries = [os.path.join(root, d) for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))]
    entries.sort(key=os.path.getmtime)
    for d in entries[:-keep] if keep > 0 else entries:
        shutil.rmtree(d, ignore_errors=True)
