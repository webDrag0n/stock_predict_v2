"""Background job manager — launches the heavy CLI scripts as subprocesses.

The 35-min training / whole-market backfill / next-day-order jobs cannot run
inside a Dash callback. This module launches them detached, writes their stdout
to a per-job log file, and exposes status + a log tail the Jobs page polls via
dcc.Interval. Jobs are tracked in-memory for the app's lifetime.
"""
from __future__ import annotations

import os
import time
import signal
import subprocess
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "outputs", "job_logs")
os.makedirs(LOG_DIR, exist_ok=True)

PY = os.path.join(ROOT, ".venv", "bin", "python")

# Catalogue of launchable jobs: id -> (label, argv, env-overrides, description)
JOB_SPECS = {
    "walkforward": {
        "label": "Walk-forward 回测 (生产预测器)",
        "argv": [PY, "-u", "scripts/run_walkforward.py"],
        "env": {},
        "desc": "用当前 config 跑一遍滚动回测，生成 outputs/walkforward_* 与仪表盘。约数分钟。",
    },
    "multiseed": {
        "label": "多 seed 稳健性复核 (全市场)",
        "argv": [PY, "-u", "scripts/research/multiseed_robust.py"],
        "env": {"TRAIN_WINDOW": "500", "SEEDS": "42,123,7"},
        "desc": "全市场 ret_weighted 多 seed 复核。每个新 seed 约 35 分钟 (已缓存的秒回)。",
    },
    "backfill": {
        "label": "全市场数据回填 (按日增量)",
        "argv": [PY, "-u", "-c",
                 "import sys;sys.path.insert(0,'.');sys.path.insert(0,'src');"
                 "from data import get_connector;from datetime import datetime,timedelta;"
                 "c=get_connector('tushare');"
                 "s=(datetime.now()-timedelta(days=1825)).strftime('%Y-%m-%d');"
                 "e=datetime.now().strftime('%Y-%m-%d');"
                 "r=c.fetch_ohlcv_whole_market(s,e);"
                 "print('done',len(r),'rows',r['symbol'].nunique(),'stocks')"],
        "env": {},
        "desc": "按交易日增量拉取全市场行情 (daily(trade_date=))。已缓存日期秒过，缺失日补拉。",
    },
    "orders": {
        "label": "生成次日订单 (预测/实盘)",
        "argv": [PY, "-u", "predict_tomorrow.py", "--capital", "1000000"],
        "env": {},
        "desc": "基于截至今天的数据训练并生成次日可执行订单 → outputs/orders_*.csv。需较新数据。",
    },
}


@dataclass
class Job:
    job_id: str
    spec_id: str
    label: str
    log_path: str
    pid: int = 0
    status: str = "running"      # running | done | failed | killed
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    _proc: object = None

    def to_dict(self) -> dict:
        # Build explicitly — NOT via dataclasses.asdict(), which deep-copies every
        # field including _proc (a subprocess.Popen holding an unpicklable thread
        # lock) before any filtering, raising "cannot pickle '_thread.lock'".
        return {
            "job_id": self.job_id,
            "spec_id": self.spec_id,
            "label": self.label,
            "log_path": self.log_path,
            "pid": self.pid,
            "status": self.status,
            "started": self.started,
            "finished": self.finished,
            "elapsed": (self.finished or time.time()) - self.started,
        }


_JOBS: dict[str, Job] = {}


def launch(spec_id: str) -> Job:
    """Launch a job from JOB_SPECS; returns the Job handle."""
    spec = JOB_SPECS[spec_id]
    ts = time.strftime("%Y%m%d_%H%M%S")
    job_id = f"{spec_id}_{ts}"
    log_path = os.path.join(LOG_DIR, f"{job_id}.log")
    env = dict(os.environ)
    env.update(spec.get("env", {}))
    logf = open(log_path, "w")
    proc = subprocess.Popen(spec["argv"], cwd=ROOT, env=env,
                            stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    job = Job(job_id=job_id, spec_id=spec_id, label=spec["label"],
              log_path=log_path, pid=proc.pid)
    job._proc = proc
    _JOBS[job_id] = job
    return job


def _refresh(job: Job) -> None:
    if job.status != "running" or job._proc is None:
        return
    rc = job._proc.poll()
    if rc is None:
        return
    job.status = "done" if rc == 0 else "failed"
    job.finished = time.time()


def all_jobs() -> list[dict]:
    for j in _JOBS.values():
        _refresh(j)
    return [j.to_dict() for j in sorted(_JOBS.values(),
                                        key=lambda x: x.started, reverse=True)]


def tail_log(job_id: str, n: int = 80) -> str:
    job = _JOBS.get(job_id)
    if job is None or not os.path.exists(job.log_path):
        return ""
    try:
        with open(job.log_path, "r", errors="replace") as f:
            lines = f.readlines()
        # drop the known proxy-noise lines so the tail stays readable
        lines = [ln for ln in lines if ln.strip() not in ("'close'",)]
        return "".join(lines[-n:])
    except Exception as e:
        return f"(无法读取日志: {e})"


def kill(job_id: str) -> bool:
    job = _JOBS.get(job_id)
    if job is None or job.status != "running" or job._proc is None:
        return False
    try:
        os.killpg(os.getpgid(job.pid), signal.SIGTERM)
        job.status = "killed"
        job.finished = time.time()
        return True
    except Exception:
        return False
