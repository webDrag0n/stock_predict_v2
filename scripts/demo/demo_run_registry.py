"""模拟一个独立 CLI：不 import 任何 webapp，只用 run_registry 登记（用于端到端演示）。"""

import sys
import time

sys.path.insert(0, "src")
import run_registry as rr  # noqa: E402

with rr.RunReporter(
    "walkforward",
    label="演示 · 增量 walk-forward",
    steps=["load_panel", "walkforward", "report"],
    params={"engine": "incremental"},
    run_id="demo_live",
) as run:
    run.step("load_panel", "加载因子面板")
    time.sleep(1)
    run.step_done("load_panel", "5,637 只 · 160 因子")
    run.step("walkforward", "滚动 walk-forward")
    for i in range(1, 10):
        run.progress("窗口", i, 9, f"第 {i} 窗口", pred_ic=0.02 + i * 0.004)
        if i == 4:
            run.note("warn", "第 4 窗口 PredIC 偏低")
        time.sleep(1.5)
    run.step_done("walkforward", "9 窗口完成")
    run.set_param(
        result_全特征={
            "sharpe": 1.11,
            "pred_ic": 0.0243,
            "nw_t": 1.05,
            "n_windows": 9,
            "sharpe_pos_ratio": 0.556,
        }
    )
    run.artifact("walkforward", "outputs/walkforward/meta.json", kind="report")
