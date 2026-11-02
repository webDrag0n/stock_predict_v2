"""风控层贡献对照实验：全风控关闭的 walk-forward（P1-4 审计）。

与标准 CLI（scripts/run_walkforward.py）唯一的区别是风控层全关——通过环境变量
（config import 时读取）：

    USE_VOL_TARGET=0 USE_DD_BRAKE=0 USE_IC_SCALE=0 USE_IV_SCALE=0 \
        uv run python scripts/research/risk_off_control.py

产物写到 outputs/ablations/walkforward_riskoff/，不覆盖 outputs/walkforward/
的 canonical 结果。用于回答：Sharpe 提升多少来自风控层、多少来自信号本身。
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
)

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import run_walkforward, save_walkforward_result


def main():
    for k in ("USE_VOL_TARGET", "USE_DD_BRAKE", "USE_IC_SCALE", "USE_IV_SCALE"):
        assert getattr(C, k) is False, f"风控层未关闭：{k}={getattr(C, k)}（需 env 置 0）"
    panel = load_factor_panel(keep_trade_cols=True, verbose=True)
    factors = list_factor_columns(panel)
    res = run_walkforward(panel, factors, save=False)
    outdir = save_walkforward_result(
        res, outdir=os.path.join(C.OUTPUT_DIR, "ablations", "walkforward_riskoff")
    )
    print(f"风控关闭对照结果 -> {outdir}")


if __name__ == "__main__":
    main()
