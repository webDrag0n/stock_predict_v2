#!/usr/bin/env bash
# Multi-horizon ablation on 2022-2024 with reduced cost:
# N_SEEDS=1, TRAIN_WINDOW=250, smaller test/step windows to keep enough WF windows.
set -o pipefail
export PYTHONUNBUFFERED=1
export ABL_START_DATE=2022-01-01
export ABL_END_DATE=2024-12-31
export N_SEEDS=1
export TRAIN_WINDOW=250
export TEST_WINDOW=60
export WF_STEP=60

# Limit threads so this does not starve the attention daily replay if run concurrently
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

.venv/bin/python -u scripts/ablation/run_ablation_multi_horizon.py > outputs/ablation_multi_horizon_fast.log 2>&1
