#!/usr/bin/env bash
# Re-run multi-horizon ablation with the same reduced-config env as before.
export USE_ALPHA158_FACTORS=0
export USE_GTJA191_FACTORS=0
export USE_OPTION_FEATURES=0
export N_SEEDS=1
export TRAIN_WINDOW=250
export TEST_WINDOW=60
export WF_STEP=60
export PYTHONUNBUFFERED=1
export ABL_START_DATE=2022-01-01
export ABL_END_DATE=2024-12-31
.venv/bin/python -u scripts/ablation/run_ablation_multi_horizon.py > outputs/ablation_multi_horizon_fast.log 2>&1
