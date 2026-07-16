#!/usr/bin/env bash
# 在 2022-2024 全市场数据上依次跑三个消融实验，避免与参数扫描同时跑多个重型任务。
set -o pipefail
export PYTHONUNBUFFERED=1
export ABL_START_DATE=2022-01-01
export ABL_END_DATE=2024-12-31
# 限制线程，避免抢光参数扫描的 CPU
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

echo "===== Attention leg ablation =====" | tee -a outputs/ablations_2022_2024.log
.venv/bin/python -u scripts/ablation/run_ablation_attn.py >> outputs/ablations_2022_2024.log 2>&1

echo "===== Multi-horizon target ablation =====" | tee -a outputs/ablations_2022_2024.log
.venv/bin/python -u scripts/ablation/run_ablation_multi_horizon.py >> outputs/ablations_2022_2024.log 2>&1

echo "===== Risk controls ablation =====" | tee -a outputs/ablations_2022_2024.log
.venv/bin/python -u scripts/ablation/run_ablation_risk_controls.py >> outputs/ablations_2022_2024.log 2>&1

echo "===== All ablations done =====" | tee -a outputs/ablations_2022_2024.log
