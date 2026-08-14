#!/usr/bin/env bash
# Ablation grid（protocol §3 note + plan）：
#   A1 utility weighting     ：ours_uniform vs ours（main grid 已含 ours）
#   A2 replay memory size    ：buffer-cap ∈ {128, 2048}（main = 512）
#   A3 λ 敏感度              ：λ ∈ {0.3, 3.0}（main = 1.0）
# 全部：main order、K ∈ {1,2}、5 seeds。$1 = 平行度（預設 4）。
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${1:-4}"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS=2

mkdir -p results logs
sem() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

run() {  # run <tag> <extra args...>
    local tag="$1"; shift
    for seed in 0 1 2 3 4; do
        sem
        echo "[launch] $tag seed=$seed"
        "$PY" scripts/cl_main.py --dataset can --order main \
            --seeds "$seed" --budgets 1 2 --tag "${tag}_s${seed}" "$@" \
            > "logs/abl_${tag}_s${seed}.log" 2>&1 &
    done
}

run ablA1 --methods ours_uniform
run ablA2a --methods ours --buffer-cap 128
run ablA2b --methods ours --buffer-cap 2048
run ablA3a --methods ours --lam 0.3
run ablA3b --methods ours --lam 3.0
wait
echo "ablation done."
