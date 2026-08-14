#!/usr/bin/env bash
# 主實驗全網格（protocol §3）：orders {main,reverse} × 7 methods × K{1,2,4} × 5 seeds。
# 以 seed 分片平行（每個 shard 一個 python process），$1 = 平行度（預設 4）。
# 用法：bash scripts/run_main_cl.sh 8
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS="${1:-4}"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS=2

mkdir -p results logs
sem() {           # 簡易信號量：最多 JOBS 個背景工作
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
}

for order in main reverse; do
    for seed in 0 1 2 3 4; do
        sem
        echo "[launch] order=$order seed=$seed"
        "$PY" scripts/cl_main.py --dataset can --order "$order" \
            --seeds "$seed" --tag "full_s${seed}" \
            > "logs/cl_${order}_s${seed}.log" 2>&1 &
    done
done
wait
echo "all shards done."
