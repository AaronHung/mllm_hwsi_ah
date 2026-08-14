#!/usr/bin/env bash
# 夜間佇列：等主網格（$1 = 其 bash PID）結束後，依序跑
#   (1) pilot40 全網格（第二資料集 generalization）
#   (2) ablation 網格（run_ablation.sh）
# 用法：bash scripts/run_night_queue.sh <MAIN_GRID_PID> [JOBS]
set -uo pipefail
cd "$(dirname "$0")/.."

MAIN_PID="${1:?need main grid pid}"
JOBS="${2:-5}"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS=2

echo "[queue] waiting for main grid (pid $MAIN_PID) ..."
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 60; done
echo "[queue] main grid done at $(date). starting pilot40 grid."

mkdir -p logs
sem() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

for seed in 0 1 2 3 4; do
    sem
    echo "[launch] pilot40 seed=$seed"
    "$PY" scripts/cl_main.py --dataset pilot40 --order main \
        --seeds "$seed" --tag "full_s${seed}" \
        > "logs/cl_pilot40_s${seed}.log" 2>&1 &
done
wait
echo "[queue] pilot40 grid done at $(date). starting ablations."

bash scripts/run_ablation.sh "$JOBS"
echo "[queue] all done at $(date)."
