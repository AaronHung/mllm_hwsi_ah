# RunPod Quickstart — fresh boot / restart card

Use this card after every pod restart. Attach the same Network Volume at
`/workspace` first. Do not memorize Python paths: bootstrap recreates the
container-side pieces and sources the persistent environment.

## Fresh boot: one block

```bash
set -euo pipefail
export PROJECT_DIR=/workspace/mllm_hwsi_ah

if [ ! -d "$PROJECT_DIR/.git" ]; then
  git clone https://github.com/AaronHung/mllm_hwsi_ah.git "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/runpod_bootstrap.sh"
git pull --ff-only origin main
source /workspace/.mllm_hwsi_ah/env.sh
export CAN_ROOT=/workspace/datasets/can_dataset

echo "commit=$(git rev-parse HEAD)"
echo "python=$PYTHON_BIN"
"$PYTHON_BIN" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
tmux ls 2>/dev/null || echo "NO_TMUX_SESSION_YET"
```

Expected: `bootstrap PASS`, data `PASS`, `cuda True`, and the expected Git
commit. If data fails, stop; do not run an experiment.

## Start / run / detach

```bash
export RUN_TAG="v2_<dataset>_<purpose>_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="/workspace/mllm_hwsi_ah/runs/v2/$RUN_TAG"
mkdir -p "$RUN_DIR/logs"
tmux new -s "$RUN_TAG"
```

Inside tmux:

```bash
cd "$PROJECT_DIR"
set -o pipefail
"$PYTHON_BIN" scripts/cl_main.py \
  --dataset can --order main --device cuda \
  --output-dir "$RUN_DIR/results" \
  --tag "$RUN_TAG" --resume \
  2>&1 | tee "$RUN_DIR/logs/$RUN_TAG.log"
```

For a short gate, add `--smoke`. For the full grid, use the documented
`scripts/run_main_cl.sh` wrapper. In the RunPod web terminal, if `Ctrl-b d`
does not work, type this inside tmux and press Enter:

```bash
tmux detach-client
```

Reattach later with:

```bash
tmux attach -t "$RUN_TAG"
```

## What survives a stopped pod

On the attached `/workspace` volume:

- repository and Git checkout;
- `/workspace/venvs/mllm_hwsi_ah`;
- `/workspace/.mllm_hwsi_ah/env.sh`;
- `/workspace/bin/tmux` if a static fallback was uploaded;
- `/workspace/datasets/can_dataset`;
- `mllm_hwsi_ah/data/can_cache`;
- `/workspace/SHA256SUMS.txt`;
- `runs/v2/<run_tag>/` logs, CSVs, metadata, and checkpoints.

What does not survive stopping/destroying the pod:

- the running Python process;
- the active tmux session;
- `/usr/bin/tmux` installed by `apt-get`;
- shell exports in the current terminal.

Therefore, a disconnect only needs `tmux attach`; a stopped/preempted pod needs
the fresh-boot block and then the same run command with `--resume`. Completed
`(seed, method, K)` units are skipped. Never reuse a Protocol-v1 filename.

The full explanations, upload procedure, aggregation, and publish workflow are
in [`RUNPOD_SOP.md`](RUNPOD_SOP.md).
