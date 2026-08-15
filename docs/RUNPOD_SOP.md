# RunPod SOP v2 — `mllm_hwsi_ah`

---

## 開機 Aaron 筆記：

已補成可重開使用的 SOP。

修正內容：

- `CAN_ROOT` 預設改為實際 volume 路徑：
  `/workspace/datasets/can_dataset`
- bootstrap 會持久化 `PYTHON_BIN`
- `smoke_test.sh` 自動使用 `/workspace/venvs/mllm_hwsi_ah`
- 新增一頁式 `docs/RUNPOD_QUICKSTART.md`

重開 pod 後真正需要記的只有：

1. 掛回同一個 `/workspace` volume
2. `source scripts/runpod_bootstrap.sh`
3. `git pull --ff-only`
4. `tmux ls` 或建立新 session
5. 用同一個 `RUN_TAG` 加 `--resume`

`/workspace` 內的 repo、venv、cache、dataset、logs、checkpoints 會保留；active tmux session、shell environment、`/usr/bin/tmux` 不保留，但 bootstrap 會重建。

---

This is the fresh-boot runbook for Mac development → GitHub → RunPod CUDA →
GitHub → Mac analysis. It is intentionally interactive: Aaron runs one block,
pastes its output, and only then do we continue. The agent cannot SSH into the
pod or infer its state.

For a short restart card, see [`RUNPOD_QUICKSTART.md`](RUNPOD_QUICKSTART.md).

## Non-negotiable rules

1. Run `scripts/smoke_test.sh --all` on the Mac before pushing a code change.
   The same commit must be used on RunPod; no remote code edits.
2. All seeds contributing to one table use one backend. Probe/mechanism runs
   may use Mac MPS; pilot40 five-seed grids run on RunPod CUDA. See
   `docs/protocol.md` §9.
3. Protocol-v1 artifacts under `results/` and `figures/` are frozen. New
   artifacts use `runs/v2/<run_tag>/`; never reuse a v1 filename or tag.
4. Checkpoints are on for every new grid: each completed
   `(seed, method, K)` writes a checkpoint manifest, and `--resume` skips only
   those completed units.
5. Do not commit data, model weights, PATs, or `.env` files.

## What was adopted from `01_navipath`

- one device resolver (`cuda > mps > cpu`) and MPS fallback;
- a tiny smoke gate before a full run;
- tmux as the disconnect-safe process wrapper;
- persistent `outputs`/run naming instead of overwriting canonical results;
- the `SPEC/ADR → implementation → smoke → worklog → commit` discipline.

The implementation is adapted to this repo as `nav/device.py`,
`scripts/smoke_test.sh`, `scripts/runpod_bootstrap.sh`, and `runs/v2/`.

## Fresh pod runbook

### 0. Mac preflight and push

```bash
cd /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah
bash scripts/smoke_test.sh --all
git status --short
git add nav scripts docs/protocol.md docs/RUNPOD_SOP.md
git commit -m "Add dual-backend RunPod infrastructure"
git push origin main
```

Do not continue if either CPU or MPS smoke fails. On Mac, the resolver may
select MPS explicitly; on RunPod it must resolve CUDA.

### 1. Start the pod and attach the persistent volume

Use the same RunPod Network Volume mounted at `/workspace`. A container/root
filesystem can disappear when the pod is stopped; `/workspace` survives only
when the same volume is attached.

Interactive probe — run exactly this and paste all output:

```bash
set -u
echo "host=$(hostname)"
id
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ls -ld /workspace /workspace/mllm_hwsi_ah 2>&1
if [ -d /workspace/mllm_hwsi_ah/.git ]; then
  git -C /workspace/mllm_hwsi_ah rev-parse --show-toplevel
  git -C /workspace/mllm_hwsi_ah rev-parse HEAD
else
  echo "REPO_MISSING"
fi
```

Expected: an NVIDIA GPU row, `/workspace` present, and either the repository
with a commit hash or `REPO_MISSING`. Do not clone or upload data until this
state is checked.

If `REPO_MISSING`, clone only the code:

```bash
git clone https://github.com/AaronHung/mllm_hwsi_ah.git /workspace/mllm_hwsi_ah
```

### 2. Bootstrap the durable environment and data check

Set Git identity before sourcing if the default address is not desired:

```bash
export GIT_USER_NAME="AaronHung (runpod)"
export GIT_USER_EMAIL="YOUR_GITHUB_EMAIL"
source /workspace/mllm_hwsi_ah/scripts/runpod_bootstrap.sh
```

The bootstrap is idempotent and does all of the following:

- installs `tmux` with `apt-get` when root is available;
- otherwise uses `/workspace/bin/tmux` if a Linux static binary was uploaded;
- creates `/workspace/venvs/mllm_hwsi_ah` with system-site packages;
- installs `requirements-nav.txt`;
- writes and sources `/workspace/.mllm_hwsi_ah/env.sh`;
- exports `PROJECT_DIR`, `CAN_ROOT`, `PATH`, `PYTHONPATH`, Git identity, and
  `PYTORCH_ENABLE_MPS_FALLBACK`;
- runs `scripts/verify_runpod_data.sh`.

`tmux` itself is a process and does not survive stopping/destroying a pod.
The package/static binary and virtual environment do survive on the attached
`/workspace` volume. A network disconnect only detaches SSH; it does not kill
the tmux process. After a crash/preemption, start a new session and use
`--resume`; files and checkpoint manifests under `/workspace` remain.

The data verifier checks first for `data/can_cache/`, all
`tcga_{esca,lung,rcc,brca}/table/*.csv`, and every
`datasplit/fold_1.npz`. It verifies the committed `SHA256SUMS.txt` and the
volume-side data `SHA256SUMS.txt`. If the data manifest does not exist, create
it once from a trusted copy with
`WORKSPACE_ROOT=/workspace CAN_ROOT=/workspace/can_dataset_min
CACHE_ROOT=/workspace/mllm_hwsi_ah/data/can_cache
bash scripts/make_data_manifest.sh`, then rerun the verifier. If a required
path or checksum is missing/corrupt, upload only that reported path and rerun
the verifier. It never uploads or deletes data automatically.

### 3. Pull the exact tested commit and verify

After bootstrap succeeds:

```bash
cd "$PROJECT_DIR"
git pull --ff-only origin main
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$VENV_DIR/bin/python" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
bash scripts/verify_runpod_data.sh
```

Expected: the commit is the one just pushed, `cuda True`, and data `PASS`.
If `git pull` is not fast-forwardable, stop and paste the output; do not
force-reset the pod.

### 4. End-to-end rehearsal

Choose a UTC run tag, for example:

```bash
export RUN_TAG="v2_rehearsal_can_main_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_DIR/logs"
tmux new -s "$RUN_TAG"
```

Inside tmux, run the tiny two-task, `K=1` rehearsal:

```bash
cd "$PROJECT_DIR"
set -o pipefail
"$VENV_DIR/bin/python" scripts/cl_main.py \
  --dataset can --order main --smoke --device cuda \
  --output-dir "$RUN_DIR/results" \
  --tag "$RUN_TAG" --resume \
  2>&1 | tee "$RUN_DIR/logs/rehearsal.log"
```

The expected result is a finite CSV with two tasks, `K=1`,
`resolved_device=cuda`, `metadata.json`, and `checkpoints.json`. Detach with
`Ctrl-b`, then `d`; reattach with:

```bash
tmux attach -t "$RUN_TAG"
```

### 5. Full grid

Only after the rehearsal passes:

```bash
cd "$PROJECT_DIR"
export DATASET=pilot40
export ORDERS=main
export RUN_TAG="v2_pilot40_main_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$PROJECT_DIR/runs/v2/$RUN_TAG"
export DEVICE=cuda
mkdir -p "$RUN_ROOT/logs"
tmux new -s "$RUN_TAG"
```

Inside tmux:

```bash
cd "$PROJECT_DIR"
set -o pipefail
RUN_TAG="$RUN_TAG" RUN_ROOT="$RUN_ROOT" DEVICE=cuda \
  DATASET="$DATASET" ORDERS="$ORDERS" \
  bash scripts/run_main_cl.sh 2 \
  2>&1 | tee "$RUN_ROOT/logs/main_grid.log"
```

The wrapper writes every seed shard below `$RUN_ROOT/results/`, uses
`--resume`, and never writes Protocol-v1 files. For ablations, use a new
`RUN_TAG` and run `scripts/run_ablation.sh`; never reuse the main-grid tag.

### 6. Aggregate and publish

After the tmux command finishes with exit code 0:

```bash
cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" scripts/aggregate_results.py \
  --dataset "$DATASET" --order "$ORDERS" \
  --tag "$RUN_TAG" \
  --input-dir "$RUN_ROOT/results" \
  --output-dir "$RUN_ROOT"
git status --short
git add "$RUN_ROOT/results" "$RUN_ROOT/figures" "$RUN_ROOT/logs"
git commit -m "results: $RUN_TAG"
git pull --rebase origin main
git push origin main
```

The Mac side then runs:

```bash
cd /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah
git pull --ff-only origin main
```

Only after the Mac verifies the run metadata, CSVs, aggregate, and logs may
the pod be stopped. Keep the Network Volume attached for the next session.

## Disconnect, crash, and preemption recovery

1. Reconnect to the same pod and source
   `/workspace/.mllm_hwsi_ah/env.sh` (or rerun bootstrap).
2. Run `tmux ls`; if the old session exists, use
   `tmux attach -t <run_tag>`.
3. If no session exists, inspect
   `runs/v2/<run_tag>/logs/`, `metadata.json`, and `checkpoints.json`.
4. Verify the commit and data again, then rerun the exact command with
   `--resume`. Completed units are skipped; a partially completed unit is
   recomputed and appended once.
5. Never rename a partial run to a completed run and never overwrite a v1
   artifact. If the backend changed, start a new run tag and table.

## Run-tag convention

Use:

```text
v2_<dataset>_<order>_<purpose>_<backend>_<UTC-compact>
```

Examples: `v2_can_main_rehearsal_cuda_20260815T031500Z`,
`v2_pilot40_main_grid_cuda_20260815T040000Z`, and
`v2_pilot40_probe_mechanism_mps_20260814T140000Z`.

Tags are immutable identifiers, not display labels. A rerun gets a new
timestamped tag even if its scientific parameters are identical.
