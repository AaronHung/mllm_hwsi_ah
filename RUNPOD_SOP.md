# RunPod SOP — mllm_hwsi_ah

> **v2 canonical SOP:** use [`docs/RUNPOD_SOP.md`](docs/RUNPOD_SOP.md) or the
> [`restart card`](docs/RUNPOD_QUICKSTART.md).
> This file is retained as the Protocol-v1 historical reference; new runs
> must use `runs/v2/<run_tag>/` and must not write the paths below.

本專案版標準流程：Mac 開發 + smoke → GitHub 傳程式 → RunPod tmux 跑完整版 → 結果 push 回 GitHub → Mac pull 分析。大檔（特徵 tar）不走 git，走 scp 或 Network Volume。

## 0. 原則

- **程式**只從 `git@github.com:AaronHung/mllm_hwsi_ah.git` 傳（push/pull），不 scp 程式。
- **特徵大檔**（`data_src/brca_pilot40_k48.tar.part-*`）走 scp 或掛 Network Volume，永不進 git。
- **結果**（`results/`、`figures/`，都是 KB～百 KB 級）跑完自動 commit + push，Mac 端 `git pull` 即可分析。
- **一律 tmux**：斷線後 job 繼續跑，`tmux attach` 隨時回去看。

## 1. 開 Pod

- GPU：RTX 4090（24GB）夠用；nav/ 系列實驗都是小模型（MLP + scorer），瓶頸只在特徵載入。
- Template：官方 PyTorch 2.x + CUDA 12.x。
- 磁碟：Container 20GB + Volume 30GB（特徵解壓後約 5.6GB）。
- 記下 SSH 連線資訊（IP、port）。

## 2. 第一次進 Pod：環境 + 程式 + 資料

```bash
# --- 在 Pod 上 ---
cd /workspace
git clone https://github.com/AaronHung/mllm_hwsi_ah.git   # 公開讀取用 https 即可
cd mllm_hwsi_ah
pip install -r requirements-nav.txt                        # 只裝 nav 需要的輕量套件

# --- 在 Mac 上（另一個 terminal）傳特徵 ---
cd /Users/aaron/research/01_mllm_hwsi/data_src
scp -P <PORT> brca_pilot40_k48.tar.part-* SHA256SUMS.txt pilot40_selection.csv \
    root@<POD_IP>:/workspace/mllm_hwsi_ah/data_src/

# --- 回到 Pod：驗證 + 解壓 ---
cd /workspace/mllm_hwsi_ah/data_src
sha256sum -c SHA256SUMS.txt          # 三個 part 必須 OK
mkdir -p ../data
cat brca_pilot40_k48.tar.part-* | tar -xf - -C ../data/
cp pilot40_selection.csv ../data/
```

若使用 Network Volume：把 `data_src/` 放在 Volume 上一次，之後每次開 Pod 直接解壓，不必重傳。

## 3. Push 結果需要的 git 身分（Pod 上一次性）

```bash
git config user.name  "AaronHung (runpod)"
git config user.email "<你的 github email>"
# push 用 PAT：
git remote set-url origin https://<PAT>@github.com/AaronHung/mllm_hwsi_ah.git
```

## 4. 跑長任務：tmux + 自動 push（斷線不掛的機制）

```bash
tmux new -s run          # 建 session；斷線後 `tmux attach -t run` 回來
```

在 tmux 內貼「一整段」，跑完自己 push，中途斷線完全不影響：

```bash
cd /workspace/mllm_hwsi_ah && \
python scripts/gate1_single_task.py 2>&1 | tee results/gate1_run.txt && \
git pull --rebase && \
git add results/ figures/ && \
git commit -m "RunPod: gate1 full run results" && \
git push
```

常用 tmux 指令：

| 動作 | 指令 |
|---|---|
| 離開但不中斷 | `Ctrl-b` 然後 `d` |
| 回去看 | `tmux attach -t run` |
| 看有哪些 session | `tmux ls` |
| 殺掉 session | `tmux kill-session -t run` |

## 4b. 主實驗（cl_main：can_dataset 4 任務序列）

can_dataset 原始特徵不必上 RunPod——實驗只吃 `data/can_cache/`（約 200MB，Mac 端
`scripts/prep_can_cache.py` 產出）。傳輸：

```bash
# --- Mac：打包 cache（一次） ---
cd /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah/data
tar -czf can_cache.tgz can_cache
scp -P <PORT> can_cache.tgz root@<POD_IP>:/workspace/mllm_hwsi_ah/data/

# --- Pod：解壓 ---
cd /workspace/mllm_hwsi_ah/data && tar -xzf can_cache.tgz && rm can_cache.tgz
# split npz 已含在 cache？否 —— fold split 需要 can_dataset 的 datasplit：
scp -P <PORT> -r /Users/aaron/research/can_dataset/tcga_*/datasplit \
    root@<POD_IP>:/workspace/can_dataset_min/   # 見下方 CAN_ROOT 說明
```

> `nav/candata.py` 讀 `CAN_ROOT`（環境變數）找 table/datasplit；
> RunPod 上設 `export CAN_ROOT=/workspace/can_dataset_min`（目錄結構要保持
> `tcga_<cohort>/datasplit/fold_1.npz` 與 `tcga_<cohort>/table/*.csv`）。

tmux 內跑（先 EWC λ 選擇一次，再兩個 order 全跑；`run_main_cl.sh` 會自動分片並在結束時 push）：

```bash
tmux new -s main
cd /workspace/mllm_hwsi_ah && \
python scripts/cl_main.py --dataset can --select-ewc-lambda 2>&1 | tee results/ewc_select.log && \
bash scripts/run_main_cl.sh 8 2>&1 | tee results/main_cl_run.log && \
git pull --rebase && git add results/ figures/ && \
git commit -m "RunPod: main CL results (can, both orders)" && git push
```

## 5. Mac 端取回結果

```bash
cd /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah
git pull        # results/ figures/ 直接進來，交給 agent 分析
```

## 6. 成本紀律

- 每個實驗先在 Mac 跑 smoke（`--smoke` 小樣本模式），通過才上 RunPod。
- 跑完就關 Pod（Volume 保留資料）；不要掛機過夜除非 job 在跑。
- Frozen MLLM 推論（16GB 權重）只在 RunPod 做；nav/ 系列小模型 Mac 就能跑。
