# RCC K16 三提示推論交付說明

此套件只新增 RCC K16 三提示實驗。執行前必須已備妥既有的 `mllm_hwsi` 環境、模型 checkpoints、K16 特徵張量、CellViT 張量，以及位於完整路徑 `/home/alan0804/CL/Navigator/pilot20_actual_selection.csv` 的 selection 檔案。ZIP **不包含**模型權重或病患資料。

將 `rcc_k16_three_prompt_inference_bundle.zip` 上傳到 `/home/alan0804/CL/Navigator/mllm_hwsi_demo/MLLM-HWSI` 後，執行：

```bash
conda activate mllm_hwsi
cd /home/alan0804/CL/Navigator/mllm_hwsi_demo/MLLM-HWSI
unzip -o rcc_k16_three_prompt_inference_bundle.zip
chmod +x run_rcc_pilot20_k16.sh
sha256sum -c SHA256SUMS.txt
python -m unittest discover -s Ours/tests -v
bash -n run_rcc_pilot20_k16.sh
```

## 凍結提示

三個正式英文 prompt 必須保持原樣：

- `P0_open`: `This whole-slide image is from a renal tumor. What is the most likely histologic subtype? Answer with one diagnosis only.`
- `P1_clear_first`: `This whole-slide image is from a renal tumor. Classify it as either clear cell renal cell carcinoma or papillary renal cell carcinoma. Answer with one diagnosis only.`
- `P2_papillary_first`: `This whole-slide image is from a renal tumor. Classify it as either papillary renal cell carcinoma or clear cell renal cell carcinoma. Answer with one diagnosis only.`

## 執行順序

先執行 preflight：

```bash
bash run_rcc_pilot20_k16.sh --preflight-only
```

再執行單張 slide 的 smoke test：

```bash
set -o pipefail
bash run_rcc_pilot20_k16.sh --limit 1 2>&1 | \
  tee outputs/rcc_pilot20_40x/inference_k16_three_prompt/smoke.log
```

最後執行完整實驗：

```bash
set -o pipefail
bash run_rcc_pilot20_k16.sh 2>&1 | \
  tee outputs/rcc_pilot20_40x/inference_k16_three_prompt/inference.log
```

## 預期里程碑

- Preflight 必須回報正好 20 張 slides、`KIRC=10`、`KIRP=10`、沒有 feature errors，且不載入模型。
- Smoke test 必須為一張 slide 寫入正好三筆成功紀錄。完整指令可安全續跑，並略過這三筆成功紀錄。
- 完整推論必須寫入正好 60 筆屬於目前實驗、成功且 key 唯一的紀錄。
- 輸出位於 `outputs/rcc_pilot20_40x/inference_k16_three_prompt/`。
- `P0`、`P1`、`P2` 必須分開計分；`P0` 不參與投票，也不作為平手裁決。
- 續跑時，符合目前協定的 `ok` keys 會略過；`error` keys、舊協定或 metadata 不符的 keys 會重試。原始 JSONL 保留 audit 紀錄，CSV 與 summary 只發布目前協定的有效成功紀錄。

## 疑難排解

- **匯入失敗：**確認套件解壓縮在 repository root，並透過 `python -m` 或 launcher 執行。
- **CUDA/checkpoint 失敗：**在目標 Linux repository root 執行下列獨立診斷；不要刪除 features：

```bash
python - <<'PY'
import gc
import torch
from mllm_hwsi import MLLMHWSI

model, loading_info = MLLMHWSI.from_pretrained(
    "checkpoints/mllm_hwsi",
    device="cuda:0",
    dtype=torch.float16,
    local_files_only=True,
    strict_projector=True,
    return_loading_info=True,
)
model.eval()
print(f"Loading info: {loading_info}")
print(f"GPU: {torch.cuda.get_device_name(torch.device('cuda:0'))}")
print(f"LLM dtype: {next(model.llm.parameters()).dtype}")
print(f"Projector dtype: {next(model.vl_projector.parameters()).dtype}")
del model
gc.collect()
torch.cuda.empty_cache()
PY
```

- **推論中斷：**重新執行相同的完整指令；成功 keys 會略過，error keys 會重試。
- **最終 exit code 非零：**檢查 missing-key 訊息與 `predictions.jsonl` 中的 error 紀錄。

上方診斷、Linux `bash -n`、CUDA/checkpoint 載入，以及所有實際推論指令都尚未在此 Windows 打包環境執行；必須在目標 Linux 環境完成後才能宣稱通過。
