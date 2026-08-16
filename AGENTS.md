# AGENTS.md — 給任何 coding agent 的冷啟動說明

這份檔案的目的：**讓任何 coding agent（Cursor Agent、Claude Code、Codex CLI…）在完全沒有先前
對話記憶的情況下，讀完這份檔案 + 下面列的幾份 docs，就能接手這個專案，不用重新問一次「誰是誰、
規則是什麼」。** 專案本身刻意用「先寫死規則、commit 之後才跑實驗」的紀律運作，所以幾乎所有重要
決策都已經是 repo 裡的檔案，不是鎖在某一個 agent 的對話 session 裡。

`CLAUDE.md`（Claude Code 會自動讀取）只是 `@AGENTS.md` 匯入這份檔案，兩者保持同步，不要分開改。

## 專案一句話

MLLM + Whole Slide Imaging (WSI) 導航策略的 continual learning 遺忘問題（"navigation forgetting"），
投稿 ICASSP 2027。詳見 `paper/main.tex` 摘要，或任一份 `docs/research_timeline_*.md` 的開頭。

## 誰是誰（動 Track B / gate 結果之前一定要先懂這段）

- **Aaron** — PI/使用者，實際操作 Cursor IDE、RunPod、下所有最終決策的人。
- **Sol、Fable** — **獨立於這個 coding agent 之外**的紅隊審查者（對應到不同的 LLM，例如
  `gpt-5.6-sol-xhigh` / `claude-fable-5-thinking-max` 這類命名），由 Aaron 拿 coding agent 產生的
  pre-registration 文件跟 verdict 去問他們，再把他們的裁定帶回來。**他們的審查流程完全不依賴
  「這個對話 session 記得什麼」**——所以換 coding agent（Cursor Agent → Claude Code）完全不影響
  紅隊審查這條線，只要新 agent 一樣把每輪決策寫回 `docs/method_gate_v033.md` 的 changelog 就好。
- **這個 coding agent**（不論是哪個工具）— 負責執行跑批、寫文件、依照凍結的判準算 verdict，
  **絕不能自己一邊寫判準一邊改判準**，也絕不能對已經「讀過結果」的 gate 章節做事後靜默編輯。

## Pre-registration 紀律（動任何 gate / 判準相關工作之前一定要先懂）

- 任何 gate 的判準（`docs/method_gate_v033.md`、`docs/protocol.md`）在訓練開始「之前」就要寫死、
  commit；結果讀出來之後，那個版本的判準永久凍結，**不能回頭重新解釋、不能因為結果不好看就改門檻**。
- 之後想追加/延伸，一律開新的、有日期、有版本編號的獨立章節或 changelog 條目，絕不靜默改舊章節。
- 每一輪紅隊裁定（不管是同意、駁回、或要求延伸驗證）都要整段寫進 `docs/method_gate_v033.md` 的
  changelog，標明是誰裁定的、日期、以及這是不是這個 coding agent 自己的分析（尚待紅隊複核）。

## Compute backend 政策

`docs/compute_policy.md` 是權威版本。一句話摘要：**同一個 method-vs-method 比較的所有跑批，必須全部
在同一個 backend 上完成**（例如全部 Mac MPS，或全部 RunPod CUDA），因為這個專案的 argmax-driven
navigator 在 CPU/MPS 之間會有真實的浮點運算路徑差異（不是雜訊），混用 backend 會把「方法效果」跟
「硬體效果」搞混。pilot40 這種跟方法比較無關的資料量體工作可以用 RunPod CUDA。

## RunPod / tmux / logging 慣例

`docs/RUNPOD_SOP.md` 是權威版本（含持續累積的踩坑記錄）。一句話摘要：
- 新開的 tmux pane 一律先 `source env.sh` 再 export 任何變數，否則會撿到系統 Python、缺套件但看起來
  「跑完了」（一分鐘就結束的全滅假象）。
- 長跑批一律用 atomic-resume runner（例如 `scripts/run_gate_v2.py`），斷線/重開後可以直接重跑同一條
  指令，會自動從 `checkpoints.json` 記錄的最後完成單位繼續，不會重跑已完成的部分。
- 監看進度用 `bash scripts/watch_run.sh <tag>` 或直接讀對應的 `checkpoints.json`，不要單靠 tail log
  檔（有些 runner 的 log 檔位置跟預期不同）。
- 大檔案（資料集）用 rsync + checksum 手動搬，不要假設 RunPod pod 會自動有東西（`data/` 在
  `.gitignore` 內，從未上傳過）。

## 現在的狀態要看哪份文件

**不要**假設這份 AGENTS.md 本身有最新進度（它只放不太會變的規則）。要看目前狀態，永遠去找
`docs/research_timeline_*.md` 裡日期最新的那份（用 `ls docs/research_timeline_*.md` 排序確認），
裡面有「今天做了什麼／為什麼／結果／跟論文的關係」＋一份會持續更新的「接下來的 scope」清單。
截至寫這份 AGENTS.md 時，最新的是 `docs/research_timeline_20260816.md`。

其他關鍵文件：
- `docs/method_gate_v033.md` — 完整的 gate 歷史 + changelog（每一輪紅隊裁定、verdict、退役紀錄）。
- `results/method_gate_v0333_verdict.md`、`results/method_gate_v2_verdict.md` — 已算好的 verdict 表。
- `docs/research_contract_v0.292.md` — 論文敘事定稿、統計呈現規則、禁止誇大用語清單。
- `docs/protocol.md` — 主實驗的凍結定義（資料集、方法、指標、統計政策）。

## 資料夾地圖

- `scripts/` — 跑批 runner + 分析腳本（`run_gate_v2.py`、`method_gate_v2_verdict.py`、
  `check_forbidden_phrases.py`、`watch_run.sh`…）。
- `docs/` — 凍結的 protocol/policy 文件 + 每天的 timeline snapshot。
- `results/` — 算好的表格、verdict、checksum 報告。
- `runs/` — 跑批輸出（checkpoint `.pt` 被 `.gitignore` 排除，csv/json/md 有進版控）。
- `paper/` — ICASSP 2027 投稿草稿（`main.tex`）。

## 論文措辭鎖

`scripts/check_forbidden_phrases.py` 會擋掉一批被紅隊要求禁止的用語（例如 "universal winner"、
"masking confirmed"）。改任何論文文字之前跑一次這支腳本。

## 如果這份檔案漏了什麼

完整的過去對話記錄（含這份文件寫成之前的所有討論細節）存在 Cursor 的 agent-transcripts JSONL 裡
（不在這個 repo 內，是本機 Cursor 的資料，路徑因機器而異）。那份是最後手段，用來查「某個特定決策當
初的完整脈絡」，不是給新 agent 冷啟動讀的——內容太大、格式是逐行 JSON、且不含工具呼叫細節。正常情況
下，這份 AGENTS.md ＋ 上面列的幾份 docs 應該就足夠讓任何 agent 接手。
