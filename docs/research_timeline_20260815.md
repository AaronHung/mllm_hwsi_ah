# 研究時間線 — 給老師/自己看的進度整理（2026-08-15）

> 這份文件回答三個問題：做了什麼、為什麼做、結果如何（成功/失敗/其他），以及這些跟論文的關係。
> 對應的互動版本是一個 Cursor Canvas（`~/.cursor/projects/.../canvases/research-timeline.canvas.tsx`），
> 但那個檔案不在 git 追蹤範圍內、只能在 Cursor IDE 裡打開，所以另外存這一份 Markdown 到 `docs/`，
> 跟著 repo 走、隨時可以搜尋到。內容截至 2026-08-15 21:32 (UTC+8)。
>
> 同一天也存了一份 `.canvas.tsx` 原始碼快照到 `docs/research_timeline_20260815.canvas.tsx`——
> 那份的排版更好讀，但要複製回 Cursor 的 canvas 資料夾才能重新渲染，檔案開頭有寫怎麼做。

## 論文在問什麼問題（一句話版本）

病理醫生看巨幅玻片（WSI）不會整張看完，而是先看縮圖、挑幾個可疑區域放大細看，在有限「觀察預算」下決定
「該看哪裡」。我們發現：當這個「該看哪裡」的政策（policy）被拿去連續學習新任務（continual learning）時，
會出現一種標準 CL 指標量不到的遺忘——選擇行為早就崩壞了（selection Jaccard 從 0.79 掉到 0.07），但因為
影像內證據有冗餘，凍結的預測頭還能靠「矇對」撐住準確率，直到預算被壓得很緊，準確率才跟著垮。這就是
**navigation forgetting**，也是整篇論文要提出並解決的問題。詳見 `paper/main.tex` 摘要與 Introduction。

## 現況總覽

| 項目 | 數字 |
|---|---|
| Mac MPS Method Gate 進度 | 16 / 33 units 完成 |
| RunPod CUDA pilot40 主網格 | 訓練已完成、已在 pod 上 commit；**push 到 origin/main 尚待確認**（5 seeds × 7 methods × K∈{1,2,4}） |
| 紅隊事前抓到的公式/設計 bug | 2（M1 退化、backend 混淆風險） |
| 待決新方法 | M1 / M2 / M3（Track B，pass/fail 由 gate 決定） |
| 資料集 | can_dataset（主表，4 任務）+ pilot40（第二資料集，2 任務，真實座標） |

## 時間線：從 Protocol 凍結到現在

### 階段 0 — Protocol v1 凍結（8/14）— 已完成

- **做了什麼**：定死主實驗的全部定義：can_dataset 四任務序列（ESCA→LUNG→RCC→BRCA）、7 個對比方法
  （seqft/ewc/lwf/replay/distill/ours_uniform/ours）、觀察預算 K∈{1,2,4}、5 個 seed、指標（AA、forgetting、
  Jaccard、action-KL、sel_utility）與統計政策（paired bootstrap CI、BH-FDR）。
- **為什麼**：沒有一份凍結的 protocol，後面任何實驗都可能被質疑「跑完才調定義」，先把科學定義釘死才能
  開始正式產生可信結果。
- **結果**：成功，凍結版存在 `docs/protocol.md`。這一批跑出來的數字就是 paper 草稿目前引用的主要數字
  （例如 Jaccard 0.79→0.07）。
- **跟 paper 的關係**：整篇論文的地基——navigation forgetting 這個現象、以及 7 個方法互相比較的主表，
  全部從這裡的 protocol 產生。

### 階段 1 — Direction freeze v0.32 / v0.292（8/14–15）— 已完成

- **做了什麼**：把論文敘事定稿：問題定名「navigation forgetting」、環境定名「causal feature pyramid」、
  統計呈現規則、以及一份「禁止誇大用語」清單（例如不能講 universal winner）。
- **為什麼**：確保之後所有寫作與紅隊審查用同一套詞彙與同一套統計呈現標準，避免論文口吻前後不一致或
  被審稿人抓到過度宣稱。
- **結果**：成功，寫入 `docs/research_contract_v0.292.md`，之後所有文件都以此為準。
- **跟 paper 的關係**：決定了 paper 摘要/intro 現在的措辭方式，以及統計呈現的格式規範。

### 階段 2 — v0.33 Two-Track 分工（8/15）— 已完成

- **做了什麼**：把剩下的工作拆成兩條並行的線：**Track A**＝把既有主線收尾（統計腳本補列、文件修正、
  第二資料集 pilot40、禁語自動檢查腳本）；**Track B**＝在既有 7 個方法之外，嘗試 3 個新方法 M1/M2/M3，
  用一個預先登記（pre-registered）的「gate」流程決定它們值不值得寫進論文。
- **為什麼**：Track A 是把已經確定要用的結果做完、做穩；Track B 是有風險的新嘗試，用 gate 機制先講好
  「贏的標準」，避免看到結果後才回頭改標準（對抗事後合理化）。
- **結果**：成功，分工方式定案。
- **跟 paper 的關係**：Track A 決定論文能不能準時交件；Track B 決定論文最後用不用得上一個「更好的方法」，
  或是維持「分析為主、誠實報告沒找到更好方法」的版本。

### 階段 3 — v0.33.1：M1 公式退化 bug（紅隊抓到）— 已完成

- **做了什麼**：紅隊審查者 Sol 發現 M1（重要性回放取樣）的原始公式有個數學漏洞：某一項在每個狀態上都
  必然等於 1，等於整個方法悄悄退化成「只看遺忘程度取樣」，跟設計初衷（重要 × 被遺忘）不符。
- **為什麼**：這種退化如果沒抓到，跑出來的「M1 有效」結論其實是另一個更簡單方法的結果，論文會下錯結論。
- **結果**：成功且及時——在任何 gate 訓練開始「之前」就抓到並修正，沒有浪費任何計算資源。
- **跟 paper 的關係**：保護了 Track B 新方法比較的正確性。

### 階段 4 — v0.33.2：M1 校準修正 + 意外發現 backend 差異 — 已完成

- **做了什麼**：把 M1 的重要性分數改成「在自己來源任務內部」算百分位排名（而不是跨任務一起排），避免
  不同任務的 evaluator 尺度差異被誤讀成「這個狀態比較重要」。同時補齊新的評估指標（ε-optimal mass、
  normalized regret）、真正的模型 checkpoint 存檔、per-task 取樣診斷。過程中依指示在 Mac CPU 上重跑舊
  方法核對數字，結果對不上凍結的舊資料（誤差最大到 0.148）。
- **為什麼**：先前一版定義（跨任務全域排名）會讓「evaluator 尺度不同」偽裝成「狀態比較重要」，是另一種
  隱藏的公式風險。核對數字則是要確保新程式碼沒有不小心改掉舊結果的計算方式。
- **結果**：修正成功；另外意外查出：當年凍結的主表數字其實是在 **Mac MPS**（不是 CPU）上產生的。用
  MPS 重跑後 12/12 組完全對上（誤差 0.0000）。這件事也記錄成一個小發現：同一份程式碼在 CPU 和 MPS 上
  會因為浮點運算順序不同而跑出不同結果（逐步 argmax 模擬常見現象，不是程式錯誤）。
- **跟 paper 的關係**：避免 M1 的貢獻被「跨任務 evaluator 尺度差異」污染；CPU/MPS 差異會寫成一條
  reproducibility 附註，屬於誠實揭露的工程細節。

### 階段 5 — v0.33.3：gate 全部改在 MPS 上跑 — 已完成

- **做了什麼**：因為上一階段發現 backend 會影響數字，原本「新方法在 RunPod CUDA 跑、拿 Mac MPS 的舊
  結果來比較」的計畫改掉：整個 gate（新方法 + 對照組）統一在 Mac MPS 上跑到底；pilot40 的資料量體工作
  維持在 RunPod CUDA（跟方法比較無關）。另外寫了 `docs/compute_policy.md` 把「什麼工作該用哪個 backend」
  的規則定下來。
- **為什麼**：如果新方法在 CUDA 跑、對照組在 MPS 跑，最後看到的「差異」有可能其實是硬體造成的，不是
  方法本身造成的——會混淆「方法效果」跟「backend 效果」，是實驗設計上的大忌。
- **結果**：成功，避免了一個可能讓整個 Track B 結論失效的混淆變因。
- **跟 paper 的關係**：保證「M1/M2/M3 比舊方法好或不好」這句話講的是方法本身，不是硬體巧合。

### 階段 6 — RunPod CUDA：pilot40 主網格 — 已完成（8/15 晚間）

- **做了什麼**：在有「真實空間座標」的第二個資料集（腎臟/乳房兩任務）上，跑同一套 7 個既有方法
  （5 seeds × 7 methods × K∈{1,2,4}）。中途踩過兩個坑：① 新開的 tmux pane 沒繼承裝好套件的 Python
  環境（`env.sh` 沒 source），導致 10 個訓練工作瞬間因缺 pandas 而崩潰、卻誤以為「跑完了」；② 修好之後
  重跑，5 個 shard 全部乾淨完成。
- **為什麼**：驗證 navigation forgetting 這個現象不是 can_dataset 偽座標環境的產物，是否在真實空間結構
  的資料上一樣成立。
- **結果**：成功。315 筆結果（5 seeds × 7 methods × 3 K × 3 stage-rows = 315，數字對得上），無任何
  Traceback，`main_table_pilot40_main.md` 與兩張圖已產生（run tag `v2_pilot40_main_cuda_20260815T122732Z`）。
  已在 pod 上 `git commit`；**`git push` 到 `origin/main` 這一步截至本文件時間尚未在遠端看到**，
  需要 Aaron 確認 push 是否成功、再到 Mac 端 `git pull` 驗證。
- **跟 paper 的關係**：提供第二資料集的主表，用來支持「navigation forgetting 及其緩解方法可以跨資料集
  重現」這個 generalization 主張。

### 正在跑（截至本文件時間）

- **Mac MPS — Track B Method Gate**：33 個單位（M1 單獨、M2 單獨、M3=M1+M2、對照 mini-arms，各 3 seeds ×
  K∈{1,2,4}），目前 **16/33** 完成，平均每單位約 10–20 分鐘（原估計 3.7 小時偏樂觀，實際預估總時長
  8–10 小時，仍遠低於 48 小時上限）。跑完會依照預先登記的 4 條判準（g1 遺忘不變差、g2 至少一項變好、
  g3 不輸 replay、g4 新任務學習力不掉超過 0.01）逐一檢查，決定 M1/M2/M3 能否進論文。

## 過程中順手解決的工程坑（跟科學結果無關，純基礎設施）

| 問題 | 根本原因 | 怎麼解決 |
|---|---|---|
| RunPod 終端機顏色分不清指令跟輸出 | 預設 prompt 太陽春 | 客製 PS1（黃色帳號＋青色路徑＋git branch，指令另起一行），寫進 bootstrap script 永久生效 |
| pilot40 訓練找不到資料檔（FileNotFoundError） | `data/` 在 `.gitignore` 內，從未上傳過 RunPod | 小檔用 scp、5.1GB 大資料夾用 rsync 手動傳，checksum 驗證 |
| tmux 內 Ctrl-b d 離開沒反應 | 網頁版終端機把組合鍵攔截給瀏覽器 | 改用純文字指令 `tmux detach-client` |
| `rsync: command not found` | RunPod 精簡映像預設沒裝 rsync | 手動 apt-get install，並把自動安裝寫進 bootstrap script |
| rsync 傳輸時一堆 `chown Operation not permitted` 警告 | 網路硬碟（Network Volume）不允許改檔案 owner | 確認只是 metadata 警告、內容仍完整傳輸，忽略即可 |
| 全網格「跑 1 分鐘就結束」，看似正常實則全滅 | 新開的 tmux pane 沒 source 到 `env.sh`，PATH 撿到缺套件的系統 Python | 規定「新 tmux pane 一律先 source env.sh 再 export 變數」，寫入 `RUNPOD_SOP.md` 固定坑位段落 |

## 接下來的 scope：還要跑什麼、寫什麼

1. **(進行中)** 跑完 Mac MPS 上剩餘的 method gate 單位（目前 16/33）。
2. **(進行中)** 確認 pilot40 結果已從 RunPod push 到 `origin/main`，並在 Mac 這邊 `git pull` 驗證。
3. 整理 `results/method_gate_verdict.md`，與紅隊（Sol、Fable）做一次 joint unblinding review。
4. 若 M1/M2/M3 通過 gate → 擴大到 5 seeds×{main,reverse}×K∈{1,2,4} 的 promotion run（deadline 8/23），
   並在 8/18 前決定方法的正式公開名稱。
5. 若沒有任何配置通過 gate → 不重跑、不硬凹，誠實把嘗試過程寫進論文附錄，主線維持「分析為主」的版本
   （這是 protocol 裡預先承認的有效結果，不算失敗）。
6. pilot40 結果已經跑完，接下來把 `main_table_pilot40_main.md` 拿來跟 can_dataset 的主表對照分析，
   餵進 Gate 1'（8/18，learned policy 是否顯著贏 random）。
7. Protocol 既定的論文級三道關卡：Gate 2'（8/22，seqft 遺忘是否顯著）、Gate 3'（8/24，ours 是否顯著贏
   seqft）。
8. Track A 收尾雜項：`paper/main.tex` 內 KL 方向文字、`docs/handoff...` 文件的 cell 編號、補齊 mechanism
   robustness 指標。
9. 把 paper 草稿裡目前的 `\xx{}` 佔位數字換成最終數字，跑一次 `scripts/check_forbidden_phrases.py`，
   準備投稿 ICASSP 2027。

## 值得跟老師特別強調的兩點

1. 到目前為止抓到的每一個公式/設計漏洞（M1 退化、backend 混淆）都是在正式跑實驗**之前**被紅隊抓到，
   沒有浪費算力，也不需要事後丟棄結果重跑。
2. Track B（新方法 M1/M2/M3）就算最後沒有一個配置通過 gate，也是 protocol 裡**預先承認的有效結果**，
   論文會誠實在附錄報告，不算失敗——這是實驗設計時就講好的規則，不是事後找台階下。
