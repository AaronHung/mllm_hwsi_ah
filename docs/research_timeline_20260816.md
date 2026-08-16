# 研究時間線 — 給老師/自己看的進度整理（2026-08-16）

> 這份文件回答三個問題：做了什麼、為什麼做、結果如何（成功/失敗/其他），以及這些跟論文的關係。
> 對應的互動版本是一個 Cursor Canvas（`~/.cursor/projects/.../canvases/research-timeline.canvas.tsx`），
> 但那個檔案不在 git 追蹤範圍內、只能在 Cursor IDE 裡打開，所以另外存這一份 Markdown 到 `docs/`，
> 跟著 repo 走、隨時可以搜尋到。內容截至 2026-08-16 10:40 (UTC+8)，接續 `docs/research_timeline_20260815.md`。
>
> 同一天也存了一份 `.canvas.tsx` 原始碼快照到 `docs/research_timeline_20260816.canvas.tsx`——
> 那份的排版更好讀，但要複製回 Cursor 的 canvas 資料夾才能重新渲染，檔案開頭有寫怎麼做。

## 論文在問什麼問題（一句話版本）

病理醫生看巨幅玻片（WSI）不會整張看完，而是先看縮圖、挑幾個可疑區域放大細看，在有限「觀察預算」下決定
「該看哪裡」。我們發現：當這個「該看哪裡」的政策（policy）被拿去連續學習新任務（continual learning）時，
會出現一種標準 CL 指標量不到的遺忘——選擇行為早就崩壞了（selection Jaccard 從 0.79 掉到 0.07），但因為
影像內證據有冗餘，凍結的預測頭還能靠「矇對」撐住準確率，直到預算被壓得很緊，準確率才跟著垮。這就是
**navigation forgetting**，也是整篇論文要提出並解決的問題。詳見 `paper/main.tex` 摘要與 Introduction。

## 現況總覽（跟昨天相比的變化）

| 項目 | 昨天（8/15） | 今天（8/16） |
|---|---|---|
| Mac MPS Method Gate | 16 / 33 units 進行中 | **33 / 33 units 跑完**，g1–g4 verdict 已計算 |
| Gate 結果 | 未知 | **M1 / M2 / M3 三個候選全部 GATE FAIL**（M2 最接近，見下） |
| RunPod CUDA pilot40 主網格 | push 待確認 | **已確認 push 到 `origin/main`** |
| 紅隊事前抓到的公式/設計 bug | 2 | 2（不變） |
| 下一個阻塞點 | 等 gate 跑完 | **等紅隊（Sol、Fable）joint unblinding review** |

## 今天的新進展

### 階段 7 — Method Gate 跑完 + verdict 計算（8/16）— 已完成

- **做了什麼**：Mac MPS 上的 33 單位 method gate（M1/M2/M3 三個主 config × 3 seeds × K∈{1,2,4}，加兩個
  mini-arm 診斷單位）全部跑完。寫了 `scripts/method_gate_verdict.py`，把 `docs/method_gate_v033.md`
  v0.33.3 裡凍結的 g1–g4 判準逐字套用到跑完的結果上（跟凍結的舊對照組資料、已通過 checksum 的 MPS
  `util_regen` 數字比對），**沒有「看到數字後才決定門檻」**。
- **為什麼**：Gate 存在的目的就是要在看到結果之前把「贏的標準」講清楚，這樣不管結果好壞都站得住腳，
  不會被質疑事後合理化。
- **結果**：
  - **三個候選（M1/M2/M3）全部 GATE FAIL。**
  - **M2（`eq_pres`）最接近過關**：g2（utility 指標 `eps_optimal_mass`/`normalized_regret` 變好，vs 兩個
    對照組都是 3/3 seeds 一致）與 g3（不輸 `replay`）都乾淨通過；但 g1（只在 K=1 的遺忘打平沒過，K=2、K=4
    都過）與 g4（`brca` 最後任務的新任務學習力，K=1、K=2 都沒過）沒過。
  - **額外做了「單一 seed 敏感度」診斷**：18 個 FAIL 子項裡有 **15 個是「3 個 seed 裡只有 1 個把平均值
    拖過門檻」**，不是 3/3 一致的訊號——例如 `eq_pres` 在 K=1 的 `brca` own-time accuracy，3 個 seed 裡有
    2 個跟 `ours_uniform` 完全 bit-identical，只有第 3 個 seed 拉出差距。這不會反轉已經凍結的 pre-registered
    判定，但是解讀「gate fail 代表什麼」時很重要的線索——3 個 seed、每個任務只有 15 筆測試資料，統計
    power 本來就薄。
  - **誠實揭露一個資料缺口**：凍結的 `ours_uniform` 對照組資料從來沒有在 K=4 跑過（Protocol-v1 五個 seed
    全部只有 K∈{1,2}），K=4 的比較誠實標成 **N/A**，沒有為了填洞去偷跑新的對照組訓練（那等於是 unblinding
    後又動 protocol）。
- **跟 paper 的關係**：如果紅隊覆核後維持 FAIL，論文誠實走「分析為主 + 附錄報告新方法嘗試」路線：M2 在
  utility 指標上有真實、3/3 一致的提升，換來一個邊界性、可能被小樣本雜訊放大的遺忘/新任務學習力代價——
  這本身是一個可報告的 trade-off，不是方法失敗。完整逐格數字見 `results/method_gate_v0333_verdict.md`；
  這次 verdict 已寫入 `docs/method_gate_v033.md` 的 changelog，明確標記是 Cursor 做的分析、還沒經過紅隊
  審核，也還沒選定補救方向。

### 附帶完成：確認 pilot40 已 push 到 `origin/main`

昨天結尾時 RunPod 端的 `git push` 卡在 GitHub 不再接受密碼登入（需要 Personal Access Token）跟 PAT 權限
設定的問題，今天已排除：教 Aaron 生成有 `Contents: Read and write` 權限的 PAT、用
`git remote set-url` 安全地嵌入 token、並在 `scripts/runpod_bootstrap.sh` 裡加了
`git credential.helper` 設定，讓 token 存在 `/workspace`（不會被容器重置清掉），以後重開 pod 不用再重新
輸入。RunPod 與 Mac 兩邊現在都跟 `origin/main` 同步。

## 過程中順手解決的工程坑（跟科學結果無關，純基礎設施，8/15 累積至今）

| 問題 | 根本原因 | 怎麼解決 |
|---|---|---|
| RunPod 終端機顏色分不清指令跟輸出 | 預設 prompt 太陽春 | 客製 PS1（黃色帳號＋青色路徑＋git branch，指令另起一行），寫進 bootstrap script 永久生效 |
| pilot40 訓練找不到資料檔（FileNotFoundError） | `data/` 在 `.gitignore` 內，從未上傳過 RunPod | 小檔用 scp、5.1GB 大資料夾用 rsync 手動傳，checksum 驗證 |
| tmux 內 Ctrl-b d 離開沒反應 | 網頁版終端機把組合鍵攔截給瀏覽器 | 改用純文字指令 `tmux detach-client` |
| `rsync: command not found` | RunPod 精簡映像預設沒裝 rsync | 手動 apt-get install，並把自動安裝寫進 bootstrap script |
| rsync 傳輸時一堆 `chown Operation not permitted` 警告 | 網路硬碟（Network Volume）不允許改檔案 owner | 確認只是 metadata 警告、內容仍完整傳輸，忽略即可 |
| 全網格「跑 1 分鐘就結束」，看似正常實則全滅 | 新開的 tmux pane 沒 source 到 `env.sh`，PATH 撿到缺套件的系統 Python | 規定「新 tmux pane 一律先 source env.sh 再 export 變數」，寫入 `RUNPOD_SOP.md` 固定坑位段落 |
| `git push` 要求輸入 Username/Password | GitHub 2021 起不接受帳密做 git-over-HTTPS | 生成有 `Contents: Read and write` 權限的 PAT，取代密碼 |
| PAT 生成後 push 仍 403 Permission denied | PAT 預設沒勾 repo 的 `Contents: Read and write` | 到 PAT 設定頁補勾該權限 |
| RunPod 容器重置後要重新輸入 PAT | 預設 credential cache 存在 `$HOME`，容器重置會清掉 | `git credential.helper` 指到 `/workspace` 下的持久檔案，寫進 bootstrap script |

## 接下來的 scope：還要跑什麼、寫什麼

1. **(阻塞點)** 把 `results/method_gate_v0333_verdict.md`（含 seed-sensitivity 診斷）交給紅隊
   （Sol、Fable）做一次 joint unblinding review。
2. 紅隊覆核後三選一：(a) 補測更多 seed 看單一 outlier 是否洗掉、(b) 補跑 `ours_uniform` 在 K=4 的
   probe-only 對照、(c) 直接誠實把 M2 的 utility-axis 提升 vs. 邊界性遺忘/新任務代價寫進論文附錄。
3. 若決定補測並最終通過 → 擴大到 5 seeds×{main,reverse}×K∈{1,2,4} 的 promotion run（deadline 8/23），
   並決定方法的正式公開名稱。
4. 若維持 FAIL（目前狀態）→ 不重跑、不硬凹，主線維持「分析為主」版本，Track B 嘗試過程寫進論文附錄
   （這是 protocol 裡預先承認的有效結果，不算失敗）。
5. pilot40 主表（`main_table_pilot40_main.md`）已產生，接下來拿來跟 can_dataset 的主表對照分析，
   餵進 Gate 1'（8/18，learned policy 是否顯著贏 random）。
6. Protocol 既定的論文級三道關卡：Gate 2'（8/22，seqft 遺忘是否顯著）、Gate 3'（8/24，ours 是否顯著贏
   seqft）。
7. Track A 收尾雜項：`paper/main.tex` 內 KL 方向文字、`docs/handoff...` 文件的 cell 編號、補齊 mechanism
   robustness 指標。
8. 把 paper 草稿裡目前的 `\xx{}` 佔位數字換成最終數字，跑一次 `scripts/check_forbidden_phrases.py`，
   準備投稿 ICASSP 2027。

## 值得跟老師特別強調的三點

1. 到目前為止抓到的每一個公式/設計漏洞（M1 退化、backend 混淆）都是在正式跑實驗**之前**被紅隊抓到，
   沒有浪費算力，也不需要事後丟棄結果重跑。
2. Method gate 的判準是**看到結果之前**寫死的，跑完之後**沒有回頭改任何門檻**——即使三個候選都沒過，
   這個「沒過」的結論本身是可信的，不是選過門檻湊出來的。
3. Track B（新方法 M1/M2/M3）就算最後沒有一個配置通過 gate，也是 protocol 裡**預先承認的有效結果**，
   論文會誠實在附錄報告，不算失敗——這是實驗設計時就講好的規則，不是事後找台階下。額外做的
   seed-sensitivity 診斷顯示大部分 FAIL 是單一 seed 造成的邊界情況，值得紅隊一起判斷這代表「方法真的
   有問題」還是「3 個 seed 對這種效應量太少」，再決定要不要花小成本補測。
