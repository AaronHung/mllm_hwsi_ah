# 研究時間線 — 給老師/自己看的進度整理（2026-08-16）

> 這份文件回答三個問題：做了什麼、為什麼做、結果如何（成功/失敗/其他），以及這些跟論文的關係。
> 對應的互動版本是一個 Cursor Canvas（`~/.cursor/projects/.../canvases/research-timeline.canvas.tsx`），
> 但那個檔案不在 git 追蹤範圍內、只能在 Cursor IDE 裡打開，所以另外存這一份 Markdown 到 `docs/`，
> 跟著 repo 走、隨時可以搜尋到。內容截至 2026-08-16 17:55 (UTC+8)（本文件當天內第二次更新，涵蓋
> Method Gate v0.33.3 verdict + 紅隊裁定 + Gate v2 延伸驗證的完整結果），接續
> `docs/research_timeline_20260815.md`。
>
> 同一天也存了一份 `.canvas.tsx` 原始碼快照到 `docs/research_timeline_20260816.canvas.tsx`——
> 那份的排版更好讀，但要複製回 Cursor 的 canvas 資料夾才能重新渲染，檔案開頭有寫怎麼做。

## 論文在問什麼問題（一句話版本）

病理醫生看巨幅玻片（WSI）不會整張看完，而是先看縮圖、挑幾個可疑區域放大細看，在有限「觀察預算」下決定
「該看哪裡」。我們發現：當這個「該看哪裡」的政策（policy）被拿去連續學習新任務（continual learning）時，
會出現一種標準 CL 指標量不到的遺忘——選擇行為早就崩壞了（selection Jaccard 從 0.79 掉到 0.07），但因為
影像內證據有冗餘，凍結的預測頭還能靠「矇對」撐住準確率，直到預算被壓得很緊，準確率才跟著垮。這就是
**navigation forgetting**，也是整篇論文要提出並解決的問題。詳見 `paper/main.tex` 摘要與 Introduction。

## 現況總覽（Track A / Track B 兩行狀態，今天內的變化）

**Track A（論文骨架）：** 8/21 freeze 四條件今日已全部達成（pilot40 R6 已結案，行為層複現、capability
判定鎖定「在這個小型設定下未有定論」）；下一步是寫作填數字。
**Track B（新方法 M1/M2/M3）：** 上午跑完 v0.33.3 gate、三候選全 FAIL；下午經 Sol+Fable 兩輪紅隊裁定，
M1/M3 正式退役，M2（`eq_pres`）跑完最後一次、預先登記的 5-seed 延伸驗證（Gate v2）——**結果仍是 FAIL**，
`eq_pres` 定案為論文分析章節的介入式證據，不再是候選方法。

| 項目 | 上午（8/16 早上） | 傍晚（8/16 17:55，本次更新） |
|---|---|---|
| Mac MPS Method Gate（v0.33.3） | 33 / 33 units 跑完，g1–g4 verdict 已計算 | 不變，**判定凍結，不再重新解讀** |
| Gate 結果 | M1/M2/M3 三個候選全部 GATE FAIL | 不變；已由 Sol+Fable 兩輪紅隊覆核，裁定 M1/M3 退役、M2 給一次延伸驗證 |
| Gate v2（`eq_pres` 專屬延伸） | 尚未開始 | **19 / 19 units 跑完（2h19m），verdict 已計算：GATE v2 FAIL** |
| pilot40 G1' 顯著性 | 8/14 舊資料 | 8/16 用新 5-seed 主網格重跑，**bit-identical，PASS 不變** |
| 下一個阻塞點 | 等紅隊 joint unblinding review（v0.33.3） | **等三方（Aaron/Sol/Fable）對 Gate v2 verdict 做 joint unblinding review** |

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

### 階段 8 — 紅隊裁定 + Gate v2 延伸驗證（8/16 下午）— 已完成

- **做了什麼**：把階段 7 的 verdict 交給紅隊後，Sol、Fable 兩輪覆核裁定：`ia_samp`（M1）、`ia_ep`（M3）
  正式退役，不再追；`eq_pres`（M2）獲得一次、也是**最後一次**的預先登記延伸驗證「Gate v2」——加 2 個新
  seed（3、4）到 `eq_pres`（K∈{1,2,4}），並把 utility 對照組 `ours_uniform` 補跑到 K=4（Protocol-v1 原本
  從未在任何 seed 跑過這個組合）。跑法完全比照原本 gate 的紀律：先在 `docs/method_gate_v033.md` §10 寫死
  新判準並 commit，才開始訓練；19 個訓練單位、Mac MPS、`caffeinate`、atomic-resume；另外自行加了一步
  「utility-axis 回填」（distill/`ours_uniform` 在新 seed 上補算兩個 v0.33.2 才有的指標欄位），這一步不是
  新實驗，是把已經凍結的舊數字重新跑一次來補欄位，一樣要通過 `|Δ|<=0.001` checksum 才採用。19 個單位
  全部跑完，耗時 2 小時 19 分，checksum 8/8 逐位元對上。
- **為什麼**：原本 3-seed 的 gate 是刻意從小規模開始的「篩選」，`eq_pres` 在 utility 指標上的訊號很一致，
  但遺忘與新任務學習力的邊界被少數幾格高變異的資料點決定——多兩個 seed 是要看這個訊號撐不撐得住統計力，
  不是「考不好再考一次」：判準、門檻、報告格式全部在跑之前寫死，且明講「就算過也不能反過來說原本的
  gate 判斷錯了」。
- **結果：GATE v2 仍然 FAIL。**
  - **好消息**：utility 指標（`eps_optimal_mass`↑、`normalized_regret`↓）這次不只 3/3，是**5/5 個 seed
    全部一致變好**，比原本的訊號更強、更值得信任，vs 兩個對照組（`ours_uniform`、`distill`）在 K=1、K=2
    都成立。
  - **壞消息**：K=1 的遺忘打平（g1）仍沒過，跟原本一樣；而且新增的 2 個 seed 讓兩個原本邊緣過關的格子
    也跟著沒過——`lung` K=4（原本 3-seed 平均 −0.0079，剛好壓在 −0.01 門檻內；加 2 個新 seed 後變成
    −0.0129，過線）與 `brca` K=2（原本 −0.0175；新 2 個 seed 平均到 −0.0590，且**兩個新 seed 各自**都遠遠
    超過門檻，其中一個差距是該任務量測解析度的 9 倍，不是單一測試樣本翻面的雜訊）。
  - 這代表「多加 seed」確實讓訊號更清楚，只是清楚的方向是**「這個代價是真的」**，不是「原本的擔心是
    雜訊，加了 seed 就會洗掉」——延伸驗證做了它該做的事：把一個 3-seed 螢幕篩選不出來的真訊號看清楚，
    不管方向是好是壞。
  - 額外檢查了一條「複製確認」的揭露規則（如果平均過關但兩個新 seed 各自都指向失敗方向，不能算穩健
    確認）：這次沒有任何一格觸發這條規則——每一個 PASS 的格子，兩個新 seed 至少有一個本身也是通過的。
- **跟 paper 的關係**：`eq_pres`（M2）正式定案——**不會被寫成論文的新方法**，而是誠實寫進分析章節的
  介入式證據：改一個 loss 讓 policy 不用照抄舊路徑，換到有效證據保存力的真實提升，但要付出新任務學習力
  （尤其最後一個、最難的任務 `brca`）的真實代價。這正是老師想看到的「機制型」發現，只是不是「我們提出
  了一個更好的方法」的敘事——這一點會在跟老師報告時特別說明。完整逐格數字見
  `results/method_gate_v2_verdict.md`，checksum 明細見 `results/util_regen_checksum_v2_seeds34.md`，兩者
  都已寫入 `docs/method_gate_v033.md` 的 changelog，標記為 Cursor 分析、待三方 joint unblinding review。

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

1. **(阻塞點)** 把 `results/method_gate_v2_verdict.md`（含 checksum 報告）交給 Aaron / Sol / Fable
   做一次**三方** joint unblinding review。Gate v2 是預先講好的**最後一次**方法補救運算——不管三方
   review 的結論是什麼，都不會再加 seed、不會再調參、不會再開 Gate v3。
2. 三方覆核通過後，把 `eq_pres`（M2）的介入式證據（utility 提升 vs. capability/plasticity 代價）正式
   寫進論文分析章節；`ia_samp`（M1）、`ia_ep`（M3）連同 per-seed 表格寫進附錄，標記為 tested-negative。
3. pilot40 sel-utility 稽核已完成：`trajectory_utility` 在 `can_dataset`／pilot40 兩邊的定義、正負號、
   聚合方式完全一致，pilot40 出現的負值是誠實的量測結果，不做美化解讀，可直接引用進論文。
4. pilot40 主表（`main_table_pilot40_main.md`）已完成並驗證（G1' 顯著性用新 5-seed 主網格 8/16 重驗，
   bit-identical PASS）；capability 結論鎖定寫成「在這個小型雙任務設定下未有定論，與 masking 假說一致
   但未驗證」，禁止寫成「已證實」一類的用語（已加進 `scripts/check_forbidden_phrases.py`）。
5. Protocol 既定的論文級三道關卡：Gate 2'（8/22，seqft 遺忘是否顯著）、Gate 3'（8/24，ours 是否顯著贏
   seqft）。
6. Track A 收尾雜項：`paper/main.tex` 內 KL 方向文字、`docs/handoff...` 文件的 cell 編號、補齊 mechanism
   robustness 指標。
7. 把 paper 草稿裡目前的 `\xx{}` 佔位數字換成最終數字，跑一次 `scripts/check_forbidden_phrases.py`，
   準備投稿 ICASSP 2027。

## 值得跟老師特別強調的三點

1. 到目前為止抓到的每一個公式/設計漏洞（M1 退化、backend 混淆）都是在正式跑實驗**之前**被紅隊抓到，
   沒有浪費算力，也不需要事後丟棄結果重跑。
2. Method gate（含 Gate v2 延伸）的判準都是**看到結果之前**寫死的，跑完之後**沒有回頭改任何門檻**——
   即使 M2 加測 2 個 seed 後依然沒過，這個「沒過」的結論本身是可信的，不是選過門檻湊出來的；而且這次
   延伸驗證證明了不是統計力不足造成的誤判：utility 訊號變得更強（3/3→5/5），capability/plasticity 的
   代價也變得更清楚（兩個原本邊緣的格子確定沒過）——兩個方向都被「看得更清楚」，不是被雜訊洗掉。
3. Track B（新方法 M1/M2/M3）就算最後沒有一個配置通過 gate，也是 protocol 裡**預先承認的有效結果**，
   論文會誠實在分析章節/附錄報告，不算失敗——這是實驗設計時就講好的規則，不是事後找台階下。`eq_pres`
   換來的「utility 真的變好、但新任務學習力有真代價」本身就是一個值得寫的機制性發現，只是它的定位是
   「我們理解了為什麼會有這個 trade-off」，不是「我們解決了這個 trade-off」。
