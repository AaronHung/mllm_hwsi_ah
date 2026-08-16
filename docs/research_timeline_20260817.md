# 研究時間線 — 給老師/自己看的進度整理（2026-08-17）

> 這份文件回答三個問題：做了什麼、為什麼做、結果如何（成功/失敗/其他），以及這些跟論文的關係。
> 接續 `docs/research_timeline_20260816.md`。
>
> **寫作時點**：本文件主體寫於 2026-08-16 深夜（CST），當時 v0.34 development gate 正在跑、
> 尚未出 verdict；「結果」一節於 **2026-08-17 05:10 CST** 跑批結束後補寫，該節標題有註明。
> 日期慣例：文件日期用本機 CST 日曆日，run tag 用 UTC timestamp。

## 論文在問什麼問題（一句話版本）

病理醫生看巨幅玻片（WSI）不會整張看完，而是先看縮圖、挑幾個可疑區域放大細看，在有限「觀察預算」下決定
「該看哪裡」。我們發現：當這個「該看哪裡」的政策被拿去連續學習新任務時，會出現一種標準 CL 指標量不到的
遺忘——**navigation forgetting**。v0.34 之後，論文多了第二個問題：**要怎麼在保住舊任務證據可及性的同時，
不擋住新任務的學習？**

## 現況總覽（Track A / Track B）

**Track A（論文骨架）：** 8/21 freeze 四條件已於 8/16 全部達成，pilot40 R6 結案。剩下的是寫作——
Fable 接手把 `paper/main.tex` 改成 framework-first 脊椎（Fig.1 領頭 → 方法 → 評估 → 發現 → 附錄），
這件事在 v0.34 成功或失敗兩個世界裡內容完全相同，所以與 Track B 並行、零日曆浪費。

**Track B（v0.34 新方法 track）：** v0.33 正式封存，v0.34 已開案、跑完 development gate，**三個 config 全部 DEV FAIL**，方法算力工作到此結束（不開 v0.35）。

| 項目 | 8/16 傍晚 | 8/17 凌晨（本次） |
|---|---|---|
| v0.33.3 / Gate v2 verdict | 三候選全 FAIL；`eq_pres` 不 promote | **不變，永久凍結** |
| 方法 track | §10.5 寫死「最後一次方法補救運算」 | 該條款經三方 supersession amendment **正式取代**（見下） |
| v0.34 預註冊 | 不存在 | `docs/method_gate_v034.md` 已 commit（訓練前） |
| v0.34 實作 | 不存在 | gradient arbiter + violation weight + conflict 儀器，已 commit |
| Bit-exact 回歸 | — | **PASS，20/20 逐位相同** |
| Development gate | — | **36/36 跑完（5h03m），三個 config 全部 DEV FAIL** |
| 機制假說 | 未量測 | **儀器直接量測，不支持 gradient-conflict 解釋** |
| 下一個阻塞點 | 三方對 Gate v2 的 joint unblinding review（已完成） | **三方對 v0.34 dev verdict 的 focused review** |

## 為什麼會有 v0.34（決策鏈，別跳過）

Gate v2 的 FAIL 不是隨機失敗，它給了一個**具體、可修復、CL-specific 的診斷**：

> `eq_pres` 的 evidence utility 保存是真的（5/5 seed 一致變好），但它的 preservation gradient
> 會干擾新任務的 acquisition，代價集中在 `brca` K=2（pooled5 −0.0341）與 `lung` K=4（−0.0129），
> 外加 K=1 的遺忘沒達 parity。

三方裁決鏈（HackMD，越後面覆蓋前面）：Cursor 交出 v2 verdict → Fable ratify 並主張「材料夠了，只要
framework-first 重排寫法」→ **Sol 只同意一半**，指出老師仍會問「所以你的 CL method 是什麼」，主張
用這個診斷長出新方法 → Fable 採納並**修正自己原本寫死的止損線**（承認「禁止 eq_pres 搶救」是對的，
「禁止一切後續方法研究」是過度的）→ Sol 補 6 項 amendment → Fable final rider（r1–r4）收斂衝突 →
三方 SIGN OFF。

處理方式刻意光明正大：**用一條註明日期的 supersession amendment 正式取代該條款**，v2 FAIL 一字不動、
`eq_pres` 仍不 promote、v0.34 有自己的一次性預算與硬性日曆止損、且明文寫死**本篇 ICASSP 不會有 v0.35**。
老師看到的 audit trail 是「gate → 誠實 FAIL → 診斷 → 新假說 → 新預註冊」，這是研究的樣子。

## v0.34 是什麼（方法一句話 + 三件組）

> **Conflict-aware equivalence preservation**：保存「還能找到哪些有用證據」，而不是「以前怎麼走」；
> 而且只在真的開始失去可及性時才出手，並在它與新任務學習衝突時讓步。

1. **Evidence-equivalent memory** — `A_eps(s)`，eps=0.05（沿用 v0.33，不調）。
2. **Violation-aware preservation** — `v(s) = 1 − Σ_{a∈A_eps(s)} π(a|s)`，stop-gradient，
   只在 policy 真的跑出有用證據集合時才加大保存力道。
3. **Conflict-aware arbitration** — 每次 update 分開取 `g_n`（新任務）與 `g_m`（memory），
   若 `g_n·g_m < 0` 就把 `g_m` 裡與 `g_n` 反向的分量投影掉，再合成 `g_n + g_m`。

三個一次性 config（不加第四個）：`proj_distill`（generic control，回答「光靠投影有沒有用」）、
`proj_eq_pres`（proposed）、`conflict_eq_pres`（full）。`eq_pres` 用凍結資料當 diagnostic precursor，
不重跑、不重新評分。

**措辭鎖（rider r4，很重要）**：不准寫「Adam 下新任務學習永不被擋」——能嚴格說的只有 raw gradient
space 裡「移除一階反向分量」；plasticity 一律維持 empirical gate。Novelty 是三件組本體，
**gradient projection 是手段不是宣稱**，不寫 "reverse A-GEM"、不寫發明 gradient projection。

## 今晚做了什麼（8/16 深夜 – 8/17 清晨）

### 階段 9 — v0.34 開案 + 實作 + 開跑

- **做了什麼**：
  1. `docs/method_gate_v034.md` 完整預註冊並 commit（**訓練前**）：supersession amendment、
     sign-off chain 與優先序、方法規格、development gate 判準 A/B/C、機制連結、winner 選擇規則、
     硬止損、confirmation 分支、措辭鎖、comparator provenance。
  2. `docs/method_gate_v033.md` 只做兩處**顯式標註**的編輯：§9 changelog 加一筆指向 v0.34，
     §10 末加一行 provenance footnote（說明標題日期沿用簽核原文、實際跑批是 8/16 CST）。
     沒有任何判準、門檻、數字被動到。
  3. 實作進 `nav/cl.py`，全部 opt-in、預設維持舊行為。**凍結的 `loss = loss + …` 累加順序一行沒改**
     ——改順序就會破壞與凍結資料的 bit-exactness，分離只靠多存兩個 reference。
  4. 跑了一輪 bit-exact 回歸：用新程式碼重跑 `eq_pres` 與 `ours_uniform` seed0 K=1，
     對凍結列 **20/20 逐位相同**（`results/method_gate_v034_bitexact_regression.md`）。
  5. 36 units 開跑（Mac MPS、tmux + `caffeinate`、atomic resume）。
- **為什麼**：判準在看到任何數字之前寫死並 commit，這樣不管結果好壞都站得住腳。bit-exact 回歸是
  「證明」而不是「宣稱」凍結路徑沒被動到——這次加 arbiter 動到了梯度組裝，不驗證就開跑是不負責任的。
- **結果**：36/36 units 乾淨跑完（5h03m，exit 0），verdict 已自動產生並 STOP。見「結果」一節。

### PI 在開工前加的兩件事

1. **儀器化複製凍結的 `eq_pres`（9 units）**。rider r2 要求 dev verdict 必附診斷格的 conflict 表，
   但 `eq_pres` 依規定不重跑，只能拿 arbiter-ON 的新 config 當近似。PI 核准加一批
   **純儀器化**重跑：診斷梯度用 `autograd.grad(retain_graph=True)` 在同一個 forward graph 上取，
   **不多做 forward、不消耗 RNG、不動 optimizer state**，更新路徑逐位不變。條件寫死：
   此批次任何 metric row **一律不採信、不得改變任何 verdict**；且必須與凍結 `eq_pres` 列
   **bit-identical**，否則**整批作廢**、退回報 arbiter-ON 的近似值並附限制說明。排在 dev gate 之後跑。
2. **儀器 self-check**：前 2 個 unit 跑完後檢查 conflict 診斷有寫出且非退化（非全 NaN、非全 0）。
   這是**程式碼檢查，不是科學早停**——Sol amendment item 2 已經把「conflict ≈0 就早停」這條刪掉，
   理由是沒有任何可辯護的數值門檻，這裡不會偷偷把它加回來。

## Development gate 的判準（已凍結，摘要）

3-seed（{0,1,2}）paired mean，mean-only；per-seed 符號與 95% bootstrap CI 只作描述性揭露。

- **A — utility 保住**：Δ`eps_optimal_mass` > 0，vs `distill` **且** `ours_uniform`，K∈{1,2}。
- **B — targeted repair + no-new-failure**（rider r1 = Sol item 1）：
  - **B-a**：**每一個** (task, K) 格（12 格）ΔA[t,t] vs `ours_uniform` ≥ −0.01。
  - **B-b**：K=2、K=4 的 ΔForgetting ≤ +0.005 vs `ours_uniform` **且** `distill`。
  - **B-c**：診斷靶 `brca` K=2 與 `lung` K=4 必須修好；K=1 ΔForgetting ≤ 0 vs 兩個 comparator。
- **C**：Jaccard **無要求**（方法本來就不要求照走舊路線）；replay safety ΔForgetting vs `replay`
  ≤ +0.005，所有 K。
- **PASS = A ∧ B ∧ C**。機制連結是預註冊假說、事後陳述，**無數值早停**。

**Winner 規則**（Cursor 擬、開跑前 commit、三方可在讀數字前反對）：`proj_distill` 是 control 不是晉升
候選；兩個候選都過就取 A 軸較強者；若 `proj_distill` 也過 A，**必須明白揭露**——那是對 novelty 定位的
直接反證。

## 結果（2026-08-17 05:10 CST 補寫；跑批 36/36 乾淨完成，耗時 5h03m）

**三個 config 全部 DEV FAIL。** 完整逐格數字見 `results/method_gate_v034_dev_verdict.md`。
但這次的 FAIL 結構比 Gate v2 那次資訊量大很多，下面三點是給三方 review 的重點。

### 1. `proj_eq_pres` **確實把兩個被診斷的格子修好了**，而且 utility 優勢保住

| 格子 | `eq_pres`（凍結，v2） | `proj_eq_pres`（v0.34） |
|---|---|---|
| `brca` K=2 ΔA[t,t] | −0.0175（3-seed）／−0.0341（pooled5） | **+0.0088** |
| `lung` K=4 ΔA[t,t] | −0.0079（3-seed）／−0.0129（pooled5） | **+0.0177** |
| K=1 ΔForgetting vs `ours_uniform`／`distill` | 未達 parity | **−0.0079／−0.0133，PASS** |
| Δε-mass K=1 vs `ours_uniform`／`distill` | +0.0308／+0.0405 | **+0.0411／+0.0509（3/3 seeds）** |
| Δε-mass K=2 | — | **+0.0293／+0.0335（3/3 seeds）** |

也就是說 **criterion A PASS、criterion B-c（診斷靶）PASS**。介入本身在它瞄準的目標上是有效的。

### 2. 殺死它的是 Sol 加的那條 no-new-failure

`proj_eq_pres` 在**沒被診斷過**的 `brca` K=4 新開了一個洞：ΔA[t,t] = **−0.0331**
[+0.0000, −0.0729, −0.0263]——這格對 `eq_pres` 本來是好的（v2 pooled5 = +0.0012）。
另外 B-b 在 K=2／K=4 對 `distill` 沒過（+0.0159／+0.0057），C 的 replay safety 在 K=2 沒過（+0.0082）。

**這正是 Sol amendment item 1 存在的理由。** 如果照 Fable 原本的 S2-B（只要求修好已知的兩格 +
K=1 遺忘），`proj_eq_pres` 會**通過** development gate 並直接進 confirmation——然後我們會拿一個
「修好兩格、弄壞另一格」的方法去跑 reverse order。這條 amendment 直接擋下了一次假陽性。

### 3. 機制假說**沒有得到儀器支持**（這點最重要，也最需要三方裁定）

`eq_pres_diag`（凍結 `eq_pres` 的純儀器化複製，90/90 逐位相同、已 ADMITTED）量到的
conflict fraction：

| 格子 | conflict fraction | mean cos |
|---|---|---|
| `brca` K=2（診斷靶，失敗格） | 0.4913 | +0.0006 |
| `lung` K=4（診斷靶，失敗格） | 0.4988 | −0.0006 |
| `rcc` K=1（健康格） | **0.5039** | +0.0005 |
| `brca` K=1（健康格） | 0.4768 | +0.0053 |
| `lung` K=1（健康格） | 0.4580 | +0.0115 |

全部 config、全部 stage、全部預算的 conflict fraction 都落在 **0.458–0.504**，mean cosine 全部 ≈ 0。
**被診斷的失敗格跟健康格分不出來**——`rcc` K=1 這個完全沒問題的格子，conflict fraction 比兩個
失敗格都還高。這是兩個高維梯度之間「沒有系統性關係」時該有的樣子（符號各半、平均餘弦為零），
不是「preservation gradient 在特定階段有系統地對抗 new-task acquisition」該有的樣子。

依 rider r2，這條連結從一開始就登記為**事後陳述的假說、不設數值門檻、不中途停跑**，所以這裡只陳述
事實：**儀器沒有支持「gradient conflict 造成 Gate-v2 那兩格 plasticity 損失」這個解釋。**
要不要據此改寫論文裡的機制敘事，是三方 review 的裁定，不是這份文件的裁定。

值得一併記下的是：投影**確實有在作用**（約半數 update 觸發，`|proj|/|g_m|` 約 0.03–0.15），而且
確實把兩個靶格修好了。所以現在的狀況是「介入有效，但我們對它為什麼有效的解釋沒被證實」。

### 4. 附帶結論：utility 優勢來自 equivalence objective，不是來自 arbiter

generic control `proj_distill`（= `ours_uniform` + 同一個 arbiter）**criterion A 沒過**
（K=2 Δε-mass vs `ours_uniform` = −0.0028）。§3.4 item 5 的揭露條款因此**未觸發**——
如果它過了，那會是對 novelty 定位的直接反證。目前的資料指向相反方向：把 arbiter 加在 generic
distillation 上拿不到 utility 優勢，那個優勢是 `L_eq` 帶來的。

## 接下來的 scope

1. **(阻塞點，現在就在這裡)** dev verdict 已產生，交 Aaron / Sol / Fable 做 focused three-way
   review。**dev FAIL，所以 confirmation 分支不啟動。** review 要裁定的其實只剩兩件事：
   - 機制敘事怎麼改寫。儀器直接量到的結果不支持 gradient-conflict 解釋（見上方結果第 3 點），
     但介入確實修好了兩個靶格。論文要怎麼誠實描述「有效但機制未證實」，需要三方定調。
   - `proj_eq_pres` 的 A + B-c 通過、B-a 新開一洞這個結構，在附錄裡要寫成什麼樣子。
   **review 前不做 promotion、不改任何論文宣稱。**
2. **confirmation 分支不啟動**（僅存檔備查）：原本規劃是 winner config、reverse order、
   **明確 seeds {0,1,2,3,4}**、K∈{1,2,4}、Mac MPS，措辭一律用 "previously unseen v0.34
   reverse-order confirmation runs"（**禁用** "fresh seeds"——那會害人跑去 seeds 5–9，直接失去
   與凍結 reverse comparator 的配對）。
3. **執行 dev FAIL 分支**：誠實封存。論文走 framework-first，`eq_pres` 當 diagnostic precursor，
   三個 v0.34 config 連同 per-seed 表寫進附錄標記 tested-negative（跟 `ia_samp`/`ia_ep` 一樣）。
   **不開 v0.35。** 所有方法算力工作到此結束，日曆上比 8/22 science freeze 提前五天。
4. Track A 並行寫作：`paper/main.tex` framework-first 重排、`\xx{}` 佔位數字換成最終數字、
   跑 `scripts/check_forbidden_phrases.py`。
5. Protocol 既定關卡：Gate 2'（8/22）、Gate 3'（8/24）。
6. Science freeze **8/22**（絕對上限 8/23）。

## 怎麼監看 / 怎麼接手

```bash
cd ~/research/01_mllm_hwsi/mllm_hwsi_ah
tmux attach -t gate_v034                       # 看即時進度（Ctrl-b d 離開）
bash scripts/watch_run.sh <tag>                # 或看 checkpoints.json + log
```

斷線、關機、跑到一半停掉都不用怕：runner 是 atomic-resume，重下同一條指令會從
`checkpoints.json` 記錄的最後完成 unit 接續，不會重跑已完成的部分。跑批全程需要
**筆電開蓋 + 接電源**（`caffeinate -i` 只擋 idle sleep，擋不了闔蓋）。

## 值得跟老師特別強調的四點

1. 到目前為止抓到的每一個公式/設計漏洞都是在正式跑實驗**之前**被紅隊抓到，沒有浪費算力。
2. 每一版 gate 的判準都是**看到結果之前**寫死的，跑完**沒有回頭改任何門檻**。
3. v0.34 不是 p-hacking：v0.33 的 FAIL 一字不動、`eq_pres` 不 promote，新開的是一條由**自己的失敗
   診斷**推導出來的新假說，而且用註明日期的 supersession amendment 公開處理，不是靜默改舊條款。
4. 這一輪第一次有「架構圖上畫得出來、且每個箱子後面都有表格」的 CL training architecture：
   evidence-equivalent memory → violation-aware preservation → conflict-aware arbitration →
   shared budgeted navigator。inference 端沒有增加任何複雜度，還是同一個 navigator。
