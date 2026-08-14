# VLN Survey 與 WSI 適用性分析

> 目的：survey 機器人領域 Vision-and-Language Navigation（VLN），評估哪些方法可引入 WSI navigation continual learning，哪些不可引入並給出確定原因。
> 撰寫日期：2026-08-14。所有 venue／年份／數字均經網路搜尋驗證；無法驗證處標註「待查證」。
> 我們的設定（對照基準）：feature-space causal pyramid（HIPT 192-d 低倍 region 全可見；CONCH 512-d 高倍 patch 需 zoom 才揭露）、單一 <1M 參數 shared navigator、固定預算 K 步、無醫師軌跡、以 counterfactual diagnostic gain 產生 teacher 分佈做 KL 模仿、順序學多診斷任務時出現 navigation forgetting（selection Jaccard 崩至 0.06）。

**目錄**
1. TL;DR
2. VLN 問題定式（含 2.1 benchmark 對照表）
3. 代表方法表（15 篇 + DUET 詳讀）
4. 訓練範式比較（IL / DAgger / RL / pretraining / zero-shot LLM）
5. VLN → WSI 對應分析（5.1 逐項對應、5.2 可借清單、5.3 不可搬清單+確定原因、5.4 預答質疑）
6. Continual VLN 現況與我們的空缺
7. 對論文的具體建議（7.1 related work 段落、7.2 引用清單、7.3 實驗優先序）
8. 參考文獻（31 篇）

---

## 1. TL;DR

1. VLN 的核心方法論：在 panorama connectivity graph 上做 instruction-conditioned 序列決策，靠三支柱——(i) 人類示範軌跡的 imitation learning（+DAgger／RL 微調）、(ii) cross-modal transformer 與大規模 pretraining、(iii) 顯式記憶（recurrent state → full history → topological map）。
2. 最可借的三個機制：DUET 的 coarse（全域 topological map）/fine（局部 egocentric）雙尺度決策與 global action space；HAMT／CVLN-ESR 式的顯式 history／logit memory（對應我們的 evidence memory 與 compressed state replay）；DAgger／pseudo interactive demonstrator——我們的 frozen evaluator 恰好是一個可任意查詢的 interactive oracle，比 VLN 的人類示範更適合 on-policy 蒸餾。
3. 指標可移植：SPL 的「success × 效率」結構移植為 accuracy-per-budget；nDTW/CLS 的 path-fidelity 思想對應我們的 selection Jaccard／trajectory 一致性。
4. 不可直接搬的根本原因（非算力）：(a) VLN 監督來自 ground-truth 人類示範路徑，WSI 無醫師 viewing 軌跡且大規模收集已被 PathAgentBench 證明成本極高；(b) VLN 的 success 是幾何定義（3m 內），WSI 的 success 只能由 frozen evaluator 的診斷結果間接定義，且該定義隨任務改變——這正是我們 forgetting 的來源，VLN 問題結構中不存在；(c) panorama 是同尺度空間移動、觀測不改變 information set，WSI zoom 是跨尺度資訊揭露 action（active perception），空間位移先驗無對應物；(d) 語言 instruction 是 per-episode 且逐步對齊，我們的 diagnostic query 是 per-task 常數，cross-modal grounding 模組無用武之地。
5. Continual VLN 已存在（CVLN BMVC 2025、VLNCL IROS 2025、M³E、TuKA 2026）但全部是 scene-domain incremental：技能不變、只換環境、每個 domain 都有 ground-truth path 可 replay。任務級（診斷語義級）navigation continual learning、且監督為弱監督 counterfactual teacher 的設定，目前沒有先例——這是我們的空缺證據。
6. 我們的定位：把 VLN 的「決策結構」（雙尺度、記憶、預算內探索）借過來，把 VLN 的「監督結構」（示範軌跡、幾何 success）整個換掉；continual 軸上做 VLN 社群尚未定義的 task-incremental navigation。

---

## 2. VLN 問題定式（標準定義）

VLN 標準設定（Anderson et al., CVPR 2018；Fried et al., NeurIPS 2018 之後的主流版本）：

- **環境**：Matterport3D Simulator，90 棟真實建築的 panoramic RGB-D，離散化為 navigation graph（節點 = viewpoint，邊 = 可通行連接）。Episode 為 POMDP：state = (位置, heading, elevation)，觀測為當前節點的 panorama。
- **Observation space**：當前 viewpoint 的 360° panorama，標準離散化為 36 個 view（12 headings × 3 elevations，30° 間隔），每個 view 以 ConvNet／ViT 特徵 + 相對角度編碼 (sinθ, cosθ, sinφ, cosφ) 表示。REVERIE 額外提供 object bounding box 特徵。
- **Action space** 三代演進：
  1. Low-level visuomotor（R2R 原版）：turn-left/right、up/down、forward、stop。
  2. Panoramic action space（Fried et al. 2018 起成為標準）：直接從「navigable directions」（鄰接節點）中選一個跳過去，或選 stop。
  3. Global action space（DUET）：從 on-the-fly topological map 上「所有已發現的 navigable node」中選目標，執行時用 Floyd 最短路徑走過去——一次決策即可跨多步回溯。
  4. Continuous（VLN-CE、ETPNav）：連續空間中的 low-level 控制或 waypoint 預測。
- **Supervision**：每條 instruction 附帶 ground-truth trajectory（人類示範或最短路徑）。訓練用 teacher forcing（在 GT 狀態上監督 GT action）、student forcing（在自身 rollout 狀態上以最短路徑 oracle 給監督，即 DAgger 式）、RL reward（到 goal 的距離縮減 + success bonus）。
- **Memory**：問題是 partially observable，記憶機制歷經：LSTM recurrent state → 單一 state token 遞迴更新（Recurrent VLN-BERT）→ 全 history 序列 transformer 編碼（HAMT）→ 顯式 topological map（DUET、ETPNav）→ video token 序列（NaVid）→ 語言化 map 放進 prompt（MapGPT）。
- **Metrics**（精確定義，出處見 §8）：
  - **TL**（Trajectory Length）：平均軌跡長度（m）。
  - **NE**（Navigation Error）：終點與 goal 的最短路徑（geodesic）距離。
  - **SR**（Success Rate）：NE < 3m（R2R 慣例）之比例。純幾何、二值。
  - **OSR**（Oracle SR）：若軌跡上任一點停下都算，衡量「路過但沒停」。
  - **SPL**（Anderson et al., arXiv 1807.06757）：SPL = (1/N) Σᵢ Sᵢ · lᵢ / max(pᵢ, lᵢ)，Sᵢ 為 success 指示、lᵢ 為最短路徑長、pᵢ 為實走路徑長。同時懲罰失敗與繞路。
  - **nDTW**（Ilharco et al., NeurIPS ViGIL Workshop 2019）：預測軌跡與參考軌跡的 normalized Dynamic Time Warping 相似度，衡量 path fidelity 而非只看終點；**SDTW** = success-gated nDTW。
  - **CLS**（Jain et al., ACL 2019）：CLS = PC（path coverage，對參考路徑節點的覆蓋）× LS（length score，實走長度對「覆蓋所需期望長度」的懲罰）。與 SPL 平行但以覆蓋取代 success。
  - REVERIE 另有 **RGS/RGSPL**（remote grounding success 及其路徑長度懲罰版）。

### 2.1 Benchmark 對照表

| Benchmark | Venue | Instruction 類型 | 成功判準 | 規模 | 對我們的參考價值 |
|---|---|---|---|---|---|
| R2R | CVPR 2018 | 步級細粒度（平均 29 詞，"walk out of the bedroom, turn right..."） | 停在 goal 3m 內 | 21,567 instructions / 7,189 paths / 90 建築 | 標準設定的原型；path 皆為最短路徑（後被 R4R/RxR 批評有 bias） |
| RxR | EMNLP 2020 | 步級、多語（en/hi/te）、逐字對齊 pose trace | 3m 內 | 126K instructions / 16.5K guide paths | 「示範軌跡 + 注視軌跡」監督的極致——反襯 WSI 什麼都沒有 |
| REVERIE | CVPR 2020 | **高階目標式**（平均 18 詞，"water the plant on the table"） | 導航到位 + 圈出目標 object bounding box | 21,702 instructions | 與我們最像的監督粒度：無步級指引，須自行探索；證明 high-level query 足以條件化 policy |
| SOON | CVPR 2021 | 場景描述式（object 屬性+關係+區域描述），任意起點 | 找到目標 object | FAO dataset | goal-oriented 探索的另一例 |
| R4R | ACL 2019（隨 CLS 提出） | 兩條 R2R path 串接（非最短路徑） | 3m 內 + path fidelity 指標 | 由 R2R 重組 | 證明「終點對≠走對路」，行為忠實度需獨立度量——對應我們 Jaccard vs accuracy 的區分 |
| VLN-CE / R2R-CE / RxR-CE | —（Habitat 上重建） | 同 R2R/RxR | 連續空間判準 | 由離散版轉換 | 離散 graph ↔ 連續控制的轉換研究；我們的 pyramid 永遠是離散的，無此議題 |

---

## 3. 代表方法表

| 方法 | Venue+年份 | Observation | Action space | Supervision | Memory 機制 | 關鍵 insight |
|---|---|---|---|---|---|---|
| Seq2Seq baseline（R2R） | CVPR 2018 | 單向 60° 視野 CNN 特徵 | low-level（turn/forward/stop） | teacher/student forcing on GT path | LSTM hidden state | 定義了 VLN 任務；student forcing 已優於 teacher forcing |
| Speaker-Follower | NeurIPS 2018 | 36-view panorama | panoramic（選 navigable direction） | GT path IL + speaker 合成增強資料 + pragmatic rerank | LSTM | speaker model 反向合成 instruction 做資料增強；panoramic action space 讓決策粒度貼近語言粒度（test unseen SR 53.5%） |
| RCM | CVPR 2019 | panorama | panoramic | RL（extrinsic 目標距離 + matching critic 的 intrinsic 跨模態對齊 reward）+ SIL | LSTM | 把「軌跡是否能重建 instruction」當 reward，減輕 sparse reward（細節待查證） |
| EnvDrop | NAACL 2019 | panorama | panoramic | 混合 IL + A2C RL；back-translation + environmental dropout | LSTM | 對「環境特徵」整體 dropout 模擬 unseen 環境，配合 speaker 生成新 (env, path, instruction) triplet；unseen 泛化大幅提升 |
| PREVALENT | CVPR 2020 | panorama | panoramic | image-text-action triplet 自監督預訓練（masked LM + action prediction）→ 下游微調 | LSTM（下游） | 第一個 VLN pretraining 範式；R2R SPL 47→51，可遷移到 CVDN/HANNA |
| Recurrent VLN-BERT | CVPR 2021 | panorama（+ REVERIE objects） | panoramic | GT path IL + RL 混合 | 單一 state token 遞迴自更新 | 用 [CLS] token 當 recurrent state，讓 V&L BERT 直接當 policy；免去外掛 memory 模組 |
| HAMT | NeurIPS 2021 | 全部歷史 panorama | panoramic | proxy task 預訓練（單步 action 預測、空間關係預測等）+ RL 微調 | hierarchical history：ViT per view → panorama 內空間 transformer → 跨時間 transformer | 把「全程 history」顯式編碼進決策，長軌跡任務（R4R、R2R-Back）增益最大 |
| **DUET** | CVPR 2022 | 當前節點 fine-grained（view+object）+ 全圖 coarse node 特徵 | **global**：topological map 上所有 navigable node ∪ stop | behavior cloning + MLM/MRC auxiliary 預訓練 + **pseudo interactive demonstrator**（PID，DAgger 式）微調 | on-the-fly topological map（visited / navigable / current 三類節點；GASA graph-aware self-attention 把節點距離矩陣加入 attention） | 見下方詳讀框 |
| ETPNav | TPAMI 2024 | RGB-D（連續環境） | 預測 waypoint → 低階避障控制 | GT path IL（+輔助） | online topological map（waypoint 自組織） | 把 VLN-CE 分解為「拓撲圖上高階規劃 + 低階控制」；R2R-CE +10%、RxR-CE +20% |
| NavGPT | AAAI 2024 | 觀測轉文字描述 | panoramic（zero-shot 選 direction） | 無訓練；GPT-4 zero-shot reasoning | 文字化 navigation history | 證明 LLM 能顯式做 sub-goal 分解／landmark 辨識／進度追蹤，但 zero-shot SR 遠低於 trained models |
| DiscussNav | ICRA 2024 | 觀測轉文字 | panoramic（zero-shot） | 無訓練；multi-expert LLM 討論後決策 | 文字歷史 | 把 instruction 理解／感知／完成度估計拆給多個 expert model 討論，優於單模型 self-thinking |
| MapGPT | ACL 2024 | 觀測轉文字 + **語言化 topological map** 進 prompt | map 上節點（global，zero-shot） | 無訓練 | online 語言化 map + 迭代更新的 multi-step plan | 把 DUET 的「地圖+全域行動」思想搬進 prompt 空間；zero-shot R2R/REVERIE SOTA（SR 約 +10%/+12%） |
| NavGPT-2 | ECCV 2024 | 多張 view 圖像（InstructBLIP 視覺對齊） | panoramic + topological graph 回溯 | 凍結 LLM，訓練 Q-former 與 policy head；GPT-4V 合成的逐步 reasoning 資料做 visual instruction tuning | topological graph + VLM latent | 折衷路線：VLM latent 同時解碼語言推理與 action，資料效率高，追平 VLN 專用模型 |
| NaVid | RSS 2024 | 單目 RGB **video stream**（無 map/depth/odometry） | 連續 low-level action | 510k 導航樣本 + 763k web 資料的 video VLM 微調 | video token 時空上下文 | 用影片時序 token 隱式承載 history；Sim2Real 遷移強 |
| ABot-N1 | arXiv 2607.10383（2026，venue 待查證） | 多任務視覺輸入 | pixel-goal anchor → 連續 waypoint | slow-fast 架構（慢速 CoT reasoner + 快速 action expert） | reasoner 的顯式語言 trace | 2026 年 VLN foundation model 路線代表：以 pixel goal 作「認知↔控制」通用介面 |

**DUET 詳讀（與我們 coarse-to-fine 最相關）**：
- Topological map 三類節點：visited（有完整 panorama 表徵）、navigable（未探索、僅由鄰近 visited 節點部分觀測）、current。每步把新發現節點加入圖中並更新表徵——與我們「低倍 region 全可見、高倍證據選了才揭露」的 partial observability 結構同型（他們的 navigable node 是「部分觀測、待探索」，我們的未 zoom region 是「低倍可見、高倍未揭露」）。
- Coarse-scale encoder：對全圖節點做 graph-aware cross-modal attention（GASA：把節點間 pairwise 距離矩陣線性投影後加進 self-attention logits），輸出每個 navigable node 的分數——長程探索與回溯。
- Fine-scale encoder：只看當前節點的 view/object 細粒度特徵，輸出 local action 分數——精確 grounding 與 stop 判斷。
- Dynamic fusion：可學權重把 fine 分數轉換到 global 空間（把經過 visited 鄰居的分數彙總成 backtrack 分數）後與 coarse 分數加權融合。實測：任務初期與收尾倚重 fine（權重約 0.64/0.58），中段探索倚重 coarse（0.45）；dynamic 比 average fusion SPL +1.79。
- 消融結論：coarse 單獨用探索強（OSR/SPL 高）但停不準；fine 單獨用停得準但探索差——**兩尺度互補是實證結論，不只是直覺**。
- 訓練：behavior cloning（LSAP 單步 action 預測）+ MLM/MRC + object grounding 預訓練，再用 pseudo interactive demonstrator（以最短路徑演算法在 agent 自己 rollout 到的狀態上即時給 oracle action，DAgger 式）微調——**PID 實測優於 RL 微調**。
- 代價：map-based global action 天然帶來回頭路，R4R 上 nDTW 等 path-fidelity 指標反而輸給 local-action 方法——「探索效率」與「軌跡忠實度」是不同目標，指標選擇必須跟著任務語義走。

**從方法表讀出的三條演化主線**（寫 related work 時可用的敘事骨架）：
1. **Action space 粒度不斷抬高**：low-level 控制 → panoramic 方向選擇 → topological map 上的 global 節點選擇。每次抬高都是為了「讓決策粒度貼近監督訊號的粒度」。我們的 region-level 選擇天然就在最高粒度，這條演化對我們是已完成式。
2. **Memory 從隱式到顯式**：LSTM 壓縮向量 →（HAMT 批評其資訊損失）→ 全 history transformer → 結構化地圖。VLN 社群用五年時間確認「顯式、結構化的記憶優於壓縮向量」——我們現在的 evidence mean pooling 正處在被批評的那一代，升級路徑清晰。
3. **監督從純示範走向混合**：teacher forcing → +student forcing/DAgger → +RL → +pretraining → LLM zero-shot（放棄任務內訓練）。每一步都在對抗同一個敵人：exposure bias 與示範資料的有限性。我們因為 teacher 可計算（不依賴示範），可以直接跳到這條線的最優解（on-policy 蒸餾）而跳過 RL。

---

## 4. 訓練範式比較

| 範式 | 代表 | 需要什麼 | 優點 | 缺點／適用性 |
|---|---|---|---|---|
| Teacher forcing IL | R2R baseline、各 transformer 方法的 BC 階段 | GT 軌跡 | 穩定、可並行 | exposure bias：測試時 distribution shift |
| Student forcing / DAgger | R2R baseline（student forcing）、DUET 的 PID；理論根基 Ross et al., AISTATS 2011 | **可在任意狀態查詢的 oracle**（VLN 用最短路徑演算法） | 直接修正 on-policy 狀態分佈；DUET 實測 PID > RL | 需要 queryable expert，VLN 靠 simulator 最短路徑才有 |
| RL 微調 | RCM（policy gradient + intrinsic reward）、EnvDrop（IL+A2C 混合）、HAMT（A2C 微調） | reward 函數 + 大量 rollout | 能優化不可微指標（SR/SPL） | sparse reward 難訓；VLN 中通常只當 IL 之後的微調，且 DUET 顯示可被 PID 取代 |
| Pretraining | PREVALENT（image-text-action triplet）、VLN-BERT、HAMT（proxy tasks）、DUET（BC+MLM/MRC） | 大規模對齊語料（合成 instruction 常用 speaker model） | unseen 泛化的最大單一來源之一 | 依賴 language-vision-action 三元對齊資料——這是 WSI 沒有的結構 |
| 資料增強 | Speaker-Follower（合成 instruction）、EnvDrop（環境 dropout + back-translation） | 生成模型 + simulator 重渲染能力 | 便宜地擴大 (env, path, instruction) 分佈 | 依賴環境可重渲染／可擾動；預抽取特徵環境不可用 |
| Zero-shot LLM | NavGPT、DiscussNav、MapGPT | 觀測可語言化 + 強 LLM | 無需訓練、可解釋 | 效能低於 trained specialist（NavGPT 自承）；每步推理成本高 |
| VLM 微調 | NavGPT-2、NaVid | 中等規模導航資料 + 凍結 LLM | 縮小 zero-shot 與 specialist 差距 | 仍需示範軌跡當監督 |

對我們的 takeaway：VLN 的經驗排序大致是「pretraining ≫ DAgger 式 on-policy 監督 > RL 微調 > 純 teacher forcing」。我們已經在做 KL 模仿 teacher 分佈（≈ soft teacher forcing）；VLN 證據顯示下一步最值得做的是 **on-policy 蒸餾**（在 navigator 自己選出的 evidence set E 上重算 gain teacher 再蒸餾），而不是引入 RL——而且我們的 teacher 是 frozen evaluator 算出來的，天生就是 DAgger 需要的 queryable oracle，這一點比 VLN 還便利（VLN 的人類示範不可查詢，只能靠最短路徑代理）。

---

## 5. VLN → WSI 對應分析（核心）

### 5.1 逐項對應表

| VLN 元素 | WSI navigation 對應物 | 對應品質 |
|---|---|---|
| panorama connectivity graph（viewpoint 節點 + 可通行邊） | causal region pyramid（低倍 region 節點 + parent-child zoom 邊） | 結構同型但語義不同：VLN 的邊是空間位移，我們的邊是尺度揭露 |
| navigable node（未探索、部分觀測） | 未 zoom 的候選 region（低倍特徵可見、高倍證據未揭露） | 高度對應——DUET 的 partial observability 處理可借 |
| language instruction（per-episode、步級對齊） | diagnostic query／task identity（per-task 常數，無步級結構） | 弱對應：語義條件存在，但沒有序列對齊 |
| human demonstration path（每條 instruction 附 GT 軌跡） | **缺**——以 counterfactual gain teacher 分佈替代 | 不對應：這是最大的結構差異 |
| simulator geometric reward（距 goal 距離） | frozen evaluator 的 CE loss 改變量（gain） | 功能對應但性質不同：幾何 reward 是 task-agnostic 常數場，gain 是 task-dependent 且隨 evidence set E 變動 |
| stop action（決定 success 的位置） | 固定預算 K 步、不學 stop | 設計上迴避了 VLN 中 stop 這個著名難點（DUET 消融顯示 fine encoder 主要價值之一就是停得準） |
| SR（3m 內幾何 success） | slide-level 診斷正確率 | 需重定義（見 5.3-b） |
| SPL（success × 路徑效率） | accuracy-per-budget（正確率 × 預算效率） | 結構可移植（見 5.2-3） |
| nDTW/CLS（對參考軌跡的忠實度） | selection Jaccard／對 teacher top-K 集合的覆蓋率 | 可移植：我們的 Jaccard 本質是 set 版 path fidelity |
| topological map memory（DUET） | 已 zoom region 的 evidence memory（目前為 mean pooling） | 可升級方向（見 5.2-2） |
| environmental dropout 資料增強 | 特徵層面的 region dropout／augmentation | 部分可移植：無法重渲染，但可在特徵空間擾動 |

### 5.2 可借清單（具體到怎麼用）

1. **DUET 的雙尺度決策與 global action space**（最高優先）。
   - 現況：我們的 navigator 每步對候選 region 打分（輸入 = 低倍特徵 + evidence mean + 剩餘預算），本質上只有「fine-scale local」一路。
   - 可借法：加一條 coarse 路——維護整張 slide 的 region graph（節點 = 所有低倍 region，已 zoom 節點掛上高倍證據表徵，未 zoom 節點只有 HIPT 特徵），用 2-3 層輕量 graph attention（GASA 式：把 region 間空間距離加進 attention bias）算全域分數；fine 路保持現有 per-candidate scoring；dynamic fusion 出最終分數。DUET 的消融給了明確預期：coarse 路提升探索覆蓋（對應我們的 OSR：teacher top-K region 是否被納入候選）、fine 路提升單步選擇精度。參數量可控制在 <1M 內（DUET 的價值在結構不在規模）。
   - 順帶可借：global action space 允許「跳選」任何未 zoom region 而不只鄰近者——我們的 pyramid 本來就無位移成本，天然是 global action space，這反而是我們比 VLN 乾淨的地方，寫論文時可作為對照點。
   - 落地實驗：以現有 navigator 為 baseline，加 GASA coarse 路做 A/B；報 teacher top-K coverage（OSR 類比）與 accuracy-per-budget 曲線位移；DUET 的預期是大 K（長 horizon）時 coarse 路增益放大。
2. **HAMT 式 history encoding／CVLN-ESR 式 logit replay 對應 evidence memory 與 compressed state replay**。
   - evidence mean pooling 對應 VLN 的「condensed fixed-size vector」——正是 HAMT 論文批評的做法。可借：對已累積證據做輕量 attention pooling（query = 當前候選 region 特徵），讓「證據對候選的相關性」進入打分；HAMT 顯示長 horizon（大 K）時增益最大，可預期 budget 越大差距越明顯。
   - CVLN 的 ESR（存每步 action logits、之後 replay 做 self-distillation）與我們的 utility-weighted policy distillation + compressed state replay 是同族方法：他們存 logits、我們存 teacher 分佈與壓縮狀態。可直接引用作為 replay-based continual navigation 的先例並比較：ESR 的 logits 在跨 domain 時語義不變（同一 instruction-following 技能），我們的 teacher 分佈跨任務語義改變，因此需要 utility weighting——這是一個可寫進論文的技術對比。
3. **SPL → accuracy-per-budget 指標移植**。
   - SPL 的結構是「二值 success × (最短資源 / 實際資源)」。固定 K 下第二項是常數，因此正確移植不是單點 SPL，而是 **budget 掃描曲線**（我們的 expB budget curve 已是此物）+ 曲線下面積（accuracy-budget AUC）作為 scalar。另可移植 OSR：「teacher top-K 中被 navigator 納入候選過的比例」，把「選錯」與「根本沒看到」分開診斷——這對分析 forgetting 特別有用（Jaccard 崩壞是因為不再看對地方，還是看了但排序錯）。
   - nDTW/CLS 提醒：不同指標對「行為忠實度」與「結果效率」的取捨不同（DUET 在 R4R 上 SR 升 nDTW 降）。我們同時報 selection Jaccard（行為）與 accuracy（結果）的做法有 VLN 先例背書，可引用。
4. **DAgger／PID 式 on-policy teacher 查詢**。
   - 我們的 counterfactual gain teacher 對任意 evidence set E 都可計算（evaluator frozen），即天生 queryable oracle。可借 DUET-PID 的訓練配方：先 teacher-state 蒸餾（現行做法），再在 navigator 自己 rollout 的狀態上重算 gain(r|E_student) 蒸餾——修正 exposure bias。VLN 的實證（PID > RL）支持我們不需要引入 RL。
   - 落地實驗：兩階段 curriculum（epoch 前半 teacher-state、後半 student-state 蒸餾），對照純 teacher-state；由於 rollout 只是查特徵表，成本幾乎不變。這同時強化 continual 實驗的說服力：student-state 蒸餾下的 forgetting 曲線更貼近部署行為。
5. **Query-conditioned policy（REVERIE/SOON 的 high-level instruction 條件化）**。
   - REVERIE 證明高階、無步級對齊的 instruction（平均 18 詞）足以條件化導航。對應做法：給 navigator 輸入 task embedding（甚至 CONCH text encoder 對診斷任務描述的編碼），單一 policy 服務多任務——這同時是 continual learning 的 architecture 緩解手段（任務條件化降低干擾），也讓「task-incremental」設定更乾淨。
   - 落地實驗：task embedding 用 (i) learned per-task vector 與 (ii) CONCH text embedding 兩版對照；(ii) 若可行，額外賣點是 zero-shot 泛化到未見任務描述的敘事（與 M³E 的 scene-aware router 形成方法對照）。
6. **EnvDrop 的特徵層擾動思想**（部分借）。
   - 無法重渲染 WSI，但 environmental dropout 的本體是「對特徵整體施加一致的 dropout mask 以模擬新環境」——在我們的 feature-space 環境可直接做（對 HIPT/CONCH 特徵通道做 slide 級一致 dropout），作為 replay buffer 的增強，緩解 compressed state replay 的過擬合。

### 5.3 不可直接搬清單 + 確定原因（算力不在此列）

- **(a) 監督來源：人類示範軌跡不存在，且已被證明無法規模化收集。**
  R2R 每條 path 有 GT 軌跡（21,567 條 instruction）；RxR 甚至有 126K 條逐字對齊 pose trace。整個 VLN 訓練棧（teacher forcing、DAgger 的 oracle、pretraining 的 action 標籤、speaker 資料增強的 path-instruction 對）都以此為地基。TCGA 只有 slide-level label；PathAgentBench 動用 10 位 board-certified pathologists 加二審流程，也只標出 1,822 WSI／17,135 條 diagnostic path——這是 benchmark 規模，不是訓練規模，且每新增一個診斷任務都要重標。**結論：監督必須改為 computed teacher（我們的 counterfactual gain），這不是 VLN 方法的超參數調整，而是換掉地基**；VLN 的 IL 配方只能在「teacher 分佈」層面借（KL 蒸餾），不能在「資料」層面借。
- **(b) Success 的定義是幾何的、外生的、task-agnostic；WSI 的 success 是診斷的、內生的、task-dependent。**
  VLN success = 停在 goal 3m 內（或 REVERIE 的 object grounding），由 simulator 幾何判定，**與任務語義和 policy 無關**，因此 reward 場在 continual 設定下不漂移——CVLN 系列換 domain 時 reward 定義完全不變。WSI 沒有幾何 goal：同一 slide 上「好 region」由 frozen evaluator 對特定診斷任務的 CE 改變量定義，換任務（renal→breast）時整個 reward 場翻掉。**結論：SR/SPL/nDTW 不能直接用（無 goal 座標、無參考軌跡）；更根本地，VLN 的 forgetting（環境外觀漂移）與我們的 forgetting（價值函數漂移）不是同一現象，VLN 的解法（存 GT episode replay）沒有直接對應物。**
- **(c) 觀測與 action 的語義不同：空間位移 vs 資訊揭露。**
  VLN 中 agent 移動不改變 information set 的結構——每個 viewpoint 都能看到完整 panorama，「看」是免費的，「走」才是 action。我們的 causal access rule 相反：所有低倍 region 一直可見（等於免費的全域 panorama），action 是「揭露某 region 的高倍證據」——更接近 hard attention／active perception，不是 spatial navigation。**確定後果：VLN 的空間先驗模組無對應物**——heading/elevation 角度編碼、egocentric 相對方位嵌入（DUET fine encoder 的兩類 location embedding）、Floyd 最短路徑執行、waypoint 預測（ETPNav）在 pyramid 上沒有意義（任兩 region 間「距離」為零）；能借的是 graph attention 的結構，不是幾何內容。
- **(d) 環境動態性：simulator 可重渲染、可擾動；預抽取特徵環境是靜態 lookup table。**
  EnvDrop 的環境增強、RxR 的 pose trace 收集、NaVid 的 video stream、所有 sim-to-real 議題，都依賴環境能對任意 (位置, 視角, 擾動) 重新渲染觀測。我們凍結 HIPT/CONCH 後，環境是有限的特徵表——**確定後果：像素級／視角級增強與 video-based 方法（NaVid 路線）整個分支不可用**；但反面是 rollout 成本趨近零（查表），on-policy 方法（5.2-4）反而比 VLN 更便宜——這要寫成優勢而非限制。
- **(e) 語言模態的角色不同：cross-modal grounding 棧無用武之地。**
  VLN 方法的主體（VLN-BERT/HAMT/DUET 的 cross-attention、MLM/MRC 預訓練、speaker model、LLM 的 zero-shot 語言推理）服務於「把逐步語言對齊到視覺」。我們的 diagnostic query 是 per-task 常數（幾個 bit 的 task id），沒有步級語言結構——**確定後果：MLM/MRC 式預訓練、instruction-trajectory 對齊任務、speaker 式資料增強（沒有 language 可反向生成）都不可移植**；語言只能以 task embedding 形式進來（5.2-5）。NavGPT/MapGPT 路線需要觀測可語言化——在 192/512 維凍結特徵上無 caption 可寫；若改用 VLM 直接看 patch 就變成 PathAgent/GIANT/PathNavigate 的 training-free agent 路線，與我們「<1M 可訓練 navigator + 特徵空間 + continual learning」的研究問題正交（他們不學習、無 forgetting 問題可言，但也因此無法累積任務知識——這是定位差異，寫進 related work）。
- **(f) Stop／變長 episode 機制不可搬（設計選擇層面）。**
  VLN 的 SR 高度依賴 stop 決策品質（DUET 消融：fine encoder 的主要貢獻之一是 SR/OSR 比值）。我們固定 K 步、不學 stop，是為了讓 budget 成為受控變因（accuracy-per-budget 曲線）；若照搬 VLN 的 stop 學習，success 判準（b）又不存在，stop 無從監督。**結論：fixed-K 是與 (b) 一致的正確設計，論文中應主動說明而非視為簡化。**
- **(g) Panorama 的連續視角變換 vs pyramid 的離散不可逆揭露。**
  VLN 的 agent 在每個 viewpoint 可自由連續調整 heading/elevation 重新取景，觀測是可逆、可重複的（回到同一節點看到同樣的 panorama，HAMT/DUET 依賴這點更新節點表徵）；WSI pyramid 的 zoom 是離散 parent→child 的單向揭露，沒有「調整視角重看」的連續自由度，已揭露的證據也不會因重訪而更新（特徵固定）。**確定後果：VLN 中所有依賴「重訪即重觀測」的機制（節點表徵的多次更新、視角微調的增益）無對應物；反之我們不需處理視角對齊與觀測噪聲，狀態轉移是確定性的——這使理論分析（例如 forgetting 的歸因）比 VLN 乾淨，應寫成設定優勢。**

**小結——「不可搬」的單句版本（口試/rebuttal 用）**：VLN 方法棧的三個地基——人類示範軌跡、幾何 success、可重渲染 simulator——在 WSI 上一個都不存在；可搬的是地基之上的決策結構（雙尺度、顯式記憶、on-policy 蒸餾、效率加權指標），而算力從頭到尾不是差異所在（國網資源可覆蓋 VLN 級訓練）。

### 5.4 預答可能的質疑：「為什麼不直接把 DUET/HAMT fine-tune 到 WSI？」

逐項檢查 DUET 的輸入輸出介面即可給出確定答案：

1. **輸入端**：DUET 的 text encoder 吃 instruction token 序列（我們沒有）、視覺端吃 36-view panorama 的 view/object 特徵與角度編碼（我們是單一 region 特徵向量，無視角結構）、位置編碼吃 egocentric 方位與距離（pyramid 中無方位語義）。三路輸入中兩路半沒有對應物，「fine-tune」實際上等於重寫輸入層並丟棄大部分預訓練權重——預訓練的遷移價值（其權重編碼的是室內場景視覺-語言對齊）對 H&E 特徵空間為零。
2. **輸出端**：DUET 輸出對 map 節點的分數 + stop + object grounding；我們需要的是對 region 的分數 + 固定 K。stop 與 grounding head 無監督可訓（見 5.3-b/f）。
3. **訓練端**：DUET 的 BC 需要 GT action（沒有）、PID 需要最短路徑 oracle（沒有幾何 goal）、MLM/MRC 需要 instruction（沒有）。它的三個 loss 一個都建構不出來。
4. **成立的部分**：GASA 的 graph attention 結構、dynamic fusion 的門控、topological map 的節點分類（visited/navigable/current ↔ zoomed/candidate/current）——這些是「思想」層面的移植，本文 5.2-1 已具體化。
結論：不是「fine-tune 既有 VLN 模型」而是「以 VLN 驗證過的決策結構為藍圖、在我們的監督棧上重新實例化」——這句話可直接用於 rebuttal。

---

## 6. Continual VLN 的現況與我們的空缺

### 6.1 搜尋到的既有工作（2024–2026）

| 工作 | Venue | Continual 軸 | 方法 | 監督 |
|---|---|---|---|---|
| CVLN（Jeong et al.） | arXiv 2403.15049 → BMVC 2025 | **scene-domain** incremental（I-CVLN 由 R2R+RxR 重組；D-CVLN 由 CVDN 重組、11 個 scene domain） | Perplexity Replay（存高困惑度 episode）＋ Episodic Self-Replay（存每步 action logits 做 self-distillation） | 各 domain 都有 GT path |
| VLNCL / Dual-SR | arXiv 2409.02561 → IROS 2025（作者名待查證） | scene/環境 incremental | 雙迴路 scenario replay（腦啟發 memory consolidation）＋ multi-scenario buffer | GT path |
| M³E | OpenReview pFh5ygjN3V（venue／作者待查證，疑為投稿中） | domain-incremental（R2R/REVERIE） | macro（場景級 router）＋ micro（步級 router）的 hierarchical MoE、動量式選擇性凍結 | GT path |
| TuKA / AlldayWalker | arXiv 2603.14276（2026；venue 待查證） | 多場景 lifelong（AML-VLN） | Tucker 分解 adapter：共享 core tensor + 場景 expert 因子 | GT path |

另有 UAV-VLN survey（arXiv 2604.13654）與 2026 年 foundation-model 路線（ABot-N1 等），均未涉及 continual。

CVLN（最完整的一篇）的實驗細節值得記錄：I-CVLN 由 R2R+RxR 的場景重組為多個 scene domain 序列；D-CVLN 由 CVDN 重組（3,737 訓練／382 驗證 episodes、11 個 scene domain）；評估用「最終訓練階段結束後對所有 domain 的 average metric」（SR/SPL 平均），並實測既有 continual learning 方法（其比較清單細節待查證）在 CVLN 上表現不足，rehearsal 系（PerpR/ESR）顯著較好——這與我們「replay 系優於 regularization 系」的內部經驗一致，可互為佐證。

### 6.2 為什麼這些工作不覆蓋我們的設定（空缺論證）

1. **Continual 軸不同**：全部四篇的任務序列是「換環境」（scene domain），instruction-following 這個技能與 success 定義（幾何 goal）自始至終不變；forgetting 來自視覺分佈漂移。我們的序列是「換診斷任務」：teacher 分佈（gain 場）本身隨任務翻轉，forgetting 來自**價值定義漂移**——selection Jaccard 崩到 0.06 的行為級崩壞在 CVLN 文獻中沒有對應度量（他們報 SR/SPL 的 average metric，不度量選擇行為本身）。
2. **監督結構不同**：四篇的 replay 都直接存「帶 GT action 的 episode」（或其 logits）；可行的前提是 GT 不隨時間失效。我們無 GT path，replay 只能存「壓縮狀態 + 當時的 teacher 分佈」，且舊任務 teacher 與新任務 teacher 語義不同，必須 utility weighting——這是方法上的實質差異，不是換個 dataset。
3. **環境型態不同**：離散 feature-space pyramid + causal access + fixed budget 的組合在 VLN continual 文獻中不存在（他們是 panorama graph + 幾何 success + 學 stop）。
4. **檢索證據**：以 "continual vision-language navigation"、"lifelong navigation agent"（2024–2026）檢索，上述四篇即近乎全部命中；沒有任何工作處理「弱監督 teacher 漂移下的 navigation continual learning」或「病理／醫學影像上的 navigation continual learning」。PathAgent／PathNavigate／GIANT 均為 training-free（無學習故無 forgetting）；PathAgentBench 是評測不是學習方法。

### 6.3 差異化主張（可直接用於論文）

> 據我們所知，這是第一個在 WSI 的 feature-space causal pyramid 上研究 navigation 行為之 continual learning 的工作：與既有 continual VLN（scene-domain incremental、幾何 success、GT 軌跡 replay）不同，我們的任務增量改變的是監督信號本身（counterfactual gain teacher），我們證明這導致選區行為的災難性遺忘（selection Jaccard 0.06），並提出不依賴 GT 軌跡的 utility-weighted policy distillation + compressed state replay。

---

## 7. 對論文的具體建議

### 7.1 Related work 段落（positioning，3-4 句，可直接改寫使用）

> Our navigation formulation is inspired by embodied Vision-and-Language Navigation (VLN), where an agent sequentially selects viewpoints on a connectivity graph under partial observability [anderson2018r2r]; in particular, our coarse-to-fine design echoes DUET's dual-scale planning over a topological map [chen2022duet], and our evidence memory parallels history-aware encoding [chen2021hamt]. However, VLN assumes human demonstration trajectories and a geometric, task-agnostic success criterion, neither of which exists for WSIs: slide-level labels are the only supervision, and "success" is defined solely by a frozen diagnostic evaluator, which moreover changes with each new task. Existing continual VLN addresses scene-domain shift with ground-truth-trajectory replay [jeong2025cvln, wang2025vlncl], leaving task-incremental navigation under weak, teacher-drifting supervision—our setting—unexplored. We therefore borrow VLN's decision structures (dual-scale scoring, budgeted exploration, DAgger-style on-policy distillation) while replacing its supervision stack with counterfactual diagnostic gain.

（中文版可自行翻譯；關鍵是四句的邏輯：借決策結構 → 監督結構不存在 → continual VLN 只做 scene-domain → 我們補上 task-incremental + 弱監督。）

### 7.2 建議引用清單（BibTeX key 建議）

必引（背景 + 方法對照）：`anderson2018r2r`、`fried2018speaker`、`chen2021hamt`、`chen2022duet`、`anderson2018spl`（SPL 定義）、`jeong2025cvln`、`hao2020prevalent`。
選引（依論文篇幅）：`tan2019envdrop`、`hong2021vlnbert`、`ku2020rxr`、`qi2020reverie`、`jain2019cls`、`ilharco2019ndtw`、`ross2011dagger`、`an2024etpnav`、`zhou2024navgpt`、`chen2024mapgpt`、`zhang2024navid`、`wang2025vlncl`、`li2026tuka`（作者待查證）。
病理側定位：`chen2025pathagent`、`pathnavigate2026`（作者待查證）、`pathagentbench2026`（作者待查證）、`buckley2025giant`。

### 7.3 由本 survey 導出的實驗優先序建議

依「VLN 實證強度 × 我們的實作成本 × 對論文主張的貢獻」排序：

1. **OSR 類比指標（teacher top-K coverage）**——零訓練成本，立即可加進現有 continual 實驗，把 Jaccard 崩壞分解為「候選覆蓋崩壞」vs「排序崩壞」；直接強化 forgetting 分析章節。
2. **On-policy（student-state）蒸餾**——DUET PID > RL 的實證背書，rollout 成本趨近零；預期同時提升單任務上限與 continual 穩定性。
3. **雙尺度 navigator（GASA coarse 路 + dynamic fusion）**——結構改動較大，但這是「借 DUET」最實質的一步，與 coarse-to-fine 敘事直接呼應；建議先在單任務驗證增益再進 continual。
4. **Task-conditioned policy（CONCH text embedding）**——中等成本；作為 architecture 系 continual 緩解與 replay 系（現行方法）的對照組。
5. **特徵層 environmental dropout 增強 replay**——低成本、風險低，作為 ablation 的一行。

（1、2 可在現行 codebase 直接做；3、4 若需要較大訓練量，依老師指示以國網 NCHC 資源執行。）

---

## 8. 參考文獻

**Benchmarks**
1. `anderson2018r2r` — Anderson et al., "Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments," CVPR 2018, pp. 3674-3683. arXiv:1711.07280.（R2R：21,567 instructions／90 buildings／success<3m）
2. `ku2020rxr` — Ku, Anderson, Patel, Ie, Baldridge, "Room-Across-Room: Multilingual VLN with Dense Spatiotemporal Grounding," EMNLP 2020, pp. 4392-4412. arXiv:2010.07954.（126K instructions／3 語言／pose traces）
3. `qi2020reverie` — Qi et al., "REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments," CVPR 2020, pp. 9982-9991. arXiv:1904.10151.（21,702 條高階指令，平均 18 詞）
4. `zhu2021soon` — Zhu et al., "SOON: Scenario Oriented Object Navigation with Graph-Based Exploration," CVPR 2021, pp. 12689-12699.（FAO dataset）

**方法**
5. `fried2018speaker` — Fried et al., "Speaker-Follower Models for Vision-and-Language Navigation," NeurIPS 2018. arXiv:1806.02724.
6. `wang2019rcm` — Wang et al., "Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for VLN," CVPR 2019.（arXiv 編號未逐一查證）
7. `tan2019envdrop` — Tan, Yu, Bansal, "Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout," NAACL 2019, pp. 2610-2621. arXiv:1904.04195.
8. `hao2020prevalent` — Hao, Li, Li, Carin, Gao, "Towards Learning a Generic Agent for VLN via Pre-training," CVPR 2020, pp. 13137-13146. arXiv:2002.10638.
9. `hong2021vlnbert` — Hong, Wu, Qi, Rodriguez-Opazo, Gould, "VLN-BERT: A Recurrent Vision-and-Language BERT for Navigation," CVPR 2021, pp. 1643-1653. arXiv:2011.13922.
10. `chen2021hamt` — Chen, Guhur, Schmid, Laptev, "History Aware Multimodal Transformer for VLN," NeurIPS 2021. arXiv:2110.13309.
11. `chen2022duet` — Chen, Guhur, Tapaswi, Schmid, Laptev, "Think Global, Act Local: Dual-scale Graph Transformer for VLN," CVPR 2022 (Oral), pp. 16537-16547. arXiv:2202.11742.
12. `an2024etpnav` — An et al., "ETPNav: Evolving Topological Planning for VLN in Continuous Environments," TPAMI 2024. arXiv:2304.03047.
13. `zhou2024navgpt` — Zhou, Hong, Wu, "NavGPT: Explicit Reasoning in VLN with Large Language Models," AAAI 2024, pp. 7641-7649.
14. `long2024discussnav` — Long, Li, Cai, Dong, "Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions," ICRA 2024, pp. 17380-17387. arXiv:2309.11382.
15. `chen2024mapgpt` — Chen, Lin, Xu, Chai, Liang, Wong, "MapGPT: Map-Guided Prompting with Adaptive Path Planning for VLN," ACL 2024, pp. 9796-9810. arXiv:2401.07314.
16. `zhou2024navgpt2` — Zhou, Hong, Wang, Wang, Wu, "NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models," ECCV 2024. arXiv:2407.12366.
17. `zhang2024navid` — Zhang et al., "NaVid: Video-based VLM Plans the Next Step for VLN," RSS 2024. arXiv:2402.15852.（第一作者名待查證）
18. `abot2026` — "ABot-N1: Toward a General Visual Language Navigation Foundation Model," arXiv:2607.10383（2026；作者／venue 待查證）。
19. `p3nav2026` — "P3Nav: End-to-End Perception, Prediction and Planning for VLN," arXiv:2603.17459（2026；作者／venue 待查證）。

**指標**
20. `anderson2018spl` — Anderson et al., "On Evaluation of Embodied Navigation Agents," arXiv:1807.06757, 2018.（SPL 定義）
21. `ilharco2019ndtw` — Ilharco, Jain, Ku, Ie, Baldridge, "General Evaluation for Instruction Conditioned Navigation using Dynamic Time Warping," NeurIPS ViGIL Workshop 2019.（nDTW/SDTW；arXiv 編號未逐一查證）
22. `jain2019cls` — Jain, Magalhaes, Ku, Vaswani, Ie, Baldridge, "Stay on the Path: Instruction Fidelity in VLN," ACL 2019, pp. 1862-1872.（CLS 定義 + R4R dataset）

**訓練範式理論**
23. `ross2011dagger` — Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS 2011.（DAgger）

**Continual VLN**
24. `jeong2025cvln` — Jeong, Kang, Choi, Kim, Zhang, "Continual Vision-and-Language Navigation," BMVC 2025. arXiv:2403.15049.（PerpR + ESR；I-CVLN/D-CVLN）
25. `wang2025vlncl` — "Vision-Language Navigation with Continual Learning," IROS 2025. arXiv:2409.02561.（Dual-SR；作者名待查證）
26. `m3e2026` — "M³E: Continual VLN via Mixture of Macro and Micro Experts," OpenReview pFh5ygjN3V.（venue／作者待查證）
27. `li2026tuka` — "All-day Multi-scenes Lifelong VLN with Tucker Adaptation," arXiv:2603.14276（2026；AlldayWalker；作者待查證）。

**WSI／病理導航**
28. `chen2025pathagent` — Chen et al., "PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via LLM-based Agentic Reasoning," arXiv:2511.17052, 2025.（training-free；Navigator/Perceptor/Executor；adaptive magnification）
29. `pathnavigate2026` — "PathNavigate: A Training-Free Pathology Agent with Surprise-Guided Scan and Shared Slide Memory for WSI-VQA," arXiv:2605.23559, 2026.（scan-search-readout；低倍 surprise field + 凍結特徵 online memory；作者待查證）
30. `pathagentbench2026` — "PathAgentBench: Benchmarking Evidence-Seeking Vision-Language Models on Whole-Slide Pathology Images," arXiv:2607.19261, 2026.（1,822 TCGA WSI／17,135 pathologist-authored paths／51,363 node descriptions；關鍵發現：multi-scale reasoning >93% 但 text-guided localization mIoU <0.09、自主探索 hit rate 隨倍率 0.522→0.185→0.020——evidence acquisition 是瓶頸；作者待查證。註：任務描述中提及之 EviPathBench 名稱未檢索到，正式名稱應為 PathAgentBench）
31. `buckley2025giant` — Buckley et al., "Navigating Gigapixel Pathology Images with Large Multimodal Models," arXiv:2511.19652, 2025.（GIANT + MultiPathQA：934 題／868 WSI／128 題病理醫師出題；GPT-5+GIANT 62.5% > TITAN 43.8% > SlideChat 37.5%）

---

## 附註：與我們研究最相關的三個交叉觀察

1. **PathAgentBench 的瓶頸結論直接支持我們的問題設定**：現有 VLM 在「給定證據做推理」上>93%，但在「自己找證據」上 mIoU<0.09、高倍 hit rate 0.020——「學會導航去取證據」正是瓶頸所在，且該文證明了醫師軌跡標註的稀缺性（我們 (a) 條的實證）。
2. **Training-free 病理 agent（PathAgent/GIANT/PathNavigate）與我們互補而非競爭**：它們不學習、每張 slide 從零推理（GIANT 每步一次 LMM 呼叫），無法跨任務累積導航知識，也就沒有 forgetting 議題；我們研究的是「可訓練、可累積、會遺忘」的 navigator。PathNavigate 的「低倍 surprise field 先掃、再 query 過濾」流程與我們的 coarse-to-fine 有暗合之處，可在 related work 對照。
3. **CVLN 的 ESR 是離我們最近的既有方法**：同為 replay + distillation，但它存 GT 語義下的 logits、我們存漂移 teacher 下的分佈——在 rebuttal 被問「與 continual VLN 差在哪」時，這是最精確的一組對比。
4. **REVERIE 的監督粒度是我們與 VLN 之間最好的橋**：它證明「高階 goal 描述 + 探索」的可行性，且其 RGS/RGSPL 指標結構（任務成功 × 路徑效率）與我們 accuracy-per-budget 同構；在 related work 中把我們的 diagnostic query 類比為「REVERIE 式 high-level instruction 的極端弱化版（per-task 常數）」，比類比 R2R 的步級 instruction 誠實且準確。

---

*檔案由 VLN survey 任務產出（2026-08-14）。驗證方式：各條目均經網路搜尋核對 venue／年份／關鍵數字；標註「待查證」處為搜尋結果未能完全確認的欄位，引用前請再確認。*
