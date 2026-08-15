// Dated snapshot of the live Cursor Canvas, captured 2026-08-15 21:30 (UTC+8).
// This copy lives in docs/ so it is tracked by git; it does NOT render on its
// own from this location (Cursor only renders .canvas.tsx files found inside
// ~/.cursor/projects/<workspace>/canvases/). To view it as the interactive
// canvas again later, copy it back into that folder, e.g.:
//   cp docs/research_timeline_20260815.canvas.tsx \
//      ~/.cursor/projects/Users-aaron-research-01-mllm-hwsi/canvases/research_timeline_20260815.canvas.tsx
// then open that path in Cursor. See docs/research_timeline_20260815.md for
// the same content as plain, always-readable Markdown.
import {
  H1,
  H2,
  Stack,
  Row,
  Grid,
  Card,
  CardHeader,
  CardBody,
  CollapsibleSection,
  Text,
  Pill,
  Stat,
  Table,
  Callout,
  Divider,
  UsageBar,
} from "cursor/canvas";

type Status = "completed" | "running" | "pending";

function StatusPill({ status }: { status: Status }) {
  const label =
    status === "completed" ? "已完成" : status === "running" ? "進行中" : "待辦";
  return (
    <Pill size="sm" active={status !== "pending"}>
      {label}
    </Pill>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <Stack gap={2}>
      <Text size="small" tone="tertiary" weight="semibold">
        {label}
      </Text>
      <Text size="small">{value}</Text>
    </Stack>
  );
}

interface Phase {
  id: string;
  title: string;
  status: Status;
  what: string;
  why: string;
  result: string;
  paper: string;
}

const phases: Phase[] = [
  {
    id: "p0",
    title: "階段 0 — Protocol v1 凍結（8/14）",
    status: "completed",
    what:
      "定死主實驗的全部定義：can_dataset 四任務序列（ESCA→LUNG→RCC→BRCA）、7 個對比方法（seqft/ewc/lwf/replay/distill/ours_uniform/ours）、觀察預算 K∈{1,2,4}、5 個 seed、指標（AA、forgetting、Jaccard、action-KL、sel_utility）與統計政策（paired bootstrap CI、BH-FDR）。",
    why:
      "沒有一份凍結的 protocol，後面任何實驗都可能被質疑「跑完才調定義」，先把科學定義釘死才能開始正式產生可信結果。",
    result:
      "成功，凍結版存在 docs/protocol.md。這一批跑出來的數字就是 paper 草稿目前引用的主要數字（例如 Jaccard 0.79→0.07）。",
    paper:
      "這是整篇論文的地基：navigation forgetting 這個現象、以及 7 個方法互相比較的主表，全部從這裡的 protocol 產生。",
  },
  {
    id: "p1",
    title: "階段 1 — Direction freeze v0.32 / v0.292（8/14–15）",
    status: "completed",
    what:
      "把論文敘事定稿：問題定名「navigation forgetting」、環境定名「causal feature pyramid」、統計呈現規則、以及一份「禁止誇大用語」清單（例如不能講 universal winner）。",
    why:
      "確保之後所有寫作與紅隊審查用同一套詞彙與同一套統計呈現標準，避免論文口吻前後不一致或被審稿人抓到過度宣稱。",
    result: "成功，寫入 research_contract_v0.292，之後所有文件都以此為準。",
    paper: "決定了 paper 摘要/intro 現在的措辭方式，以及 Section 4 統計呈現的格式規範。",
  },
  {
    id: "p2",
    title: "階段 2 — v0.33 Two-Track 分工（8/15）",
    status: "completed",
    what:
      "把剩下的工作拆成兩條並行的線：Track A＝把既有主線收尾（統計腳本補列、文件修正、第二資料集 pilot40、禁語自動檢查腳本）；Track B＝在既有 7 個方法之外，嘗試 3 個新方法 M1/M2/M3，用一個預先登記（pre-registered）的『gate』流程決定它們值不值得寫進論文。",
    why:
      "Track A 是把已經確定要用的結果做完、做穩；Track B 是有風險的新嘗試，用 gate 機制先講好「贏的標準」，避免看到結果後才回頭改標準（一種對抗事後合理化的做法）。",
    result: "成功，分工方式定案，兩條線各自有自己的任務清單。",
    paper:
      "Track A 決定論文能不能準時交件；Track B 決定論文最後用不用得上一個『更好的方法』，或是維持『分析為主、誠實報告沒找到更好方法』的版本。",
  },
  {
    id: "p3",
    title: "階段 3 — v0.33.1：M1 公式退化 bug（紅隊抓到）",
    status: "completed",
    what:
      "紅隊審查者 Sol 發現 M1（重要性回放取樣）的原始公式有個數學漏洞：某一項在每個狀態上都必然等於 1，等於整個方法悄悄退化成「只看遺忘程度取樣」，跟設計初衷（重要 × 被遺忘）不符。",
    why:
      "這種退化如果沒抓到，跑出來的『M1 有效』結論其實是另一個更簡單方法的結果，論文會下錯結論。",
    result:
      "成功且及時：在任何 gate 訓練開始『之前』就抓到並修正，沒有浪費任何計算資源，也不需要事後丟棄結果。",
    paper: "保護了 Track B 新方法比較的正確性——如果沒抓到，M1 的貢獻宣稱會是假的。",
  },
  {
    id: "p4",
    title: "階段 4 — v0.33.2：M1 校準修正 + 意外發現 backend 差異",
    status: "completed",
    what:
      "把 M1 的重要性分數改成「在自己來源任務內部」算百分位排名（而不是跨任務一起排），避免不同任務的 evaluator 尺度差異被誤讀成『這個狀態比較重要』。同時把新的評估指標（ε-optimal mass、normalized regret）、真正的模型 checkpoint 存檔、per-task 取樣診斷都補齊。過程中依指示在 Mac CPU 上重跑舊方法核對數字，結果對不上凍結的舊資料（誤差最大到 0.148）。",
    why:
      "先前一版定義（跨任務全域排名）雖然沒有退化成常數，但會讓『evaluator 尺度不同』偽裝成『狀態比較重要』，是另一種隱藏的公式風險，同樣由紅隊在跑之前抓到。核對數字失敗則是要確保新程式碼沒有不小心改掉舊結果的計算方式。",
    result:
      "修正成功；另外意外查出：當年凍結的主表數字其實是在 Mac MPS（不是 CPU）上產生的。用 MPS 重跑後 12/12 組完全對上（誤差 0.0000）。這件事本身也記錄成一個小發現：同一份程式碼在 CPU 和 MPS 上會因為浮點運算順序不同而跑出不同結果（這是一個逐步 argmax 模擬常見的現象，不是程式錯誤）。",
    paper:
      "避免 M1 的貢獻被『跨任務 evaluator 尺度差異』污染；CPU/MPS 差異則會被寫成一條 reproducibility 附註/腳注，屬於誠實揭露的工程細節。",
  },
  {
    id: "p5",
    title: "階段 5 — v0.33.3：gate 全部改在 MPS 上跑",
    status: "completed",
    what:
      "因為上一階段發現 backend 會影響數字，原本『新方法在 RunPod CUDA 跑、拿 Mac MPS 的舊結果來比較』的計畫改掉：整個 gate（新方法 + 對照組）統一在 Mac MPS 上跑到底，pilot40 的資料量體工作則維持在 RunPod CUDA 進行（跟方法比較無關）。另外寫了一份 docs/compute_policy.md 把『什麼工作該用哪個 backend』的規則定下來。",
    why:
      "如果新方法在 CUDA 跑、對照組在 MPS 跑，最後看到的『差異』有可能其實是硬體造成的，不是方法本身造成的——這會混淆『方法效果』跟『backend 效果』，是實驗設計上的大忌。",
    result: "成功，避免了一個可能讓整個 Track B 結論失效的混淆變因。",
    paper: "保證『M1/M2/M3 比舊方法好或不好』這句話，講的是方法本身，不是硬體巧合。",
  },
  {
    id: "p6",
    title: "階段 6 — RunPod CUDA：pilot40 主網格（8/15 晚間）",
    status: "completed",
    what:
      "在有『真實空間座標』的第二個資料集（腎臟/乳房兩任務）上，跑同一套 7 個既有方法（5 seeds × 7 methods × K∈{1,2,4}）。中途踩過一個環境變數 bug：新開的 tmux pane 沒繼承裝好套件的 Python 環境（env.sh 沒 source），導致 10 個訓練工作瞬間因缺 pandas 而崩潰、卻誤以為『跑完了』；修好後重跑，5 個 shard 全部乾淨完成。",
    why:
      "驗證 navigation forgetting 這個現象不是 can_dataset 偽座標環境的產物，在真實空間結構的資料上是否一樣成立。",
    result:
      "成功。315 筆結果（5 seeds × 7 methods × 3 K × 3 stage-rows）數字對得上，無任何 Traceback，main_table_pilot40_main.md 與兩張圖已產生。已在 pod 上 commit，push 到 origin/main 這一步尚待 Aaron 確認完成。",
    paper:
      "提供第二資料集的主表，支持『navigation forgetting 及其緩解方法可以跨資料集重現』的 generalization 主張。",
  },
];

const currentlyRunning = [
  {
    id: "r1",
    title: "Mac MPS — Track B Method Gate（決定 M1/M2/M3 能否進論文）",
    done: 16,
    total: 33,
    detail:
      "33 個單位 = {M1 單獨, M2 單獨, M3=M1+M2, 對照 mini-arms} × 3 seeds × K∈{1,2,4}。平均每單位約 10–20 分鐘（原估計 3.7 小時偏樂觀，因為 M1 每 50 步要對整個 buffer 做一次額外的 forward pass，實際預估總時長 8–10 小時，仍遠低於 48 小時上限）。跑完會依照預先登記的 4 條判準（g1 遺忘不變差、g2 至少一項變好、g3 不輸 replay、g4 新任務學習力不掉超過 0.01）逐一檢查。",
  },
];

const pitfalls: [string, string, string][] = [
  [
    "RunPod 終端機顏色分不清指令跟輸出",
    "預設 prompt 太陽春",
    "客製 PS1（黃色帳號＋青色路徑＋git branch，指令另起一行），寫進 bootstrap script 永久生效",
  ],
  [
    "pilot40 訓練找不到資料檔（FileNotFoundError）",
    "data/ 在 .gitignore 內，從未上傳過 RunPod",
    "小檔用 scp、5.1GB 大資料夾用 rsync 手動傳，checksum 驗證",
  ],
  [
    "tmux 內 Ctrl-b d 離開沒反應",
    "網頁版終端機把組合鍵攔截給瀏覽器",
    "改用純文字指令 tmux detach-client",
  ],
  [
    "rsync: command not found",
    "RunPod 精簡映像預設沒裝 rsync",
    "手動 apt-get install，並把自動安裝寫進 bootstrap script",
  ],
  [
    "rsync 傳輸時一堆 chown Operation not permitted 警告",
    "網路硬碟（Network Volume）不允許改檔案 owner",
    "確認只是 metadata 警告、內容仍完整傳輸，忽略即可",
  ],
  [
    "全網格「跑 1 分鐘就結束」，看似正常實則全滅",
    "新開的 tmux pane 沒 source 到 env.sh，PATH 撿到缺套件的系統 Python",
    "規定「新 tmux pane 一律先 source env.sh 再 export 變數」，並寫入 RUNPOD_SOP.md 的固定坑位段落",
  ],
];

type NextStatus = "running" | "pending";
interface NextStep {
  status: NextStatus;
  text: string;
}

const nextSteps: NextStep[] = [
  { status: "running", text: "跑完 Mac MPS 上剩餘的 method gate 單位（目前 16/33）" },
  {
    status: "running",
    text: "確認 pilot40 結果已從 RunPod push 到 origin/main，並在 Mac 這邊 git pull 驗證",
  },
  {
    status: "pending",
    text: "整理 results/method_gate_verdict.md，與紅隊（Sol、Fable）做一次 joint unblinding review",
  },
  {
    status: "pending",
    text: "若 M1/M2/M3 通過 gate → 擴大到 5 seeds×{main,reverse}×K∈{1,2,4} 的 promotion run（deadline 8/23），並在 8/18 前決定方法的正式公開名稱",
  },
  {
    status: "pending",
    text: "若沒有任何配置通過 gate → 不重跑、不硬凹，誠實把嘗試過程寫進論文附錄，主線維持『分析為主』的版本（這是 protocol 裡預先承認的有效結果，不算失敗）",
  },
  {
    status: "pending",
    text: "pilot40 主表（main_table_pilot40_main.md）已產生，接下來拿來跟 can_dataset 主表對照分析，餵進 Gate 1'（8/18，learned policy 是否顯著贏 random）",
  },
  {
    status: "pending",
    text: "Protocol 既定的論文級三道關卡：Gate 2'（8/22，seqft 遺忘是否顯著）、Gate 3'（8/24，ours 是否顯著贏 seqft）",
  },
  {
    status: "pending",
    text: "Track A 收尾雜項：main.tex 內 KL 方向文字、docs/handoff 文件的 cell 編號、補齊 mechanism robustness 指標",
  },
  {
    status: "pending",
    text: "把 paper 草稿裡目前的 \\xx{} 佔位數字換成最終數字，跑一次 check_forbidden_phrases.py，準備投稿 ICASSP 2027",
  },
];

function PhaseCard({ phase }: { key?: string; phase: Phase }) {
  return (
    <CollapsibleSection
      title={phase.title}
      trailing={<StatusPill status={phase.status} />}
    >
      <Stack gap={10} style={{ paddingTop: 4, paddingBottom: 8 }}>
        <Field label="做了什麼" value={phase.what} />
        <Field label="為什麼要做" value={phase.why} />
        <Field label="結果" value={phase.result} />
        <Field label="跟 paper 的關係" value={phase.paper} />
      </Stack>
    </CollapsibleSection>
  );
}

function RunningCard({
  item,
}: {
  key?: string;
  item: (typeof currentlyRunning)[number];
}) {
  const pct = item.total > 0 ? Math.round((item.done / item.total) * 100) : 0;
  return (
    <Card>
      <CardHeader trailing={<Pill active size="sm">進行中</Pill>}>
        {item.title}
      </CardHeader>
      <CardBody>
        <Stack gap={10}>
          <Text size="small">{item.detail}</Text>
          <Stack gap={4}>
            <Row justify="space-between">
              <Text size="small" tone="tertiary">
                {item.total > 0 ? `${item.done} / ${item.total} 單位完成` : "剛重新啟動"}
              </Text>
              {item.total > 0 && (
                <Text size="small" tone="tertiary">
                  {pct}%
                </Text>
              )}
            </Row>
            {item.total > 0 && (
              <UsageBar
                segments={[{ id: "done", value: item.done, color: "blue" }]}
                total={item.total}
              />
            )}
          </Stack>
        </Stack>
      </CardBody>
    </Card>
  );
}

export default function ResearchTimeline() {
  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 980, margin: "0 auto" }}>
      <Stack gap={8}>
        <H1>Navigation Forgetting × MLLM-WSI — 研究時間線</H1>
        <Text tone="secondary">
          給老師的口頭報告用整理：目前做了什麼、為什麼做、結果如何、跟論文的關係，以及接下來的 scope。
        </Text>
        <Text size="small" tone="quaternary">
          最後更新：2026-08-15 21:30 (UTC+8)
        </Text>
      </Stack>

      <Callout tone="info" title="這篇論文在問什麼問題（一句話版本）">
        <Text size="small">
          病理醫生看巨幅玻片（WSI）不會整張看完，而是先看縮圖、挑幾個可疑區域放大細看，在有限「觀察預算」下決定「該看哪裡」。
          我們發現：當這個「該看哪裡」的政策（policy）被拿去連續學習新任務（continual learning）時，會出現一種標準 CL
          指標量不到的遺忘——選擇行為早就崩壞了（Jaccard 從 0.79 掉到 0.07），但因為影像內證據有冗餘，凍結的預測頭還能靠『矇對』撐住準確率，
          直到預算被壓得很緊，準確率才跟著垮。這就是「navigation forgetting」，也是整篇論文要提出並解決的問題。
        </Text>
      </Callout>

      <Grid columns={4} gap={12}>
        <Stat value="16 / 33" label="Method gate 進度（Mac MPS）" tone="info" />
        <Stat value="2" label="紅隊事前抓到的公式/設計 bug" tone="success" />
        <Stat value="3" label="待決新方法 M1 / M2 / M3" />
        <Stat value="2 / 2" label="資料集完成：can_dataset + pilot40" tone="success" />
      </Grid>

      <Stack gap={4}>
        <H2>正在跑</H2>
        <Text size="small" tone="secondary">
          pilot40 全網格已經成功跑完（見下方階段 6）；現在唯一還在跑的是決定新方法能不能寫進論文的 method gate。
        </Text>
      </Stack>
      <Grid columns={1} gap={12}>
        {currentlyRunning.map((item) => (
          <RunningCard key={item.id} item={item} />
        ))}
      </Grid>

      <Divider />

      <Stack gap={4}>
        <H2>時間線：從 Protocol 凍結到現在</H2>
        <Text size="small" tone="secondary">
          點開每一項看「做了什麼／為什麼／結果／跟論文的關係」。全部已完成，且都是在正式跑實驗「之前」抓到問題，沒有浪費算力。
        </Text>
      </Stack>
      <Stack gap={2}>
        {phases.map((phase) => (
          <PhaseCard key={phase.id} phase={phase} />
        ))}
      </Stack>

      <Divider />

      <H2>過程中順手解決的工程坑（跟科學結果無關，純基礎設施）</H2>
      <Table
        headers={["問題", "根本原因", "怎麼解決"]}
        rows={pitfalls}
        striped
      />

      <Divider />

      <Stack gap={4}>
        <H2>接下來的 scope：還要跑什麼、寫什麼</H2>
        <Text size="small" tone="secondary">
          前兩項是現在進行式，其餘照順序是等它們跑完之後的必要步驟。
        </Text>
      </Stack>
      <Stack gap={0}>
        {(() => {
          let pendingIndex = 0;
          return nextSteps.map((step, i) => {
            if (step.status === "pending") pendingIndex += 1;
            return (
              <div key={i}>
                <Row gap={10} align="start" style={{ padding: "8px 0" }}>
                  <Pill size="sm" active={step.status === "running"}>
                    {step.status === "running" ? "進行中" : `待辦 ${pendingIndex}`}
                  </Pill>
                  <Text size="small" style={{ flex: 1 }}>
                    {step.text}
                  </Text>
                </Row>
              </div>
            );
          });
        })()}
      </Stack>
    </Stack>
  );
}
