# RunPod SOP v2 — `mllm_hwsi_ah`

## Aaron 加入實際可操作的重新開 RunPod 的操作筆記：

---

---

### 先記錄一下我的 Cursor 在它自己內部 tsx 的瀏覽器，我們進行的 tasks 的各種訊息：Cmd+P找檔（在Cursor中），貼上去路徑：

```bash
/Users/aaron/.cursor/projects/Users-aaron-research-01-mllm-hwsi/canvases/research-timeline.canvas.tsx
```

怎麼樣才能看到
最簡單、保證有效的方式：直接點我上一則訊息裡貼的那個連結——

research-timeline

在 Cursor 的對話視窗裡，這種檔案路徑連結是可以點擊的，點下去 Cursor 會知道這是一個 canvas，直接在旁邊開一個渲染好的畫面（不是純文字/程式碼視圖）。這也是官方文件明確講的：「使用者可以直接點聊天室裡的 canvas 路徑連結來打開它，就跟點其他檔案路徑一樣。」

如果你是想事後、不透過我貼的連結、自己去找到它重新打開，可以試試：

在 Cursor 裡按 Cmd+P（快速開檔），貼上完整路徑 /Users/aaron/.cursor/projects/Users-aaron-research-01-mllm-hwsi/canvases/research-timeline.canvas.tsx 直接跳轉。

或用 Finder／終端機導覽到那個資料夾，把 .canvas.tsx 檔案本身拖進 Cursor 視窗裡打開（用 Cursor 打開，而不是用系統預設程式或瀏覽器）。
一個提醒
因為它活在 Cursor 的 app 資料夾裡、不在你的 git repo 裡，它不會被 git 追蹤、也不會隨你的研究專案備份。如果哪天你清了 Cursor 的快取或換了一台機器，這個 canvas 檔案不會自動跟過去。如果你想要它長期保留（例如當作進度報告存檔），跟我說一聲，我可以把同樣的內容另外存成一份 Markdown 或加進 docs/ 底下，那樣就會跟著 git 走、也比較好找。要幫你做這件事嗎？

---

好的，這個狀況其實完全在預期內，而且比看起來簡單很多。先講重點讓你安心：**RunPod 換新 pod 時，容器（container）本身的 root filesystem 一定會被清空，只有掛載的 Network Volume（也就是** `/workspace`**）會保留。這正是 SOP 設計成「fresh pod」流程的原因** —— repo、venv、dataset cache 全部都放在 `/workspace` 底下,理論上都還在,只是這次啟動的容器是全新的,所以 `tmux`、shell 環境變數、PATH 這些「容器層」的東西被重置了而已。

順便更新一下背景：Mac 這邊的 MPS gate 還在跑（`method_gate_v0333_run1`，目前 33 個 unit 完成 2 個，`ia_samp seed0 K=1/K=2`），不受你這邊 RunPod 操作影響。

下面是一步一步的實際操作，全部用複製貼上就好。

## 第 0 步：先確認你「進到哪裡」

RunPod 網頁上，你的 pod 卡片右上角有一個 **Connect** 按鈕，點開選 **Start Web Terminal**（或如果你有設 SSH，就用 SSH 連進去）。打開的黑底終端機視窗，就是你要貼指令的地方。

## 第 1 步：健康檢查（先貼這段，看看 `/workspace` 到底剩什麼）

```bash
set -u
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ls -ld /workspace /workspace/mllm_hwsi_ah 2>&1
if [ -d /workspace/mllm_hwsi_ah/.git ]; then
  git -C /workspace/mllm_hwsi_ah rev-parse HEAD
else
  echo "REPO_MISSING"
fi
```

**預期結果**：應該會印出一個 GPU 型號、`/workspace/mllm_hwsi_ah` 存在、以及一個 commit hash（不是 `REPO_MISSING`）。把這段結果貼給我看一下最保險，但如果你想直接往下走也可以。

- 如果印出 `REPO_MISSING`（代表 volume 沒掛對，或這是全新 volume）：先不要往下做，回報給我，我們要先確認 Network Volume 有沒有選對再繼續，不然會重新下載一次 dataset。

---

---

# `Aaron筆記開始`

## 改PS1，command 的 prompt 以及 git pull

Pushed successfully. Now let's write the response.

## PS1 修改方案

我把它直接寫進 `scripts/runpod_bootstrap.sh`（已 commit + push，commit `09a3ac4`），這樣以後每次開新 pod 跑 bootstrap 就會自動套用，不用每次手動改一次、也不會被容器重置吃掉。

**設計：**

- 使用者名 → 黃色
- 拿掉 hostname
- 一個灰色分隔符 `|`
- 目前完整路徑 → 青色
- 如果剛好在 git repo 裡面 → 自動顯示 branch 名（洋紅色，例如 `(main)`）
- **重點**：真正輸入指令的那個 `#`/`$` 符號自己獨立一行 —— 這樣長 scrollback 裡你一眼就能抓到「這行開頭有孤零零一個 `#` = 這是一條新指令」，其他都是 output，複製貼給我的時候也比較好分段。

長這樣（示意，實際會有顏色）：

```
aaron | /workspace/mllm_hwsi_ah (main)
# git status
```

## 怎麼套用到你現在這個 pod

因為你現在已經在跑 bootstrap 流程了，最簡單的方式是**直接重新** `git pull` **+ 重跑 bootstrap 一次**，順便把 v0.33.3 的 commit 也一起拉下來（你剛剛健康檢查印出的 commit 是 `116b0ad...`，是拉之前的舊版，這是預期中的，因為 `git pull` 本來就是 SOP 的下一步）：

```bash
cd /workspace/mllm_hwsi_ah
git pull --ff-only origin main
source scripts/runpod_bootstrap.sh
```

跑完之後 prompt 就會自動變成新樣子。如果你想先不動 bootstrap、只想馬上看到效果，也可以只貼這一行（當次 session 立即生效，但重開 pod 或開新 terminal 分頁就會消失，除非你有跑過上面那個 `source scripts/runpod_bootstrap.sh`）：

```bash
export PS1='\[\033[1;33m\]\u\[\033[0m\] \[\033[2m\]|\[\033[0m\] \[\033[1;36m\]\w\[\033[0m\]\[\033[35m\]$(b=$(git symbolic-ref --short HEAD 2>/dev/null); [ -n "$b" ] && printf " (%s)" "$b")\[\033[0m\]\n\$ '
```

## 想微調的話

- 只想顯示資料夾名字不要整串路徑：把上面指令裡的 `\w` 換成 `\W`（大寫）。
- 顏色代碼對照：`33m`=黃、`36m`=青、`35m`=洋紅、`32m`=綠、`31m`=紅，想換色直接改數字即可。
- 這段邏輯現在寫在 `write_env_file()` 裡（`scripts/runpod_bootstrap.sh`），是 append 進 `/workspace/.mllm_hwsi_ah/env.sh` 的最後一段，之後想再調整風格，改這個檔案再重跑 bootstrap 就會生效，不用每台 pod 手動改。

---

---

## 第 2 步：跑 bootstrap（這一步會自動裝好 tmux、venv、PATH，全部一次到位）

```bash
export GIT_USER_NAME="AaronHung (runpod)"
export GIT_USER_EMAIL="hung.aaron@gmail.com"
source /workspace/mllm_hwsi_ah/scripts/runpod_bootstrap.sh
```

這一行 `source` 會自動做完你問的所有事：

- **裝 tmux**：容器重置後 `tmux` 通常會不見，這行會用 `apt-get` 重裝（如果你是 root）。
- **重建/檢查 python venv**：venv 資料夾本身在 `/workspace/venvs/mllm_hwsi_ah`，是持久的，如果它還在，這步只是重新 `pip install -r requirements-nav.txt` 確保套件齊全（很快，多半是 no-op）；如果 venv 資料夾也被清了，這步會從頭建。
- **restore PATH／環境變數**：寫一份 `/workspace/.mllm_hwsi_ah/env.sh`（這個檔案在 `/workspace` 底下，是持久的），裡面把 `PROJECT_DIR`、`CAN_ROOT`、`VENV_DIR`、`PYTHON_BIN`、`PYTHONPATH`，以及把 `PATH` 指到 `$VENV_DIR/bin`（讓 `python`/`pip` 用對的那一份），全部 export 好，然後立刻 `source` 生效。
- **驗證 dataset**：最後自動跑 `verify_runpod_data.sh`，檢查 `can_cache`、`tcga_*/table`、`datasplit/fold_1.npz`，並用 `SHA256SUMS.txt` 做 checksum 驗證。

**預期結果**：最後一行應該是

```
[bootstrap] PASS — environment and data are ready.
```

- 如果卡在 `tmux is unavailable` 的錯誤：代表你不是 root 而且沒有網路裝 tmux，回報給我,我們用備用的 static binary 方案。
- 如果卡在 `[data] FAIL`：代表某個 dataset 路徑不見了，它會告訴你缺哪個檔案，先不要往下跑實驗，回報給我。

## 第 3 步：確認 commit 是最新的（v0.33.3 已經 commit 好了）

```bash
cd /workspace/mllm_hwsi_ah
git pull --ff-only origin main
echo "commit=$(git rev-parse HEAD)"
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

**預期結果**：`commit` 應該是 `4ac6953...`（v0.33.3），`cuda True`。

- 如果 `git pull`／`git push` 跳出要你輸入帳號密碼／token：這代表這個新容器沒有存 git 憑證（憑證通常存
  在 `$HOME`，不在 `/workspace`，所以會被清掉），而且 GitHub 從 2021 年起 git-over-HTTPS 不再接受帳號
  密碼，只吃 **Personal Access Token（PAT）**。bootstrap 已經把憑證快取位置指到
  `$STATE_DIR/.git-credentials`（在 `/workspace` 底下，持久），所以只要**在同一顆 volume 上輸入過一次
  PAT，之後所有新開的 pod（只要掛同一顆 volume）都不會再問**。第一次輸入：
  1. 到 <https://github.com/settings/tokens?type=beta> 建一個 fine-grained token，repository 選這個
     repo，Permissions → Contents 設成 Read and write，產生後立刻複製（只顯示一次）。
  2. 貼這行（一次性，之後就不用再打）：
     ```bash
     git push origin main
     ```
     跳出 `Username` 填 GitHub 帳號，跳出 `Password` **貼 PAT**（不是真正密碼）。成功後憑證就存進
     `$STATE_DIR/.git-credentials`，之後不會再問。

## 第 4 步：確認有沒有殘留的 tmux session

```bash
tmux ls 2>/dev/null || echo "NO_TMUX_SESSION_YET"
```

如果之前有跑過東西且 session 還在（`tmux ls` 有印出東西），用 `tmux attach -t <名字>` 接回去看；沒有的話就是 `NO_TMUX_SESSION_YET`，代表要重新起一個。

## 第 5 步：啟動 pilot40（先 rehearsal，再全網格）

先跑一個小的 rehearsal 確認 CUDA 真的能跑完一輪：

```bash
export RUN_TAG="v2_pilot40_rehearsal_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_DIR/logs"
tmux new -s pilot40
```

進到 tmux 裡面之後（畫面會換成 tmux 的樣子）：

```bash
cd "$PROJECT_DIR"
set -o pipefail
"$PYTHON_BIN" scripts/cl_main.py \
  --dataset pilot40 --order main --smoke --device cuda \
  --output-dir "$RUN_DIR/results" --tag "$RUN_TAG" --resume \
  2>&1 | tee "$RUN_DIR/logs/rehearsal.log"
```

跑完應該幾分鐘內結束，CSV 有兩個 task、`K=1`、`resolved_device=cuda`。確認沒問題後，按 `Ctrl-b` 再按 `d` 離開 tmux（不會中斷程式），然後開全網格：

```bash
export DATASET=pilot40 ORDERS=main
export RUN_TAG="v2_pilot40_main_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_ROOT/logs"
tmux new -s pilot40_full
```

tmux 裡面：

```bash
cd "$PROJECT_DIR"
set -o pipefail
RUN_TAG="$RUN_TAG" RUN_ROOT="$RUN_ROOT" DEVICE=cuda \
  DATASET="$DATASET" ORDERS="$ORDERS" \
  bash scripts/run_main_cl.sh 2 \
  2>&1 | tee "$RUN_ROOT/logs/main_grid.log"
```

再 `Ctrl-b` `d` detach，就可以關掉網頁/斷線離開，之後回來用 `tmux attach -t pilot40_full` 接回去看進度即可。

---

**建議做法**：你就照第 1～3 步先做，把每一步的終端機輸出貼給我看，我幫你確認每個 checkpoint 都對了再進到第 4、5 步——這樣萬一哪一步卡住，我們可以立刻抓到是哪個環節，不用整個重來。

---

---

---

先回答比較急的 tmux 問題,再處理 pilot40 資料遺失的問題。

## 一、tmux 基本操作(不依賴 Ctrl-b d)

`Ctrl-b` 再按 `d` 沒反應,十之八九是因為 RunPod 的網頁 web terminal 會把 `Ctrl+b` 這類組合鍵攔截給瀏覽器用,根本沒傳到 tmux。**直接用文字指令代替按鍵組合最穩,不管什麼終端機都會動。**

| 操作                                     | 指令(在 tmux session **裡面**打,按 Enter)                          |
| ---------------------------------------- | ------------------------------------------------------------------ |
| 離開但不中斷程式(detach)                 | `tmux detach-client` 或縮寫 `tmux detach`                          |
| 回到剛剛那個 session(在 tmux **外面**打) | `tmux attach -t pilot40`(或你取的 session 名字)                    |
| 列出目前有哪些 session                   | `tmux ls`                                                          |
| 往上捲動看之前的 output                  | 先 `Ctrl-b` 再按 `[` 進入捲動模式,方向鍵/PageUp 往上捲,按 `q` 離開 |
| 真的要砍掉某個 session(小心用)           | `tmux kill-session -t pilot40`                                     |

如果你想確認 `Ctrl-b` 到底有沒有送進去,可以直接改用 `tmux detach-client` 那行文字指令,一定有效。

### ⚠️ 千萬不要用 Ctrl-C 代替 detach(2026-08-17 實際踩到,Mac 端)

`Ctrl-b d` 沒反應時,很自然的下一個反射動作是按 `Ctrl-C` 想「跳出來」。**不要。**
`Ctrl-C` 送的是 SIGINT,收件人是那個 pane 的前景程序,也就是你正在跑的那個跑批本身——
它會被當場中斷,而不是讓你離開 tmux。v0.34 development gate 就是這樣掉了一個進行中的 unit
(還好 runner 是 atomic resume,只掉當下那一個,重下同一條指令就接回去了)。

正確做法,由好到壞:

1. **根本不要 attach**。監看用唯讀的方式,在一般終端機下
   `bash scripts/watch_run.sh <tag>` 或 `tail -f logs/<tag>.log`——這兩個怎麼 Ctrl-C 都只會停掉
   監看本身,碰不到跑批。
2. 真的 attach 進去了,要離開就打 `tmux detach-client`(文字指令,按 Enter)。
3. 真的要停跑批,用 `tmux kill-session -t <session>` 或 `pkill -f <runner>.py`,
   而且盡量停在 unit 與 unit 之間。

長跑批的 launcher script 建議在開頭加 `trap '' INT`(被忽略的 signal disposition 會跨 exec
繼承下去,所以 `caffeinate` 跟 `python` 子程序一起被保護),這樣誤按 Ctrl-C 就完全無害。
v0.34 的 launcher 已經這樣做了。

## 二、pilot40 資料沒上傳的問題

錯誤是對的、也在預期範圍內——`FileNotFoundError: .../data/pilot40_selection.csv`。原因：

`data/` 整個資料夾(包含 `pilot40_selection.csv` 跟 5.1GB 的 `brca_pilot40_k48/`)是 `.gitignore` 排除的,**從來沒有進過 git**，只存在你 Mac 本機。之前只有 `can_dataset` 走過完整的「上傳到 RunPod `/workspace`」流程,pilot40 這批資料這是第一次要送到 RunPod,SOP 裡確實漏了這段——我等下會把它補進 `docs/RUNPOD_SOP.md`。現在先讓你能動起來。

**需要傳的東西**（都在 Mac 這邊 `/Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah/data/`）：

- `pilot40_selection.csv`（14KB，很快）
- `brca_pilot40_k48/`（**5.1GB**，要花一點時間,視你上傳頻寬而定）

---

---

### 正確的(aaron)

抓到問題了，是 `-e` 裡面的 `root@host` 位置放錯，不是你操作錯，是我上一版指令的說明沒講清楚 `-e` 到底該放什麼。

## Bug 在哪

`rsync -e "..."` 這個 `-e` 只能放「ssh 程式本身 + 它的選項」（像 `-p PORT`、`-i KEY`），**不能把** `root@host` **也塞進去**——因為 rsync 執行時，會自己把目的地那串 `root@157.157.221.29` 接到你 `-e` 給的指令**後面**。你等於把 host 放了兩次：

```
你給的 -e 內容：       ssh root@157.157.221.29 -p 30125 -i ~/.ssh/id_ed25519
rsync 自動接上目的地：                                                      root@157.157.221.29 rsync --server ...
實際執行變成：         ssh root@157.157.221.29 -p 30125 -i ~/.ssh/id_ed25519 root@157.157.221.29 rsync --server ...
```

ssh 的參數規則是「第一個不是 `-` 開頭的字」就當作連線目標，後面全部都會被當成「要在遠端執行的指令」，不會再當 ssh 的選項解析。所以 `-p 30125 -i ~/.ssh/id_ed25519 root@157.157.221.29` 這串全部被當成遠端指令送過去執行，遠端的 bash 才會回你 `root@157.157.221.29: command not found`——它是在你的遠端 shell 上被誤執行的，不是本機。

## 修正版（`-e` 只放 ssh 選項，host 只出現在 source/destination 那個參數）

```bash
export RUNPOD_HOST="157.157.221.29"
export RUNPOD_PORT="30125"
export RUNPOD_KEY="$HOME/.ssh/id_ed25519"
cd /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah

# 先測連線
ssh -p "$RUNPOD_PORT" -i "$RUNPOD_KEY" "root@${RUNPOD_HOST}" "echo ok"

# 小檔案 CSV
scp -P "$RUNPOD_PORT" -i "$RUNPOD_KEY" \
  data/pilot40_selection.csv \
  "root@${RUNPOD_HOST}:/workspace/mllm_hwsi_ah/data/pilot40_selection.csv"

# 大資料夾 5.1GB，用 rsync（中斷可以重跑同一行只補傳沒傳完的）
rsync -avz --progress -e "ssh -p $RUNPOD_PORT -i $RUNPOD_KEY" \
  data/brca_pilot40_k48 \
  "root@${RUNPOD_HOST}:/workspace/mllm_hwsi_ah/data/"
```

## 注意先安裝：

為什麼會這樣
rsync 是靠 ssh 連過去之後，在兩邊都要有 rsync 這支程式才能運作——本機負責發起，遠端要跑起 rsync --server ... 來接收，兩邊缺一個都不行。你 Mac 上 brew 裝的 rsync 沒問題，但 RunPod 這個新開的 CUDA 容器是精簡映像，預設沒有裝 rsync（只給了基本的 scp／ssh），所以 bash: line 1: rsync: command not found 這行其實是遠端 pod 的 bash 在抱怨它自己找不到 rsync。

（那個 zsh: command not found: # 沒關係，是你把註解那行也貼進去執行了——zsh 互動模式預設不把 # 開頭當註解處理，純粹是貼太多行的小插曲，跟真正的錯誤無關，不用管它。）

修法：先在 pod 上裝 rsync，再重跑同一行
回到你 SSH 進 pod 的那個視窗（或用 ssh -p "$RUNPOD_PORT" -i "$RUNPOD_KEY" root@${RUNPOD_HOST} 再開一個），跑：

```bash
apt-get update -qq && apt-get install -y rsync
```

裝完之後，回 Mac 這邊，原封不動重跑剛剛那行（不用加註解那行了）——這是錯的，因為指令少了行尾的 、 換行：

```bash
rsync -avz --progress -e "ssh -p $RUNPOD_PORT -i $RUNPOD_KEY"
  data/brca_pilot40_k48
  "root@${RUNPOD_HOST}:/workspace/mllm_hwsi_ah/data/"
```

跑這個

### 修法：直接用單行指令，不靠 \ 換行

這樣不管貼上去的環境怎麼處理換行都不會出事：

```bash
rsync -avz --progress -e "ssh -p $RUNPOD_PORT -i $RUNPOD_KEY" data/brca_pilot40_k48 "root@${RUNPOD_HOST}:/workspace/mllm_hwsi_ah/data/"
```

### 貼之前先確認變數還在（你可能是開了新的分頁/新的 shell）

你現在這個 prompt 看起來是全新的一行（pt-exp conda env、時間戳都跑出來了），先確認 RUNPOD_HOST/RUNPOD_PORT/RUNPOD_KEY 這三個變數還有值，沒有的話要重新 export：

```bash
echo "HOST=$RUNPOD_HOST PORT=$RUNPOD_PORT KEY=$RUNPOD_KEY"
```

如果印出來是空的，就重新跑一次：

```bash
export RUNPOD_HOST="157.157.221.29"
export RUNPOD_PORT="30125"
export RUNPOD_KEY="$HOME/.ssh/id_ed25519"
```

再貼上面那行單行 rsync 指令即可。之後只要是要你貼到終端機執行的指令，我都會盡量寫成單行、不用 \ 換行，就是為了避免這種「反斜線在複製貼上路上消失」的狀況再發生。

#### 傳輸相關：

這個是**完全無害、常見的警告，不是真正的錯誤，資料有在正常傳，讓它繼續跑完就好，不用中斷**。

## 為什麼會有這些 `chown ... Operation not permitted`

`rsync -a`（archive 模式）裡面的 `a` 其實是好幾個選項的縮寫（`-rlptgoD`），其中 `-o`／`-g` 的意思是「連同檔案的 owner／group 也一起在目的地設成跟來源一樣」。你的來源是 Mac 上你自己的帳號（例如 UID 501），rsync 想在 RunPod 那邊把檔案 owner 改成同一個 UID——但 RunPod 掛的 `/workspace` 通常是**網路硬碟（Network Volume）**，這類網路檔案系統很多時候會限制 chown 操作，即使你在容器內是 `root` 也一樣做不到（跟一般本機磁碟權限規則不同）。

**重點：這只影響「檔案 owner 標記」這個 metadata，完全不影響檔案內容。** 你可以直接對照看到：

```
27801001 100%    5.62MB/s   00:00:04 (xfer#1, to-check=4/470)
```

這種「檔名 + 100% + 傳輸速度」的行代表**內容確實已經完整傳完**，chown 警告只是插在旁邊的雜訊。

## 現在怎麼辦

**什麼都不用做，讓它繼續跑到底。** 提醒你：

- 最後這個 rsync 指令執行完，**很可能會回報一個非 0 的結束狀態（看起來像「失敗」）**——這是因為每個 chown 失敗都會被算成一個 error，不是資料真的漏傳。不要看到「失敗」就以為要整個重來。
- 跑完之後，用檔案數量／總大小去驗證，而不是看 exit code：

```bash
du -sh data/brca_pilot40_k48
```

在 Mac 上跑這行拿到本機的大小，跟之前你在 pod 上跑的 `du -sh /workspace/mllm_hwsi_ah/data/brca_pilot40_k48` 做比對，數字量級一致（rsync 對每個檔案內容都有做 checksum 驗證，不會傳出破損檔案，只是 owner 沒對齊而已）就代表傳輸是完整、正確的。

## 如果你想以後傳輸不要有這些雜訊（非必要，純粹讓 log 乾淨）

下次可以把 `-a` 拆開、拿掉 `o`/`g` 這兩個 owner/group 相關的字母，改成：

```bash
rsync -rlptDvz --progress -e "ssh -p $RUNPOD_PORT -i $RUNPOD_KEY" data/brca_pilot40_k48 "root@${RUNPOD_HOST}:/workspace/mllm_hwsi_ah/data/"
```

`-rlptD` = recursive + 保留 symlink + 保留權限(permission) + 保留時間戳 + 保留裝置檔案，就是拿掉 `-a` 裡的 `o`(owner)/`g`(group)，效果對你這個用途完全一樣，只是不會再嘗試 chown。但**這次已經在跑的不用中斷重來**，讓它跑完，等一下我們驗證檔案數量/大小就知道有沒有問題。

傳成功了：

```text
rsync: [receiver] chown "/workspace/mllm_hwsi_ah/data/brca_pilot40_k48/regions/stitches/.TCGA-UZ-A9PX-01Z-00-DX1.jpg.tk2nzk" failed: Operation not permitted (1)

sent 4820144972 bytes  received 9448 bytes  5975928 bytes/sec
total size is 5476897396  speedup is 1.14
rsync(78635): warning: child 78636 exited with status 23
```

---

---

## 現在要 rehearsal 以及跑 pilot40:

太好了，資料傳完了（`total size is 5476897396` ≈ 5.1GB，跟 Mac 本機大小對得上，`speedup 1.14` 是正常的；那個 exit status 23 就是先前講的 chown 警告造成的，不用管）。接下來就是照 SOP 走完 rehearsal → full grid 這兩步，我一步步帶你，你只要照順序貼就好。

## 第 1 步：先在 pod 上做個小驗證（不用進 tmux）

```bash
ls -la /workspace/mllm_hwsi_ah/data/pilot40_selection.csv
find /workspace/mllm_hwsi_ah/data/brca_pilot40_k48 -type f | wc -l
```

只要 csv 在、檔案數量看起來是幾百上千個（不是 0 或個位數），就代表資料沒問題，可以往下走。

## 第 2 步：確認環境變數還在，不在就重建

你可能中間開了新的 SSH session，先檢查：

```bash
echo "PROJECT_DIR=$PROJECT_DIR PYTHON_BIN=$PYTHON_BIN"
```

如果是空的，重新載入之前 bootstrap 寫好的持久設定即可（不用整個重跑 bootstrap）：

```bash
source /workspace/.mllm_hwsi_ah/env.sh
```

## 第 3 步：看看之前的 `pilot40` tmux session 還在不在

```bash
tmux ls
```

- 有看到 `pilot40` → `tmux attach -t pilot40` 接回去（會回到當初失敗的畫面，直接往下貼新指令即可，舊的錯誤訊息留著沒差）。
- 沒有任何 session（顯示 `no server running`）→ 開新的：`tmux new -s pilot40`

## 第 4 步：重跑 rehearsal（這次資料應該齊了）

進到 tmux 裡面之後：

```bash
export RUN_TAG="v2_pilot40_rehearsal_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_DIR/logs"
```

```bash
cd "$PROJECT_DIR" && set -o pipefail && "$PYTHON_BIN" scripts/cl_main.py --dataset pilot40 --order main --smoke --device cuda --output-dir "$RUN_DIR/results" --tag "$RUN_TAG" --resume 2>&1 | tee "$RUN_DIR/logs/rehearsal.log"
```

這是小規模測試（`--smoke`），應該幾分鐘內跑完，不會再有 `FileNotFoundError`。跑完把最後幾行貼給我看一下，我確認 `resolved_device=cuda`、CSV 正常再讓你進全網格——這步務必先過，不要跳過直接上全網格。

## 第 5 步（rehearsal 過了才做）：detach，開全網格

`Ctrl-b` `d`（不行就打 `tmux detach-client`），回到 pod 的一般畫面，貼：

```bash
export DATASET=pilot40
export ORDERS=main
export RUN_TAG="v2_pilot40_main_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_ROOT/logs"
tmux new -s pilot40_full
```

進到新 tmux 裡：

```bash
cd "$PROJECT_DIR" && set -o pipefail && RUN_TAG="$RUN_TAG" RUN_ROOT="$RUN_ROOT" DEVICE=cuda DATASET="$DATASET" ORDERS="$ORDERS" bash scripts/run_main_cl.sh 2 2>&1 | tee "$RUN_ROOT/logs/main_grid.log"
```

這是真正的全網格（5 seeds），會跑比較久。跑起來之後 `Ctrl-b d` detach、關掉網頁都沒關係，之後用 `tmux attach -t pilot40_full` 隨時回來看進度。

---

**另外跟你回報一下 Mac 這邊 MPS gate 的狀態**：目前 33 個 unit 跑了 7 個（跑了快 3 小時）。發現一件事要老實跟你說：當初 timing decision rule 選的代表性 unit 是 `eq_pres`（沒用到 M1 的 importance sampler），但實際跑下去 `ia_samp`（有 M1）單一個 unit 就比 `eq_pres` 慢了 3 倍以上——`d_t(s)` 每 50 步要對整個 buffer 做一次 forward pass的成本，被那個代表性 unit 低估了。粗抓一下，整個 gate 完成時間可能落在 8-10 小時左右，不是原本估的 3.7 小時。**但兩者都遠低於 48 小時那個門檻，所以「全部用 MPS 跑」這個決定本身不受影響，只是完成時間比預期晚**。等 gate 真的跑完，我會把實際的總耗時補記到 `docs/method_gate_v033.md` 的 changelog 裡，如實記錄這個 timing 估計偏樂觀的狀況。這邊你不用做任何事，我會持續盯著。

## 接下來 Rehearsal 過了之後：

Rehearsal 過了，完全符合預期：`resolved_device`（`device = cuda`）對、兩個方法（`seqft`/`ours`，`--smoke` 預設跑的兩個）都跑完、CSV 有寫出來、11 秒就結束，速度也證實 CUDA 真的有在用（不是掉回 CPU）。可以放心進到全網格。

## 下一步：detach，開全網格

先離開 tmux（不會中斷任何東西，因為 rehearsal 已經跑完了）：

```bash
tmux detach-client
```

回到 pod 一般畫面後，貼這幾行（環境變數，一行一行貼沒問題）：

```bash
export DATASET=pilot40
export ORDERS=main
export RUN_TAG="v2_pilot40_main_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_ROOT/logs"
tmux new -s pilot40_full
```

進到新開的 `pilot40_full` tmux 裡面，貼這一行（單行，啟動真正的全網格）：

```bash
cd "$PROJECT_DIR" && set -o pipefail && RUN_TAG="$RUN_TAG" RUN_ROOT="$RUN_ROOT" DEVICE=cuda DATASET="$DATASET" ORDERS="$ORDERS" bash scripts/run_main_cl.sh 2 2>&1 | tee "$RUN_ROOT/logs/main_grid.log"
```

這是 5 seeds、多個方法、多個 K 的完整網格，會跑得比剛剛的 rehearsal 久很多（小時等級）。啟動後看到開始有 log 印出來（跟 rehearsal 那種 `[method seedN K=X] AA=...` 同樣格式）就代表正常在跑，這時候就可以：

```bash
tmux detach-client
```

離開、關網頁、甚至斷線都沒關係。之後要看進度回來打：

```bash
tmux attach -t pilot40_full
```

就好。中斷或斷線後想重新接手，記得 `--resume` 已經包在 `run_main_cl.sh` 裡了，直接重跑同一段（`RUN_TAG` 用同一個）即可接續，不會從頭重來。

跑起來之後你就可以先不用管它，我這邊也會持續看 Mac 這邊 gate 的進度，兩邊都完成之前我們先不用做別的事。

---

---

PS1 這次有正確顯示（`root | /workspace/mllm_hwsi_ah (main)`），而且**這次沒有再出現 `tee: ... No such file or directory` 的錯誤**，代表上次那個「新 tmux pane 沒 source 到 env.sh」的坑這次沒有再中招——這是好消息，跟上次那個「秒殺」不是同一種失敗。

但你的疑慮是對的，10 分鐘還是偏快，不能只看「沒有紅字」就當作成功，得實際檢查數字。有兩個可能：

1. **是真的**：pilot40 只有 2 個任務（renal→breast），資料量遠比 4 任務的 `can_dataset` 小很多，且這次用 `run_main_cl.sh 2`（平行度 2）+ `ORDERS=main`（pilot40 protocol 本來就只需要 main 方向，不用 reverse），所以只有 5 個 shard，不是 10 個——真的用 RTX 4090 跑，10 分鐘內跑完 5 個 shard 是有可能的。
2. **又是假的**：某個環境/資料問題讓每個 shard 幾秒內就 crash，只是這次剛好沒有觸發那個特定的 tee 錯誤訊息。

請貼這幾個指令的結果，我幫你判斷是哪一種：

```bash
echo "RUN_TAG=[$RUN_TAG] RUN_ROOT=[$RUN_ROOT]"
wc -l "$RUN_ROOT"/results/main/seed_*/*.csv 2>/dev/null
grep -l "Traceback" "$RUN_ROOT"/logs/cl_main_s*.log 2>/dev/null
tail -n 20 "$RUN_ROOT"/logs/cl_main_s0.log
```

- 第 1 行：確認這次 `RUN_ROOT` 真的有值（不是空的）。
- 第 2 行：每個 seed 的 CSV 應該要有 **21 列左右**（7 個方法 × 3 個 K，pilot40 是 2-task 序列，每個 method×K 組合會產生多筆 stage 記錄，所以實際數字可能更高，但絕對不會是 0 或個位數）。如果是 0 或 1，代表根本沒真的跑。
- 第 3 行：如果有任何一個 log 檔名被印出來，代表那個 shard 有 crash，把對應的 log 貼給我看 traceback。
- 第 4 行：看最後 20 行，確認裡面有實際的訓練/評估輸出（例如 `bal_acc=...`、`wall_s=...` 這類數字），而不是空白或錯誤訊息。

貼結果給我，我馬上告訴你這次是真的成功了、還是要重抓 bug。

---

---

```bash
root | /workspace/mllm_hwsi_ah (main)
# wc -l "$RUN_ROOT"/results/main/seed_*/*.csv 2>/dev/null
    64 /workspace/mllm_hwsi_ah/runs/v2/v2_pilot40_main_cuda_20260815T122732Z/results/main/se
ed_0/cl_main_pilot40_main_v2_pilot40_main_cuda_20260815T122732Z_main_s0.csv
    64 /workspace/mllm_hwsi_ah/runs/v2/v2_pilot40_main_cuda_20260815T122732Z/results/main/se
ed_1/cl_main_pilot40_main_v2_pilot40_main_cuda_20260815T122732Z_main_s1.csv
    64 /workspace/mllm_hwsi_ah/runs/v2/v2_pilot40_main_cuda_20260815T122732Z/results/main/se
ed_2/cl_main_pilot40_main_v2_pilot40_main_cuda_20260815T122732Z_main_s2.csv
    64 /workspace/mllm_hwsi_ah/runs/v2/v2_pilot40_main_cuda_20260815T122732Z/results/main/se
ed_3/cl_main_pilot40_main_v2_pilot40_main_cuda_20260815T122732Z_main_s3.csv
    64 /workspace/mllm_hwsi_ah/runs/v2/v2_pilot40_main_cuda_20260815T122732Z/results/main/se
ed_4/cl_main_pilot40_main_v2_pilot40_main_cuda_20260815T122732Z_main_s4.csv
   320 total
root | /workspace/mllm_hwsi_ah (main)
# grep -l "Traceback" "$RUN_ROOT"/logs/cl_main_s*.log 2>/dev/null
root | /workspace/mllm_hwsi_ah (main)
# tail -n 20 "$RUN_ROOT"/logs/cl_main_s0.log
[seqft seed0 K=4] AA=0.750 T1(bal=0.667, jac=0.02, kl=0.052) (23s)
[ewc seed0 K=1] AA=0.833 T1(bal=0.667, jac=0.17, kl=0.119) (26s)
[ewc seed0 K=2] AA=0.750 T1(bal=0.667, jac=0.06, kl=0.043) (33s)
[ewc seed0 K=4] AA=0.833 T1(bal=0.667, jac=0.05, kl=0.043) (44s)
[lwf seed0 K=1] AA=0.833 T1(bal=0.833, jac=0.50, kl=0.049) (47s)
[lwf seed0 K=2] AA=0.750 T1(bal=0.833, jac=0.11, kl=0.022) (51s)
[lwf seed0 K=4] AA=0.833 T1(bal=0.833, jac=0.05, kl=0.011) (59s)
[replay seed0 K=1] AA=0.917 T1(bal=0.833, jac=0.33, kl=0.052) (62s)
[replay seed0 K=2] AA=0.917 T1(bal=0.833, jac=0.11, kl=0.033) (68s)
[replay seed0 K=4] AA=0.833 T1(bal=0.833, jac=0.08, kl=0.017) (80s)
[distill seed0 K=1] AA=0.833 T1(bal=0.833, jac=0.33, kl=0.038) (83s)
[distill seed0 K=2] AA=0.833 T1(bal=0.667, jac=0.22, kl=0.012) (89s)
[distill seed0 K=4] AA=0.917 T1(bal=0.833, jac=0.05, kl=0.009) (97s)
[ours seed0 K=1] AA=0.833 T1(bal=0.833, jac=0.33, kl=0.040) (100s)
[ours seed0 K=2] AA=0.833 T1(bal=0.667, jac=0.17, kl=0.014) (106s)
[ours seed0 K=4] AA=0.750 T1(bal=0.833, jac=0.24, kl=0.007) (117s)
[joint seed0 K=1] AA=0.917 T1(bal=0.833, jac=0.33, kl=0.069) (121s)
[joint seed0 K=2] AA=0.833 T1(bal=0.833, jac=0.17, kl=0.055) (126s)
[joint seed0 K=4] AA=0.750 T1(bal=0.833, jac=0.08, kl=0.009) (137s)
done -> /workspace/mllm_hwsi_ah/runs/v2/v2_pilot40_main_cuda_20260815T122732Z/results/main/s
eed_0/cl_main_pilot40_main_v2_pilot40_main_cuda_20260815T122732Z_main_s0.csv
root | /workspace/mllm_hwsi_ah (main)
```

好消息：**這次是真的成功了，不是又爆掉**——幫你對一下數字：

## 為什麼可以確定是真的

1. **列數對得上**：每個 seed 的 CSV 是 64 行 = 1 個標頭 + 63 筆資料。pilot40 的 main grid 是 **7 個方法**（`seqft/ewc/lwf/replay/distill/ours/joint`——`ours_uniform` 本來就只在 ablation 網格跑，不在這個 7-method 主網格裡，跟你之前 can_dataset 那邊查到的規則一致）× **3 個 K**（1/2/4）= 21 組，每組 2-task 序列會在 T1、T2 兩個 stage 各記一次舊任務指標 = 3 筆/組 → 21 × 3 = **63 筆，剛好對上**。
2. **完全沒有 Traceback**：`grep -l "Traceback"` 對 5 個 seed 的 log 都跑過，一個都沒印出來，代表 5 個 shard 全部乾淨跑完，不是上次那種「秒殺後假裝成功」。
3. **`tail` 內容是貨真價實的訓練輸出**：`AA=0.750~0.917`、`bal_acc`、`jaccard`（0.02~0.50 都有，method 之間有明顯區分度）、`kl` 都是有意義、會變動的數字，而且**累積秒數是遞增的**（23s → 26s → 33s → ... → 137s），這是持續在跑東西才會有的模式，瞬間 crash 不可能長這樣。
4. **時間也對得上**：單一 seed 跑完全部 21 組要 137 秒（~2.3 分鐘），`run_main_cl.sh 2` 是平行度 2，5 個 seed 兩兩一組跑，理論總時間 ≈ 3 輪 × 137s ≈ 7 分鐘，跟你感覺的「10 分鐘左右」完全吻合——因為 pilot40 只有 2 個任務，資料量比 4 任務的 `can_dataset`小很多，本來就該比較快，不是 bug。

**pilot40 的 main grid 到這裡就正式跑完了**（pilot40 本來就只需要 main 順序，不像 `can_dataset` 還要跑 reverse）。

## 接下來一步：跑聚合腳本 + 存回 git

```bash
cd "$PROJECT_DIR" && "$PYTHON_BIN" scripts/aggregate_results.py --dataset pilot40 --order main --tag "$RUN_TAG" --input-dir "$RUN_ROOT/results" --output-dir "$RUN_ROOT" && git status --short
```

跑完把 `git status --short` 的結果貼給我看一下（應該會看到 `$RUN_ROOT` 底下新增的 results/figures/logs），我再帶你做 `git add` + `commit` + `push`，這樣 Mac 這邊之後 `git pull` 就能直接分析這批 pilot40 的數字。

## 順手回答 tmux 翻頁的問題

不用跟 tmux 的 scrollback 奮鬥了——你剛剛已經在用更好的方法：**直接讀寫在硬碟上的 log 檔案**（`tail`/`cat`/`grep` 那些）永遠比在終端機視窗裡往上滑可靠，因為 log 檔案是完整存檔、不受終端機視窗大小或捲動限制影響，之後不管缺哪一段都可以直接查檔案，不用回頭滾畫面找。如果真的想要 tmux 也能滑鼠捲動，可以貼這行（但不是必要）：

```bash
tmux set -g mouse on
```

---

---

---

---

---

---

---

## 跑的時候出問題，env 不對，的解法：

抓到底層原因了，跟我猜的是同一件事：**這個 tmux pane 沒有 source 到** `env.sh`，所以 PATH 里沒有 venv 的 `bin` 目錄，`run_main_cl.sh` 裡的 `python`（`PY="${PYTHON:-python}"`）就撿到了系統內建的 python3（沒裝 pandas），而不是 bootstrap 幫你裝好一堆套件的 venv python。這也連帶解釋了 `RUN_TAG`/`RUN_ROOT` 沒對上的問題——因為 `env.sh` 沒 source，你之前 `export RUN_TAG=...` 那些也很可能是在別的 shell 打的。

一次解決：在**這個** tmux pane 裡先重新 source 環境，再檢查沒問題才launch。

```bash
source /workspace/.mllm_hwsi_ah/env.sh
which python
python -c "import pandas; print('pandas ok', pandas.__version__)"
```

這三行的預期結果：

- `which python` 應該印出 `/workspace/venvs/mllm_hwsi_ah/bin/python`（不是 `/usr/bin/python`）
- `pandas ok ...` 正常印出版本號，不會 crash

如果這裡就過了，代表環境對了。接著才進 tmux 重新跑（這次連 `PYTHON` 都明確 export，雙重保險，不管 PATH 對不對都會用對的 python）：

```bash
tmux kill-server
tmux new -s pilot40_full
```

進到新的 pane 裡面，**第一件事就是 source env**，然後才 export 這次 run 專用的變數：

```bash
source /workspace/.mllm_hwsi_ah/env.sh
export PYTHON="$PYTHON_BIN"
export DATASET=pilot40
export ORDERS=main
export RUN_TAG="v2_pilot40_main_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_ROOT/logs"
echo "PYTHON=[$PYTHON] RUN_TAG=[$RUN_TAG] RUN_ROOT=[$RUN_ROOT]"
```

確認這行印出來三個中括號都非空、`PYTHON` 是 venv 路徑，貼給我看一眼，我們再繼續下最後的啟動指令：

```bash
cd "$PROJECT_DIR" && set -o pipefail && bash scripts/run_main_cl.sh 2 2>&1 | tee "$RUN_ROOT/logs/main_grid.log"
```

順手把這個坑也記錄進 `docs/RUNPOD_SOP.md`，避免以後重開 pod 又中招——要不要我現在就去補這段（「每次開新 tmux pane 都要先 `source env.sh` 再 export 該次 run 的變數，且已存在的 tmux server 不會自動繼承外面 shell 的新 export」）？

---

---

我順手把「自動裝 rsync」也補進 scripts/runpod_bootstrap.sh（跟裡面本來就有的「自動裝 tmux」用同一套邏輯），這樣以後每次開新 pod 只要跑過 bootstrap，就不會再卡在這個坑。

## 關於你提的 scp 那個寫法

你給的那個範例（`RUNPOD_HOST`/`RUNPOD_PORT` 變數 + `scp -P "$RUNPOD_PORT" file root@host:path`）完全正確、而且更不容易出錯——因為 `scp` 的 `-P`（大寫）跟 `-i` 是**自己獨立的選項**，`root@host:path` 只在最後那個位置出現一次，天生不會有 `-e` 這種「host 被重複塞」的陷阱。這也是為什麼上面我把 CSV 那個小檔案直接用 scp、只有 5.1GB 的資料夾才用 rsync（scp 不支援斷點續傳，這麼大的資料夾一斷線要整個重來，rsync 才值得用）。

跑完之後，回 pod 上驗證：

```bash
ls -la /workspace/mllm_hwsi_ah/data/pilot40_selection.csv
du -sh /workspace/mllm_hwsi_ah/data/brca_pilot40_k48
```

兩個都對了之後，回 `tmux attach -t pilot40`，重新貼一次原本的 rehearsal 指令。

---

---

### 方法：用 rsync 直接從 Mac 推到 RunPod（推薦，你已經有 SSH 連線了）

### 下面這是錯的，不要再試：

回到你 **Mac 本機的終端機**（不是剛剛那個已經 SSH 進 pod 的視窗），先去 RunPod 網頁的 pod 卡片 → **Connect** → 找 **SSH over exposed TCP**（會給你一組 `ssh root@<IP> -p <PORT> -i ~/.ssh/xxx` 這樣的字串，跟你剛剛連進去用的應該是同一組），然後在 Mac 上跑：

```bash
cd /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah
rsync -avz --progress -e "ssh -p <PORT> -i ~/.ssh/xxx" \
  data/pilot40_selection.csv \
  root@<IP>:/workspace/mllm_hwsi_ah/data/

rsync -avz --progress -e "ssh -p <PORT> -i ~/.ssh/xxx" \
  data/brca_pilot40_k48 \
  root@<IP>:/workspace/mllm_hwsi_ah/data/
```

把 `<PORT>`、`<IP>`、`-i ~/.ssh/xxx` 換成你 RunPod Connect 面板給的實際值（如果你當初是用密碼登入沒有 key，就把 `-i ~/.ssh/xxx` 拿掉）。`rsync` 比 `scp` 好的地方是：中斷了重跑同一行指令會**只補傳沒傳完的部分**，5.1GB 這種大小很值得用它。

### 備案：如果 SSH 直連的 port 不通（RunPod proxy 常見狀況）

改用 RunPod 官方的 `runpodctl send/receive`（不用管 SSH port，走它自己的 P2P 通道）：

Mac 上（先裝：`brew install runpod/runpodctl/runpodctl`）：

```bash
runpodctl send /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah/data/pilot40_selection.csv
runpodctl send /Users/aaron/research/01_mllm_hwsi/mllm_hwsi_ah/data/brca_pilot40_k48
```

每個指令會印出一組一次性代碼（像 `8090-tree-happy-lion`），拿去 pod 的終端機貼：

```bash
cd /workspace/mllm_hwsi_ah/data
runpodctl receive 8090-tree-happy-lion
```

### 傳完之後

回到 pod 的 tmux（`tmux attach -t pilot40`），確認檔案在對的位置：

```bash
ls -la /workspace/mllm_hwsi_ah/data/pilot40_selection.csv
du -sh /workspace/mllm_hwsi_ah/data/brca_pilot40_k48
```

再重新貼一次原本的 rehearsal 指令即可（`--resume` 不會受影響，因為根本還沒成功跑過一個 unit）。

要不要我現在就去把 `docs/RUNPOD_SOP.md` 補上「pilot40 資料上傳」這一段（跟 `can_dataset` 一樣走 checksum 驗證），這樣以後不會再漏？

---

---

---

# `[Aaron 筆記結束]`

---

---

## 開機 Aaron 筆記：（此處無法操作，僅做觀念參考）

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

For a short restart card, see `[RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)`.

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
6. Every new tmux pane sources `env.sh` first, before exporting any
   run-specific variable (`RUN_TAG`, `RUN_ROOT`, `DATASET`, ...). See
   "Known pitfall: new tmux panes do not inherit fresh shell exports" below.

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
`WORKSPACE_ROOT=/workspace CAN_ROOT=/workspace/can_dataset_min CACHE_ROOT=/workspace/mllm_hwsi_ah/data/can_cache bash scripts/make_data_manifest.sh`, then rerun the verifier. If a required
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

## Known pitfall: new tmux panes do not inherit fresh shell exports

**Symptom:** a full grid (e.g. `run_main_cl.sh`) exits in ~1 minute instead of
hours, `tee` prints `No such file or directory` for a log path that is missing
its `$RUN_TAG` component, and/or per-shard logs under
`$RUN_ROOT/logs/cl_*.log` show `ModuleNotFoundError: No module named 'pandas'`
(or similar — the interpreter used was a bare system `python`, not the venv).

**Root cause:** once a tmux _server_ is already running on the pod (e.g. from
an earlier rehearsal session), `tmux new -s <name>` only asks that existing
server to open a new session — it does **not** pick up variables the outer
shell `export`ed moments earlier. The new pane's environment can end up
without `env.sh` sourced at all (so `PATH` lacks `$VENV_DIR/bin`, and
`RUN_TAG`/`RUN_ROOT`/`PYTHON_BIN` are unset), which is why `run_main_cl.sh`'s
`PY="${PYTHON:-python}"` silently falls back to a system Python missing the
project's dependencies, and why a `RUN_ROOT` used by an outer `tee` can
disagree with the one used inside the script.

**Fix — always do this immediately after creating or attaching to a tmux
pane, before exporting any run-specific variable:**

```bash
source /workspace/.mllm_hwsi_ah/env.sh
export PYTHON="$PYTHON_BIN"   # belt-and-suspenders: run_main_cl.sh reads $PYTHON, not $PYTHON_BIN
which python                  # must print $VENV_DIR/bin/python, not /usr/bin/python
python -c "import pandas; print('pandas ok', pandas.__version__)"
```

Then export the run-specific variables **in the same pane**, and verify they
are non-empty before launching anything long-running:

```bash
export DATASET=pilot40 ORDERS=main
export RUN_TAG="v2_pilot40_main_cuda_$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="$PROJECT_DIR/runs/v2/$RUN_TAG"
mkdir -p "$RUN_ROOT/logs"
echo "PYTHON=[$PYTHON] RUN_TAG=[$RUN_TAG] RUN_ROOT=[$RUN_ROOT]"
```

If any bracket above is empty, stop and fix it before running
`run_main_cl.sh` — do not assume a fast finish means success; a grid that
completes in about a minute instead of hours has failed silently.

If a run already produced this failure mode, kill the stale tmux server
before retrying so a fresh one starts with a clean, correctly-sourced
environment:

```bash
tmux kill-server
tmux new -s <name>
```

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
