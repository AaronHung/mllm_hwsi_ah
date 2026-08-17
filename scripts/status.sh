#!/usr/bin/env bash
# 「現在到底在跑什麼、跑到哪、在等誰」——不需要知道 run tag，也不需要相信任何人的說法。
#
#   bash scripts/status.sh
#
# 全部唯讀。你可以隨時 Ctrl-C，不會影響任何跑批。
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

hr() { printf '%s\n' "────────────────────────────────────────────────────────"; }

hr; echo "現在時間：$(date '+%F %H:%M:%S %Z')"

hr; echo "1. 實際在跑的訓練程序（指令列會直接說出是哪個腳本）"
if ! pgrep -fl "scripts/run_.*\.py|scripts/cl_main\.py" 2>/dev/null; then
    echo "   （沒有任何跑批在跑）"
fi

hr; echo "2. tmux sessions"
tmux ls 2>/dev/null || echo "   （沒有 tmux session）"
echo "   離開已 attach 的 session 請打：tmux detach-client"
echo "   ⚠ 不要按 Ctrl-C —— 那是送 SIGINT 給跑批本身，不是離開 tmux"

hr; echo "3. 最近的跑批進度（依 checkpoints.json 修改時間排序，最新 3 個）"
found=0
for ck in $(ls -t runs/v2/*/checkpoints.json 2>/dev/null | head -3); do
    found=1
    tag="$(basename "$(dirname "$ck")")"
    python3 - "$ck" "$tag" <<'PY'
import json, sys, time, os
ck, tag = sys.argv[1], sys.argv[2]
d = json.load(open(ck))
done = len(d.get("completed", []))
last = d.get("last_unit", {})
age = (time.time() - os.path.getmtime(ck)) / 60
print(f"   {tag}")
print(f"      完成 {done} units | 最後一個：{last.get('method','?')} "
      f"seed{last.get('seed','?')} K={last.get('K','?')} "
      f"({last.get('seconds','?')}s) | {age:.0f} 分鐘前更新")
PY
done
[ "$found" = 0 ] && echo "   （runs/v2/ 底下還沒有 checkpoints.json）"

hr; echo "4. 已寫出的 verdict / 報告（最新 5 個）"
ls -t results/*verdict*.md results/*bitexact*.md results/*closeout*.md 2>/dev/null \
    | head -5 | sed 's/^/   /' || echo "   （無）"

hr; echo "5. git 狀態"
echo "   HEAD: $(git log --oneline -1)"
echo "   與 origin/main: $(git status -sb | head -1)"
dirty=$(git status --porcelain | wc -l | tr -d ' ')
echo "   未提交的變更: ${dirty} 個檔案"

hr; echo "怎麼讀這份輸出"
cat <<'TXT'
   · 第 1 段有程序在跑  → 跑批進行中，你不用做事，等就好
   · 第 1 段空、第 3 段的 units 數已達該批的總數（C0 是 12、v0.34 是 36）
     → 跑完了，去看第 4 段最新的 verdict
   · 第 1 段空、但 units 數沒到 → 跑批中斷了，重下同一條啟動指令即可續跑
     （atomic resume，不會重跑已完成的 unit）
   · verdict 一旦寫出，依協議就是 STOP 等三方 review —— 那時候才輪到你
TXT
hr
