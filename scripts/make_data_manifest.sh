#!/usr/bin/env bash
# Create a volume-side data manifest from a trusted, already-verified copy.
# This is a one-time operation; verification uses the resulting file without
# rehashing the data on every pod boot.

set -euo pipefail
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
CAN_ROOT="${CAN_ROOT:-$WORKSPACE_ROOT/can_dataset_min}"
CACHE_ROOT="${CACHE_ROOT:-$WORKSPACE_ROOT/mllm_hwsi_ah/data/can_cache}"
OUT="${DATA_MANIFEST:-$WORKSPACE_ROOT/SHA256SUMS.txt}"

[[ -d "$CACHE_ROOT" ]] || {
    echo "missing cache root: $CACHE_ROOT" >&2
    exit 1
}
[[ -d "$CAN_ROOT" ]] || {
    echo "missing CAN_ROOT: $CAN_ROOT" >&2
    exit 1
}

shopt -s globstar nullglob
files=()
for path in "$CACHE_ROOT"/**/* \
            "$CAN_ROOT"/tcga_*/table/*.csv \
            "$CAN_ROOT"/tcga_*/datasplit/*.npz; do
    [[ -f "$path" ]] || continue
    files+=("${path#"$WORKSPACE_ROOT"/}")
done
(( ${#files[@]} > 0 )) || {
    echo "no data files found to hash" >&2
    exit 1
}

tmp="${OUT}.tmp.$$"
(
    cd "$WORKSPACE_ROOT"
    sha256sum "${files[@]}"
) >"$tmp"
mv "$tmp" "$OUT"
echo "[data] wrote $OUT (${#files[@]} files)"
