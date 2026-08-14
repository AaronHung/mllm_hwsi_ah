#!/usr/bin/env bash
# Verify persistent RunPod data before any experiment starts.
# Exit 0 only when the required layout exists and available checksum manifests
# pass.  No upload is attempted by this script.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CAN_ROOT="${CAN_ROOT:-/workspace/can_dataset_min}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_DIR/data/can_cache}"
status=0

missing() {
    echo "[data] MISSING: $1" >&2
    status=1
}

[[ -d "$CACHE_ROOT" ]] || missing "$CACHE_ROOT"
[[ -f "$CACHE_ROOT/proj_512to64.npz" ]] || missing \
    "$CACHE_ROOT/proj_512to64.npz"
for cohort in esca lung rcc brca; do
    [[ -d "$CACHE_ROOT/$cohort" ]] || missing "$CACHE_ROOT/$cohort"
    [[ -f "$CACHE_ROOT/${cohort}_manifest.csv" ]] || missing \
        "$CACHE_ROOT/${cohort}_manifest.csv"
    croot="$CAN_ROOT/tcga_${cohort}"
    [[ -d "$croot/table" ]] || missing "$croot/table"
    shopt -s nullglob
    table_files=("$croot/table"/*.csv)
    (( ${#table_files[@]} > 0 )) || missing "$croot/table/*.csv"
    [[ -f "$croot/datasplit/fold_1.npz" ]] || missing \
        "$croot/datasplit/fold_1.npz"
done

check_manifest() {
    local manifest="$1"
    echo "[data] checksum: $manifest"
    (cd "$(dirname "$manifest")" && sha256sum -c "$(basename "$manifest")") \
        || status=1
}

# Always verify the committed source-artifact manifest.
if [[ -f "$PROJECT_DIR/SHA256SUMS.txt" ]]; then
    check_manifest "$PROJECT_DIR/SHA256SUMS.txt"
else
    missing "$PROJECT_DIR/SHA256SUMS.txt"
fi

# Data manifests live on the persistent volume, not in git.  Accept the first
# one present; this avoids hashing hundreds of cache files on every fresh boot.
data_manifest=""
for candidate in "$CAN_ROOT/SHA256SUMS.txt" \
                 "$CACHE_ROOT/SHA256SUMS.txt" \
                 "$PROJECT_DIR/data/SHA256SUMS.txt" \
                 "/workspace/SHA256SUMS.txt"; do
    if [[ -f "$candidate" ]]; then
        data_manifest="$candidate"
        break
    fi
done
if [[ -n "$data_manifest" ]]; then
    check_manifest "$data_manifest"
else
    echo "[data] MISSING: volume-side data SHA256SUMS.txt" >&2
    echo "[data] Structural checks passed, but cache/data hashes are not"
    echo "[data] attestable. Generate it from a trusted data copy with"
    echo "[data] scripts/make_data_manifest.sh, then rerun this check." >&2
    status=1
fi

if [[ "$status" -ne 0 ]]; then
    echo "[data] FAIL: missing or corrupt persistent data; upload only the"
    echo "[data] paths reported above, then rerun scripts/verify_runpod_data.sh."
    exit "$status"
fi
echo "[data] PASS: required can_cache and table/datasplit layout is present."
