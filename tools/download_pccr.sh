#!/usr/bin/env bash
# Download, verify (size + sha256) and extract the PCCR benchmark
# (Plentiful CARLA Camera Rigs, Bader et al., ECCV 2026, arXiv 2606.27554).
#
#   Source : https://huggingface.co/datasets/timb2001/plentiful-carla-camera-rigs  (CC-BY-NC-4.0)
#   Pinned : dataset revision below + tools/pccr_sha256.txt
#
# Usage:
#   bash tools/download_pccr.sh [DEST_DIR] [RIG ...]
#     DEST_DIR  default: /data2/PCCR
#     RIG ...   default: all 14 rigs
#
# Result (layout expected by the PCCR benchmark scripts):
#   DEST_DIR/<RIG>/<RIG>_infos_{train,val,test,mini}.pkl
#   DEST_DIR/<RIG>/samples/  v1.0-{trainval,test,mini}/  can_bus/  logs/
#
# Re-running is safe: finished rigs (DEST_DIR/<RIG>/.pccr_done) are skipped and
# partially downloaded zips are resumed.
set -uo pipefail

REVISION="18edf4f4bbc94840d7e4026f6f47b063ba173f04"   # HF dataset commit, lastModified 2026-07-07
BASE_URL="https://huggingface.co/datasets/timb2001/plentiful-carla-camera-rigs/resolve/${REVISION}/pccr/data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/pccr_sha256.txt"            # "<sha256>  <bytes>  <file>"

DEST="${1:-/data2/PCCR}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then
  RIGS=("$@")
else
  RIGS=(R1 R1-c6 R1-c10 R1-f R1-r R1-t R2 R3 R4 R5 R6 R7 R8 R9)
fi
ZIPDIR="${DEST}/.zips"
LOG="${DEST}/download.log"

for tool in unzip sha256sum stat awk; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 1; }
mkdir -p "$ZIPDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
expected_sha()  { awk -v f="$1.zip" '$3==f {print $1}' "$MANIFEST"; }
expected_size() { awk -v f="$1.zip" '$3==f {print $2}' "$MANIFEST"; }

download() {  # $1 = rig name
  local rig="$1"
  if command -v aria2c >/dev/null; then
    aria2c -c -x 16 -s 16 -k 64M --file-allocation=none --auto-file-renaming=false \
      --console-log-level=warn --summary-interval=0 \
      -d "$ZIPDIR" -o "${rig}.zip" "${BASE_URL}/${rig}.zip" >>"$LOG" 2>&1
  else
    wget -c -q -O "${ZIPDIR}/${rig}.zip" "${BASE_URL}/${rig}.zip" >>"$LOG" 2>&1
  fi
}

verify_and_extract() {  # $1 = rig name (runs in the background while the next rig downloads)
  local rig="$1"
  local zip="${ZIPDIR}/${rig}.zip" stage="${DEST}/.stage_${rig}" target="${DEST}/${rig}"
  local want_sha want_size got_size got_sha src

  want_sha="$(expected_sha "$rig")"; want_size="$(expected_size "$rig")"
  got_size="$(stat -c %s "$zip")"
  if [ "$got_size" != "$want_size" ]; then
    log "[$rig] SIZE MISMATCH: got $got_size, expected $want_size"; return 1
  fi
  got_sha="$(sha256sum "$zip" | cut -d' ' -f1)"
  if [ "$got_sha" != "$want_sha" ]; then
    log "[$rig] SHA256 MISMATCH: got $got_sha"; return 1
  fi
  log "[$rig] sha256 ok -> extracting"

  mkdir -p "$stage"
  if ! unzip -q -o "$zip" -d "$stage" >>"$LOG" 2>&1; then
    log "[$rig] UNZIP FAILED (staging kept at $stage)"; return 1
  fi

  # The zip may hold the rig files directly or inside one (or more nested) top-level folder(s).
  src="$stage"
  for _ in 1 2 3; do
    local entries=("$src"/*)
    if [ "${#entries[@]}" -eq 1 ] && [ -d "${entries[0]}" ] && [ ! -f "$src/${rig}_infos_train.pkl" ]; then
      src="${entries[0]}"
    else
      break
    fi
  done
  mv -- "$src" "$target" || { log "[$rig] MOVE FAILED ($src -> $target)"; return 1; }
  [ -d "$stage" ] && find "$stage" -depth -type d -empty -delete

  for split in train val test; do
    [ -f "$target/${rig}_infos_${split}.pkl" ] || log "[$rig] WARNING: missing ${rig}_infos_${split}.pkl"
  done
  echo "${want_sha}  ${rig}.zip  revision=${REVISION}" > "$target/.pccr_done"
  rm -f -- "${ZIPDIR:?}/${rig:?}.zip"
  log "[$rig] done -> $target ($(du -sh "$target" | cut -f1))"
}

fail=0
pending=""
for rig in "${RIGS[@]}"; do
  if [ -z "$(expected_sha "$rig")" ]; then log "[$rig] unknown rig (not in manifest)"; fail=1; continue; fi
  if [ -f "${DEST}/${rig}/.pccr_done" ]; then log "[$rig] already done, skipping"; continue; fi
  if [ -e "${DEST}/${rig}" ]; then
    log "[$rig] ${DEST}/${rig} exists without .pccr_done; move it away and re-run"; fail=1; continue
  fi
  log "[$rig] downloading $(expected_size "$rig") bytes"
  if ! download "$rig"; then log "[$rig] DOWNLOAD FAILED (re-run to resume)"; fail=1; continue; fi
  if [ -n "$pending" ]; then wait "$pending" || fail=1; fi
  verify_and_extract "$rig" &
  pending=$!
done
if [ -n "$pending" ]; then wait "$pending" || fail=1; fi

log "finished (fail=${fail}); disk: $(df -h "$DEST" | tail -1 | awk '{print $4 " free of " $2}')"
exit "$fail"
