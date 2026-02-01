#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUNPOD_START_SERVICES:-1}" == "1" ]]; then
  /start.sh &
  sleep 2
fi

cd /app

QUEUE_DIR="${QUEUE_DIR:-/workspace/queue-${GRID_N:-default}}"
THREE_BIN="${THREE_BIN:-/app/three}"
FRONTIER_BIN="${FRONTIER_BIN:-/app/three_frontier}"
STATS_INTERVAL="${STATS_INTERVAL:-60}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
REQUEUE_RUNNING="${REQUEUE_RUNNING:-1}"

if [[ "${IDLE_FOREVER:-0}" == "1" ]]; then
  echo "[run.sh] IDLE_FOREVER=1; sleeping indefinitely."
  sleep infinity
fi
if [[ -n "${GRID_N:-}" ]]; then
  cmake -S /app -B /app/build -DTHREE_N="${GRID_N}"
  cmake --build /app/build --target three three_frontier -j"$(nproc)"
  cp /app/build/three /app/three
  cp /app/build/three_frontier /app/three_frontier
fi

if [[ "${SKIP_SHARD_GEN:-0}" != "1" ]]; then
  if [[ ! -f "${QUEUE_DIR}/queue.db" ]]; then
    args=(--frontier-bin "$FRONTIER_BIN" --queue-dir "$QUEUE_DIR")
    if [[ -n "${SHARDS_MIN_ON:-}" ]]; then args+=(--min-on "$SHARDS_MIN_ON"); fi
    if [[ -n "${SHARDS_MAX_ON:-}" ]]; then args+=(--max-on "$SHARDS_MAX_ON"); fi
    if [[ -n "${SHARDS_STEPS:-}" ]]; then args+=(--steps "$SHARDS_STEPS"); fi
    if [[ -n "${SHARDS_TARGET:-}" ]]; then args+=(--target-shards "$SHARDS_TARGET"); fi
    python3 -u scripts/generate_shards.py "${args[@]}"
  fi
fi

cmd=(python3 -u scripts/run_workers.py --three "$THREE_BIN" --queue-dir "$QUEUE_DIR" \
     --stats-interval "$STATS_INTERVAL" --poll-interval "$POLL_INTERVAL")

"${cmd[@]}"
rc=$?

if command -v runpodctl >/dev/null 2>&1; then
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "[run.sh] stopping pod ${RUNPOD_POD_ID}"
    runpodctl stop pod "${RUNPOD_POD_ID}"
  else
    echo "[run.sh] RUNPOD_POD_ID not set; cannot stop pod"
  fi
else
  echo "[run.sh] runpodctl not found; cannot stop pod"
fi

exit $rc
