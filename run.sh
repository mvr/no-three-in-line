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
if [[ -n "${GRID_N:-}" ]]; then
  cmake -S /app -B /app/build -DTHREE_N="${GRID_N}"
  cmake --build /app/build --target three three_frontier -j"$(nproc)"
  cp /app/build/three /app/three
  cp /app/build/three_frontier /app/three_frontier
fi

count_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d '[:space:]'
}

pending_count="$(count_dir "${QUEUE_DIR}/pending")"
running_count="$(count_dir "${QUEUE_DIR}/running")"
done_count="$(count_dir "${QUEUE_DIR}/done")"
failed_count="$(count_dir "${QUEUE_DIR}/failed")"

if [[ "${SKIP_SHARD_GEN:-0}" != "1" ]]; then
  if [[ "$pending_count" -eq 0 && "$running_count" -eq 0 && "$done_count" -eq 0 && "$failed_count" -eq 0 ]]; then
    args=(--frontier-bin "$FRONTIER_BIN" --queue-dir "$QUEUE_DIR")
    if [[ -n "${SHARDS_MIN_ON:-}" ]]; then args+=(--min-on "$SHARDS_MIN_ON"); fi
    if [[ -n "${SHARDS_MAX_ON:-}" ]]; then args+=(--max-on "$SHARDS_MAX_ON"); fi
    if [[ -n "${SHARDS_STEPS:-}" ]]; then args+=(--steps "$SHARDS_STEPS"); fi
    if [[ -n "${SHARDS_TARGET:-}" ]]; then args+=(--target-shards "$SHARDS_TARGET"); fi
    python3 scripts/generate_shards.py "${args[@]}"
  fi
fi

cmd=(python3 scripts/run_workers.py --three "$THREE_BIN" --queue-dir "$QUEUE_DIR" \
     --stats-interval "$STATS_INTERVAL" --poll-interval "$POLL_INTERVAL")
if [[ "$REQUEUE_RUNNING" == "1" ]]; then
  cmd+=(--requeue-running)
fi

exec "${cmd[@]}"
