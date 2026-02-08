#pragma once

#ifdef THREE_N
const unsigned N = THREE_N;
#else
const unsigned N = 17;
#endif

const unsigned WARPS_PER_BLOCK = 4;
const unsigned LAUNCH_MIN_BLOCKS = 12;
const unsigned STACK_CAPACITY = 1 << 23;
const unsigned BATCH_MAX_SIZE = 1 << 16;
const unsigned BATCH_WARMUP_SIZE = 1 << 12;
constexpr float BATCH_FEEDBACK_TARGET_RATIO = 1.2f;
constexpr float BATCH_FEEDBACK_GAIN_RATIO = 0.2f;
const unsigned SOLUTION_BUFFER_CAPACITY = 128;

constexpr bool LINE_TABLE_FULL_WARP_LOAD = false;
constexpr unsigned LINE_TABLE_ROWS = LINE_TABLE_FULL_WARP_LOAD ? 32 : ((N + 7) & ~7u);
static_assert(LINE_TABLE_ROWS <= 32);

constexpr unsigned DEFAULT_SEED_ROW = 0;
