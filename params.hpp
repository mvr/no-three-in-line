#pragma once

#ifdef THREE_N
const unsigned N = THREE_N;
#else
const unsigned N = 15;
#endif

const unsigned WARPS_PER_BLOCK = 4;
const unsigned LAUNCH_MIN_BLOCKS = 12;
const unsigned STACK_CAPACITY = 1 << 23;
const unsigned BATCH_MIN_SIZE = WARPS_PER_BLOCK * LAUNCH_MIN_BLOCKS;
const unsigned BATCH_MAX_SIZE = 1 << 16;
const unsigned BATCH_WARMUP_SIZE = 1 << 12;
constexpr float BATCH_FEEDBACK_TARGET_RATIO = 1.2f;
constexpr float BATCH_FEEDBACK_GAIN_RATIO = 0.2f;
const unsigned SOLUTION_BUFFER_CAPACITY = 128;

constexpr bool LINE_TABLE_FULL_WARP_LOAD = false;
constexpr unsigned LINE_TABLE_ROWS = LINE_TABLE_FULL_WARP_LOAD ? 32 : ((N + 7) & ~7u);
static_assert(LINE_TABLE_ROWS <= 32);

constexpr unsigned DEFAULT_SEED_ROW = 0;

// Symmetry-forced cell choice threshold
constexpr unsigned SYM_FORCE_MAX_ON = N - 2; // Empirically best at N=12,13

// Score multiplier for rows with exactly one known_on
constexpr unsigned ROW_SINGLE_ON_PENALTY = 5;

constexpr unsigned CELL_BRANCH_ROW_SCORE_THRESHOLD = 20;
constexpr int CELL_BRANCH_W_COL_UNKNOWN = 3;
constexpr int CELL_BRANCH_W_ROW_UNKNOWN = 4;
constexpr int CELL_BRANCH_W_COL_ON = 3;
constexpr int CELL_BRANCH_W_COL_OFF = 8;
constexpr int CELL_BRANCH_W_ENDPOINT_OFF = 0;
constexpr int CELL_BRANCH_W_ENDPOINT_ON = 10;
