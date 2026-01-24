#pragma once

const unsigned N = 13;
static_assert(N <= 64);

const unsigned WARPS_PER_BLOCK = 4;
const unsigned LAUNCH_MIN_BLOCKS = 12;
const unsigned MAX_BATCH_SIZE = 1<<15;
const unsigned STACK_CAPACITY = 1<<20;
const unsigned SOLUTION_BUFFER_CAPACITY = 128;

const unsigned SOFT_BRANCH_1_THRESHOLD = 2;
const unsigned SOFT_BRANCH_2_THRESHOLD = 3;
// Multiplier for rows with exactly one known_on when estimating constraint.
constexpr unsigned ROW_SINGLE_ON_PENALTY = 5;

constexpr bool LINE_TABLE_FULL_WARP_LOAD = false;
constexpr unsigned LINE_TABLE_ROWS = LINE_TABLE_FULL_WARP_LOAD ? 32 : ((N + 7) & ~7u);
static_assert(LINE_TABLE_ROWS <= 32);

constexpr unsigned DEFAULT_SEED_ROW = N / 2;
static_assert(DEFAULT_SEED_ROW < N);
