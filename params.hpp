#pragma once

const unsigned N = 13;
static_assert(N <= 64);

const unsigned WARPS_PER_BLOCK = 4;
const unsigned MAX_BATCH_SIZE = 1<<15;
const unsigned STACK_CAPACITY = 1<<20;
const unsigned SOLUTION_BUFFER_CAPACITY = 128;

const unsigned SOFT_BRANCH_1_THRESHOLD = 2;
const unsigned SOFT_BRANCH_2_THRESHOLD = 3;
const unsigned COL_BRANCH_THRESHOLD = 3;
