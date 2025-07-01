#pragma once

const unsigned N = 12;
static_assert(N <= 64);

const unsigned WARPS_PER_BLOCK = 4;
const unsigned MAX_BATCH_SIZE = 1<<14;
const unsigned SOFT_BRANCH_1_THRESHOLD = 1000;
const unsigned SOFT_BRANCH_2_THRESHOLD = 5;
