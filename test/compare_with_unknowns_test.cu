#include <cuda/std/utility>
#include <array>

#include "gtest/gtest.h"

#include "common.hpp"
#include "parsing.hpp"
#include "board.cu"
#include "three_board.cu"

// Kernel launcher -----------------------------------------------------------

template <unsigned N, unsigned W>
__global__ void compare_with_unknowns_kernel(board_row_t<W> *a_on_data, board_row_t<W> *a_off_data,
                                             board_row_t<W> *b_on_data, board_row_t<W> *b_off_data,
                                             LexStatus *result) {
  BitBoard<W> a_on = BitBoard<W>::load(a_on_data);
  BitBoard<W> a_off = BitBoard<W>::load(a_off_data);
  BitBoard<W> b_on = BitBoard<W>::load(b_on_data);
  BitBoard<W> b_off = BitBoard<W>::load(b_off_data);

  *result = compare_with_unknowns<N, W>(a_on, a_off, b_on, b_off);
}

template <unsigned N, unsigned W>
LexStatus run_compare(const board_array_t<W> &a_on, const board_array_t<W> &a_off,
                      const board_array_t<W> &b_on, const board_array_t<W> &b_off) {
  board_row_t<W> *d_a_on, *d_a_off, *d_b_on, *d_b_off;
  LexStatus *d_result;
  LexStatus h_result;

  cudaMalloc((void**) &d_a_on, sizeof(board_array_t<W>));
  cudaMalloc((void**) &d_a_off, sizeof(board_array_t<W>));
  cudaMalloc((void**) &d_b_on, sizeof(board_array_t<W>));
  cudaMalloc((void**) &d_b_off, sizeof(board_array_t<W>));
  cudaMalloc((void**) &d_result, sizeof(LexStatus));

  cudaMemcpy(d_a_on, a_on.data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a_off, a_off.data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_on, b_on.data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_off, b_off.data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);

  compare_with_unknowns_kernel<N, W><<<1, (W == 64 ? 32 : W)>>>(d_a_on, d_a_off, d_b_on, d_b_off, d_result);
  cudaMemcpy(&h_result, d_result, sizeof(LexStatus), cudaMemcpyDeviceToHost);

  cudaFree(d_a_on);
  cudaFree(d_a_off);
  cudaFree(d_b_on);
  cudaFree(d_b_off);
  cudaFree(d_result);

  return h_result;
}

template <unsigned N, unsigned W>
LexStatus run_compare_rle(const char *a_on_rle, const char *a_off_rle,
                          const char *b_on_rle, const char *b_off_rle) {
  return run_compare<N, W>(parse_rle<W>(a_on_rle), parse_rle<W>(a_off_rle),
                           parse_rle<W>(b_on_rle), parse_rle<W>(b_off_rle));
}

// Tests --------------------------------------------------------------------

TEST(CompareWithUnknowns, FullyKnownEqual4) {
  EXPECT_EQ((run_compare_rle<4, 32>("!", "4o$4o$4o$4o!",
                                    "!", "4o$4o$4o$4o!")),
            LexStatus::Equal);
}

TEST(CompareWithUnknowns, FullyKnownLess4) {
  EXPECT_EQ((run_compare_rle<4, 32>("!", "4o$4o$4o$4o!",
                                    "o$!", "b3o$4o$4o$4o!")),
            LexStatus::Less);
}

TEST(CompareWithUnknowns, FullyKnownGreater4) {
  EXPECT_EQ((run_compare_rle<4, 32>("o$!", "b3o$4o$4o$4o!",
                                    "!", "4o$4o$4o$4o!")),
            LexStatus::Greater);
}

TEST(CompareWithUnknowns, SharedUnknownBeforeDifference) {
  EXPECT_EQ((run_compare_rle<4, 32>("bo$!", "2b2o$4o$4o$4o!",
                                    "!", "b3o$4o$4o$4o!")),
            LexStatus::Unknown);
}

TEST(CompareWithUnknowns, UnknownAfterDifferenceDoesNotMatter) {
  EXPECT_EQ((run_compare_rle<4, 32>("o$!", "2b2o$4o$4o$4o!",
                                    "!", "4o$4o$4o$4o!")),
            LexStatus::Greater);
}

TEST(CompareWithUnknowns, FullyKnownEqual6) {
  EXPECT_EQ((run_compare_rle<6, 64>("!", "6o$6o$6o$6o$6o$6o!",
                                    "!", "6o$6o$6o$6o$6o$6o!")),
            LexStatus::Equal);
}

TEST(CompareWithUnknowns, ColumnDifferenceInHighHalf) {
  EXPECT_EQ((run_compare_rle<6, 64>("!", "6o$6o$6o$6o$6o$6o!",
                                    "5bo$!", "5o$6o$6o$6o$6o$6o!")),
            LexStatus::Less);
}

TEST(CompareWithUnknowns, HighColumnSharedUnknownKeepsUnknown) {
  EXPECT_EQ((run_compare_rle<6, 64>("5bo$!", "b4o$6o$6o$6o$6o$6o!",
                                    "!", "b5o$6o$6o$6o$6o$6o!")),
            LexStatus::Unknown);
}

TEST(CompareWithUnknowns, HighColumnKnownForBothResolves) {
  EXPECT_EQ((run_compare_rle<6, 64>("o$!", "b5o$6o$6o$6o$6o$6o!",
                                    "!", "6o$6o$6o$6o$6o$6o!")),
            LexStatus::Greater);
}
