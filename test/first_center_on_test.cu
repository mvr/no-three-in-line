#include <cuda/std/utility>
#include <array>

#include "gtest/gtest.h"

#include "parsing.hpp"
#include "board.cu"
template <unsigned N, unsigned W>
__global__ void first_center_on_kernel(board_row_t<W> *board_data, int *result_x, int *result_y) {
  BitBoard<W> board = BitBoard<W>::load(board_data);
  cuda::std::pair<int, int> result = board.template first_center_on<N>();
  *result_x = result.first;
  *result_y = result.second;
}

template <unsigned N, unsigned W>
void test_first_center_on(const std::string &input_rle, cuda::std::pair<int, int> expected) {
  board_row_t<W> *d_board;
  int *d_result_x, *d_result_y;
  int h_result_x, h_result_y;

  cudaMalloc((void**) &d_board, sizeof(board_array_t<W>));
  cudaMemcpy(d_board, parse_rle<W>(input_rle).data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_result_x, sizeof(int));
  cudaMalloc((void**) &d_result_y, sizeof(int));

  first_center_on_kernel<N, W><<<1, 32>>>(d_board, d_result_x, d_result_y);
  
  cudaMemcpy(&h_result_x, d_result_x, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_result_y, d_result_y, sizeof(int), cudaMemcpyDeviceToHost);

  cuda::std::pair<int, int> result = {h_result_x, h_result_y};

  EXPECT_EQ(expected, result) << "Failed for size " << N << " on " << input_rle;

  cudaFree(d_board);
  cudaFree(d_result_x);
  cudaFree(d_result_y);
}

template <unsigned N>
void test_first_center_on_32(const std::string &input_rle , cuda::std::pair<int, int> expected) {
  test_first_center_on<N, 32>(input_rle, expected);
}

template <unsigned N>
void test_first_center_on_64(const std::string &input_rle , cuda::std::pair<int, int> expected) {
  test_first_center_on<N, 64>(input_rle, expected);
}

TEST(FirstCenterOn, SingleCellAtCenter) {
  test_first_center_on_32<3>("b$bo$b!", {1, 1});
  test_first_center_on_32<4>("b$bo!", {1, 1});
  test_first_center_on_32<5>("2$2bo!", {2, 2});
}

TEST(FirstCenterOn, DominoAtCenter) {
  test_first_center_on_32<3>("$2o!", {1, 1});
  test_first_center_on_32<3>("bo$bo!", {1, 1});
  test_first_center_on_32<3>("$b2o!", {1, 1});
  test_first_center_on_32<3>("$bo$bo!", {1, 1});

  test_first_center_on_32<4>("bo$bo!", {1, 1});
  test_first_center_on_32<4>("$2b2o!", {2, 1});
  test_first_center_on_32<4>("2$2bo$2bo!", {2, 2});
  test_first_center_on_32<4>("2$2o!", {1, 2});
}

TEST(FirstCenterOn, SingleCellVariousPositions) {
  test_first_center_on_32<3>("o!", {0, 0});
  test_first_center_on_32<3>("2bo!", {2, 0});
  test_first_center_on_32<3>("2$o!", {0, 2});
  test_first_center_on_32<3>("2$2bo!", {2, 2});
  test_first_center_on_32<3>("bo!", {1, 0});
  test_first_center_on_32<3>("$o!", {0, 1});
}

TEST(FirstCenterOn, BadCases) {
  test_first_center_on_32<4>("2b2o$4o$o$4o!", {2, 1});
  test_first_center_on_32<4>("$4o$obo$4o!", {2, 2});
  test_first_center_on_32<4>("$4o$3o$4o!", {2, 2});
}

TEST(FirstCenterOn64, SingleCellVariousPositions) {
  test_first_center_on_64<3>("o!", {0, 0});
  test_first_center_on_64<3>("$o!", {0, 1});
  test_first_center_on_64<3>("2$o!", {0, 2});
  test_first_center_on_64<3>("2$2bo!", {2, 2});
  test_first_center_on_64<3>("bo!", {1, 0});
  test_first_center_on_64<3>("$bo!", {1, 1});
}

TEST(FirstCenterOn64, ChoosesClosestToCenter) {
  test_first_center_on_64<4>("2$2bo$2bo!", {2, 2});
  test_first_center_on_64<4>("$2b2o!", {2, 1});
  test_first_center_on_64<5>("2$2bo!", {2, 2});
}


TEST(FirstCenterOn, MultipleCellsClosestToCenter) {
  test_first_center_on_32<3>("o$bo$b!", {1, 1});
  test_first_center_on_32<4>("o$bo$2b$3b!", {1, 1});
  test_first_center_on_32<5>("o$4b$2bo$4b$4b!", {2, 2});
}

TEST(FirstCenterOn, TieBreaking) {
  test_first_center_on_32<3>("bo$b$bo!", {1, 0});
  test_first_center_on_32<4>("bo$b$2b$bo!", {1, 3});
}

TEST(FirstCenterOn, EdgeCases) {
  test_first_center_on_32<2>("o$b!", {0, 0});
  test_first_center_on_32<6>("5b$5b$5b$5b$5b$5bo!", {5, 5});
  test_first_center_on_32<3>("ooo$2b$2b!", {1, 0});
  test_first_center_on_32<3>("o$o$o!", {0, 1});
}

TEST(FirstCenterOn, LargerGrids) {
  test_first_center_on_32<8>("7b$7b$7b$3bo$7b$7b$7b$7b!", {3, 3});
  test_first_center_on_32<10>("9b$9b$9b$9b$4bo$9b$9b$9b$9b$9b!", {4, 4});
}

TEST(FirstCenterOn, MultiplePatterns) {
  test_first_center_on_32<4>("obo$bobo$obo$bobo!", {2, 2});
  test_first_center_on_32<5>("2bo$2bo$ooooo$2bo$2bo!", {2, 2});
  test_first_center_on_32<6>("o$bo$2bo$3bo$4bo$5bo!", {3, 3});
}

TEST(FirstCenterOn, AsymmetricPatterns) {
  test_first_center_on_32<5>("o$o$o$o$ooooo!", {2, 4});
  test_first_center_on_32<4>("oo$obo$2b$3b!", {2, 1});
}
