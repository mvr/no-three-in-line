#include <cuda/std/utility>
#include <array>

#include "gtest/gtest.h"

#include "parsing.hpp"
#include "board.cu"

template <unsigned W>
__global__ void mirror_around_kernel(board_row_t<W> *input_board_data, board_row_t<W> *output_board_data, int x, int y) {
  BitBoard<W> input_board = BitBoard<W>::load(input_board_data);
  BitBoard<W> result_board = input_board.mirror_around(cuda::std::make_pair(x, y));
  result_board.save(output_board_data);
}

template <unsigned W>
void test_mirror_around(const std::string &input_rle, const std::string &expected_rle, int x, int y) {
  board_row_t<W> *d_input, *d_output;
  board_array_t<W> h_output;

  cudaMalloc((void**) &d_input, W * sizeof(board_row_t<W>));
  cudaMemcpy(d_input, parse_rle<W>(input_rle).data(), W * sizeof(board_row_t<W>), cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_output, W * sizeof(board_row_t<W>));

  constexpr unsigned threads = (W == 64) ? 32 : W;
  mirror_around_kernel<W><<<1, threads>>>(d_input, d_output, x, y);
  cudaMemcpy(h_output.data(), d_output, W * sizeof(board_row_t<W>), cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected_rle, (to_rle<W, W>(h_output)));

  cudaFree(d_input);
  cudaFree(d_output);
}

TEST(BitBoardMirrorAround, Basics) {
  test_mirror_around<32>("!", "!", 15, 15);

  test_mirror_around<32>("4$4bo$4bo$4bo$4b3o!", "$2b3o$4bo$4bo$4bo!", 4, 4);
  test_mirror_around<32>("o$o$o$3o!", "o!", 0, 0);

  test_mirror_around<32>("obob5o$2b4obo$2bob2o3bo$2o2bo3b2o$2ob7o$b2obob2o$bobo3bobo$obo2bo2b2o$ob2o2b2obo$o2b3ob2o$bob2ob2obo!", "10$9bob2ob2obo$10b2ob3o2bo$9bob2o2b2obo$9b2o2bo2bobo$9bobo3bobo$11b2obob2o$9b7ob2o$9b2o3bo2b2o$9bo3b2obo$11bob4o$10b5obobo!", 9, 10);
}

TEST(BitBoardMirrorAround64, Basics) {
  test_mirror_around<64>("!", "!", 31, 31);
  test_mirror_around<64>("o$o$o$3o!", "o!", 0, 0);
  test_mirror_around<64>("50$5bo!", "30$15bo!", 10, 40);
  test_mirror_around<64>("10$10bo!", "!", 2, 2);
}
