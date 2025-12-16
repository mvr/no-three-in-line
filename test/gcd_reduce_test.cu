#include <vector>

#include "gtest/gtest.h"

#include "board.cu"
#include "board_test_helpers.cuh"

template <unsigned W>
__global__ void gcd_reduce_kernel(board_row_t<W> *input_rows, board_row_t<W> *output_rows) {
  BitBoard<W> board = BitBoard<W>::load(input_rows);
  BitBoard<W> reduced = board.gcd_reduce();
  reduced.save(output_rows);
}

template <unsigned W>
void run_gcd_reduce_case(const std::vector<cuda::std::pair<int, int>> &input,
                         const std::vector<cuda::std::pair<int, int>> &expected) {
  init_lookup_tables_host();

  board_row_t<W> *d_input;
  board_row_t<W> *d_output;
  auto h_input = board_from_cells<W>(input);

  cudaMalloc((void**)&d_input, W * sizeof(board_row_t<W>));
  cudaMalloc((void**)&d_output, W * sizeof(board_row_t<W>));
  cudaMemcpy(d_input, h_input.data(), W * sizeof(board_row_t<W>), cudaMemcpyHostToDevice);

  constexpr unsigned threads = (W == 64) ? 32 : W;
  gcd_reduce_kernel<W><<<1, threads>>>(d_input, d_output);

  board_array_t<W> h_output{};
  cudaMemcpy(h_output.data(), d_output, W * sizeof(board_row_t<W>), cudaMemcpyDeviceToHost);

  auto expected_board = board_from_cells<W>(expected);
  EXPECT_EQ(expected_board, h_output);

  cudaFree(d_input);
  cudaFree(d_output);
}

TEST(BitBoardGcdReduce, Basic32) {
  std::vector<cuda::std::pair<int, int>> input = {
    {0, 0},
    {0, 6},
    {6, 0},
    {4, 6},
    {10, 15},
    {5, 7}
  };

  std::vector<cuda::std::pair<int, int>> expected = {
    {0, 0},
    {0, 1},
    {1, 0},
    {2, 3},
    {2, 3},
    {5, 7}
  };

  run_gcd_reduce_case<32>(input, expected);
}

TEST(BitBoardGcdReduce, Basic64) {
  std::vector<cuda::std::pair<int, int>> input = {
    {48, 48},
    {12, 20},
    {20, 40},
    {63, 0},
    {0, 35},
    {18, 30},
    {36, 54}
  };

  std::vector<cuda::std::pair<int, int>> expected = {
    {1, 1},
    {3, 5},
    {1, 2},
    {1, 0},
    {0, 1},
    {3, 5},
    {2, 3}
  };

  run_gcd_reduce_case<64>(input, expected);
}
