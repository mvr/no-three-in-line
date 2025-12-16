#include <cuda/std/utility>
#include <vector>

#include "gtest/gtest.h"

#include "board.cu"
#include "board_test_helpers.cuh"

template <unsigned W>
__global__ void on_cells_kernel(board_row_t<W> *board_rows, cuda::std::pair<uint8_t, uint8_t> *cells) {
  BitBoard<W> board = BitBoard<W>::load(board_rows);
  board.on_cells(cells);
}

template <unsigned W>
void run_on_cells_case(const std::vector<cuda::std::pair<int, int>> &expected_coords) {
  board_row_t<W> *d_board;
  auto board = board_from_cells<W>(expected_coords);
  cudaMalloc((void**)&d_board, W * sizeof(board_row_t<W>));
  cudaMemcpy(d_board, board.data(), W * sizeof(board_row_t<W>), cudaMemcpyHostToDevice);

  cuda::std::pair<uint8_t, uint8_t> *d_cells;
  cudaMalloc((void**)&d_cells, expected_coords.size() * sizeof(cuda::std::pair<uint8_t, uint8_t>));

  constexpr unsigned threads = (W == 64) ? 32 : W;
  on_cells_kernel<W><<<1, threads>>>(d_board, d_cells);

  std::vector<cuda::std::pair<uint8_t, uint8_t>> actual(expected_coords.size());
  cudaMemcpy(actual.data(), d_cells, expected_coords.size() * sizeof(cuda::std::pair<uint8_t, uint8_t>), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < expected_coords.size(); ++i) {
    EXPECT_EQ(static_cast<uint8_t>(expected_coords[i].first), actual[i].first);
    EXPECT_EQ(static_cast<uint8_t>(expected_coords[i].second), actual[i].second);
  }

  cudaFree(d_board);
  cudaFree(d_cells);
}

TEST(BitBoardOnCells, Enumerates32) {
  std::vector<cuda::std::pair<int, int>> coords = {
    {0, 0},
    {3, 0},
    {1, 2},
    {2, 2},
    {5, 2},
    {0, 4}
  };
  run_on_cells_case<32>(coords);
}

TEST(BitBoardOnCells, Enumerates64) {
  std::vector<cuda::std::pair<int, int>> coords = {
    {1, 0},
    {4, 1},
    {0, 2},
    {63, 3},
    {12, 32},
    {7, 33},
    {40, 47}
  };
  run_on_cells_case<64>(coords);
}
