#include <array>
#include <algorithm>
#include <vector>
#include <iostream>

#include "board.cu"
#include "three_board_c4.cu"

#include "three_kernel_c4.hpp"
#include "parsing.hpp"
#include "params.hpp"
#include "three_search.cuh"
#include "three_traits_c4.cuh"

template <unsigned N>
__global__ void init_c4_line_table_kernel(uint32_t *table) {
  constexpr unsigned cell_count = N * N;
  const unsigned pair_idx = blockIdx.x;
  if (pair_idx >= cell_count * cell_count) {
    return;
  }
  const unsigned lane = threadIdx.x & 31;

  const unsigned p_idx = pair_idx / cell_count;
  const unsigned q_idx = pair_idx - p_idx * cell_count;
  const unsigned px = p_idx % N;
  const unsigned py = p_idx / N;
  const unsigned qx = q_idx % N;
  const unsigned qy = q_idx / N;

  ThreeBoardC4<N> board;
  board.known_on.set(px, py);
  board.known_on.set(qx, qy);
  board.eliminate_all_lines_slow({px, py});
  board.eliminate_all_lines_slow({qx, qy});
  board.propagate_slow();
  if (lane < LINE_TABLE_ROWS) {
    table[pair_idx * LINE_TABLE_ROWS + lane] = board.known_off.state;
  }
}

template <unsigned N>
void init_line_table_c4_host() {
  static bool initialized = false;
  static uint32_t *d_table = nullptr;
  if (initialized) {
    return;
  }
  initialized = true;

  constexpr unsigned cell_count = N * N;
  constexpr size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
  constexpr size_t total_rows = total_entries * LINE_TABLE_ROWS;

  cudaMalloc((void **)&d_table, total_rows * sizeof(uint32_t));

  const unsigned blocks = static_cast<unsigned>(total_entries);
  init_c4_line_table_kernel<N><<<blocks, 32>>>(d_table);
  cudaDeviceSynchronize();

  cudaMemcpyToSymbol(g_c4_line_table_32, &d_table, sizeof(d_table));
}

template <unsigned N>
int solve_with_device_stack_c4() {
  return solve_with_device_stack_impl<C4Traits<N>, false>(nullptr, nullptr, nullptr);
}

template int solve_with_device_stack_c4<N>();
