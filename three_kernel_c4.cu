#include <array>
#include <vector>
#include <iostream>
#include <algorithm>

#include "board.cu"
#include "three_board_c4.cu"

#include "three_kernel_c4.hpp"
#include "three_kernel.hpp"
#include "parsing.hpp"
#include "params.hpp"

template <unsigned N>
struct DeviceProblemC4 {
  board_array_t<32> known_on;
  board_array_t<32> known_off;
};

template <unsigned N>
struct DeviceStackC4 {
  DeviceProblemC4<N> problems[STACK_CAPACITY];
  unsigned size;
};

template <unsigned N>
struct SolutionBufferC4 {
  board_array_t<ThreeBoardC4<N>::FULL_W> solutions[SOLUTION_BUFFER_CAPACITY];
  unsigned size;
};

template <unsigned N>
__device__ bool stack_push(DeviceStackC4<N> *stack, const DeviceProblemC4<N> &problem) {
  unsigned old_size;
  if ((threadIdx.x & 31) == 0) {
    old_size = atomicAdd(&stack->size, 1);
  }
  old_size = __shfl_sync(0xffffffff, old_size, 0);

  if (old_size >= STACK_CAPACITY) {
    return false;
  }

  int row = threadIdx.x & 31;
  if (row < static_cast<int>(N)) {
    stack->problems[old_size].known_on[row] = problem.known_on[row];
    stack->problems[old_size].known_off[row] = problem.known_off[row];
  }

  return true;
}

template <unsigned N>
__device__ bool solution_buffer_push(SolutionBufferC4<N> *buffer, const ThreeBoardC4<N> &board) {
  unsigned pos;
  if ((threadIdx.x & 31) == 0) {
    pos = atomicAdd(&buffer->size, 1);
  }
  pos = __shfl_sync(0xffffffff, pos, 0);

  if (pos >= SOLUTION_BUFFER_CAPACITY) {
    return false;
  }

  ThreeBoardC4<N> expanded_board = board;
  auto full = expanded_board.expand_to_full();
  full.known_on.save(buffer->solutions[pos].data());

  return true;
}

template <unsigned N>
__device__ void resolve_outcome_cell(const ThreeBoardC4<N> &board,
                                     cuda::std::pair<unsigned, unsigned> cell,
                                     DeviceStackC4<N> *stack,
                                     SolutionBufferC4<N> *solution_buffer) {
  {
    ThreeBoardC4<N> sub_board = board;
    sub_board.known_on.set(static_cast<int>(cell.first), static_cast<int>(cell.second));
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();
    if (sub_board.consistent()) {
      DeviceProblemC4<N> problem;
      sub_board.known_on.save(problem.known_on.data());
      sub_board.known_off.save(problem.known_off.data());
      stack_push(stack, problem);
    }
  }
  {
    ThreeBoardC4<N> sub_board = board;
    sub_board.known_off.set(static_cast<int>(cell.first), static_cast<int>(cell.second));
    sub_board.propagate();
    if (sub_board.consistent()) {
      DeviceProblemC4<N> problem;
      sub_board.known_on.save(problem.known_on.data());
      sub_board.known_off.save(problem.known_off.data());
      stack_push(stack, problem);
    }
  }
}

template <unsigned N>
__global__ void initialize_stack_kernel(DeviceStackC4<N> *stack) {
  if ((threadIdx.x & 31) == 0) {
    stack->size = 1;
  }
  if ((threadIdx.x & 31) < N) {
    stack->problems[0].known_on[threadIdx.x & 31] = 0;
    stack->problems[0].known_off[threadIdx.x & 31] = 0;
  }
}

template <unsigned N>
__global__ void work_kernel(DeviceStackC4<N> *stack,
                            SolutionBufferC4<N> *solution_buffer,
                            unsigned batch_start,
                            unsigned batch_size) {
  const unsigned problem_offset = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / 32);
  const unsigned problem_idx = batch_start + problem_offset;

  if (problem_offset >= batch_size)
    return;

  ThreeBoardC4<N> board;
  board.known_on = BitBoard<32>::load(stack->problems[problem_idx].known_on.data());
  board.known_off = BitBoard<32>::load(stack->problems[problem_idx].known_off.data());

  board.propagate();
  if (!board.consistent())
    return;

  BitBoard<32> bounds = ThreeBoardC4<N>::bounds();
  BitBoard<32> unknown = ~(board.known_on | board.known_off);
  unknown &= bounds;

  if (unknown.empty()) {
    solution_buffer_push(solution_buffer, board);
    return;
  }


  cuda::std::pair<unsigned, unsigned> cell;

  BitBoard<32> vulnerable = board.vulnerable();
  if(!vulnerable.empty()) {
    auto vulnerable_choice = vulnerable.first_origin_on<N>();
    cell = {static_cast<unsigned>(vulnerable_choice.first),
            static_cast<unsigned>(vulnerable_choice.second)};
  } else {
    BitBoard<32> unknown = ~(board.known_on | board.known_off) & board.bounds();

    auto [row, _] = board.most_constrained_row();
    board_row_t<32> row_unknown = unknown.row(row);

    if (row_unknown != 0) {
      unsigned col = find_first_set<32>(row_unknown);
      cell = {col, row};
    } else {
      board_row_t<32> col_unknown = unknown.column(row);
      unsigned col = find_first_set<32>(col_unknown);
      cell = {row, col};
    }
  }

  resolve_outcome_cell(board, cell, stack, solution_buffer);
}

template <unsigned N>
int solve_with_device_stack_c4() {
  init_lookup_tables_host();
  init_relevant_endpoint_host(ThreeBoardC4<N>::FULL_N);
  init_relevant_endpoint_host_64(ThreeBoardC4<N>::FULL_N);

  DeviceStackC4<N> *d_stack;
  SolutionBufferC4<N> *d_solution_buffer;

  cudaMalloc((void**)&d_stack, sizeof(DeviceStackC4<N>));
  cudaMalloc((void**)&d_solution_buffer, sizeof(SolutionBufferC4<N>));

  cudaMemset(d_stack, 0, sizeof(DeviceStackC4<N>));
  cudaMemset(d_solution_buffer, 0, sizeof(SolutionBufferC4<N>));

  initialize_stack_kernel<N><<<1, 32>>>(d_stack);

  unsigned start_size;
  cudaMemcpy(&start_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);

  while (start_size > 0) {
    unsigned batch_size = std::min(start_size, static_cast<unsigned>(MAX_BATCH_SIZE));
    unsigned batch_start = start_size - batch_size;

    unsigned blocks = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    work_kernel<N><<<blocks, WARPS_PER_BLOCK * 32>>>(d_stack, d_solution_buffer, batch_start, batch_size);

    unsigned new_size;
    cudaMemcpy(&new_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaMemcpy(&d_stack->problems[batch_start], &d_stack->problems[start_size],
               (new_size - start_size) * sizeof(DeviceProblemC4<N>), cudaMemcpyDeviceToDevice);

    start_size = new_size - batch_size;
    cudaMemcpy(&d_stack->size, &start_size, sizeof(unsigned), cudaMemcpyHostToDevice);

    unsigned solution_count;
    cudaMemcpy(&solution_count, &d_solution_buffer->size, sizeof(unsigned), cudaMemcpyDeviceToHost);

    if (solution_count > 0) {
      std::vector<board_array_t<ThreeBoardC4<N>::FULL_W>> host_solutions(solution_count);
      for (unsigned i = 0; i < solution_count; ++i) {
        cudaMemcpy(&host_solutions[i], &d_solution_buffer->solutions[i],
                   sizeof(board_array_t<ThreeBoardC4<N>::FULL_W>), cudaMemcpyDeviceToHost);
        std::cout << to_rle<ThreeBoardC4<N>::FULL_N, ThreeBoardC4<N>::FULL_W>(host_solutions[i]) << std::endl;
        exit(0);
      }
      cudaMemset(&d_solution_buffer->size, 0, sizeof(unsigned));
    }
  }

  cudaFree(d_stack);
  cudaFree(d_solution_buffer);

  return 0;
}

template int solve_with_device_stack_c4<N>();
