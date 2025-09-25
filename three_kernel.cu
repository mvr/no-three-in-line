#include <array>
#include <vector>
#include <numeric>
#include <iostream>

#include "board.cu"
#include "three_board.cu"

#include "three_kernel.hpp"
#include "parsing.hpp"

#include "params.hpp"

template<unsigned W>
struct DeviceProblem {
  BitBoard<W> known_on;
  BitBoard<W> known_off;
};

template <unsigned W>
__device__ bool stack_push(DeviceStack<W> *stack, const DeviceProblem<W> &problem) {
  unsigned old_size;
  
  if ((threadIdx.x & 31) == 0) {
    old_size = atomicAdd(&stack->size, 1);
  }
  old_size = __shfl_sync(0xffffffff, old_size, 0);
  
  if (old_size >= STACK_CAPACITY) {
    return false; // Stack overflow
  }
  
  problem.known_on.save(stack->problems[old_size].known_on.data());
  problem.known_off.save(stack->problems[old_size].known_off.data());

  return true;
}

template <unsigned W>
__device__ bool solution_buffer_push(SolutionBuffer<W> *buffer, BitBoard<W> &solution) {
  unsigned pos;
  if ((threadIdx.x & 31) == 0) {
    pos = atomicAdd(&buffer->size, 1);
  }
  pos = __shfl_sync(0xffffffff, pos, 0);
  
  if (pos >= SOLUTION_BUFFER_CAPACITY) {
    return false;
  }

  solution.save(buffer->solutions[pos].data());

  return true;
}

template <unsigned N, unsigned W, Axis Dir>
__device__ void resolve_outcome_row(const ThreeBoard<N, W> board, unsigned ix, DeviceStack<W> *stack) {
  board_row_t<W> line_known_on, line_known_off;
  if constexpr (Dir == Axis::Horizontal) {
    line_known_on = board.known_on.row(ix);
    line_known_off = board.known_off.row(ix);
  } else { // Axis::Vertical
    line_known_on = board.known_on.column(ix);
    line_known_off = board.known_off.column(ix);
  }

  board_row_t<W> remaining = ~line_known_on & ~line_known_off & (((board_row_t<W>)1 << N) - 1);

  auto make_cell = [&](unsigned c) {
    if constexpr (Dir == Axis::Horizontal) {
      return cuda::std::pair<unsigned, unsigned>{c, ix};
    } else {
      return cuda::std::pair<unsigned, unsigned>{ix, c};
    }
  };

  ThreeBoard<N, W> tried_board = board;
  if (line_known_on == 0) {
    unsigned keep = find_last_set<W>(remaining);
    remaining &= ~(board_row_t<W>(1) << keep);
  }

  while(remaining != 0) {
    auto cell = make_cell(find_first_set<W>(remaining));

    ThreeBoard<N, W> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_one_hop(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      DeviceProblem<W> problem = {sub_board.known_on, sub_board.known_off};
      stack_push(stack, problem);
    }

    tried_board.known_off.set(cell);
    remaining &= (remaining - 1);
  }
}

template <unsigned N, unsigned W>
__device__ void resolve_outcome_cell(const ThreeBoard<N, W> board, cuda::std::pair<unsigned, unsigned> cell, DeviceStack<W> *stack, SolutionBuffer<W> *solution_buffer) {
  {
    ThreeBoard<N, W> sub_board = board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_one_hop(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();
    if(sub_board.consistent()) {
      DeviceProblem<W> problem = {sub_board.known_on, sub_board.known_off};
      stack_push(stack, problem);
    }
  }
  {
    ThreeBoard<N, W> sub_board = board;
    sub_board.known_off.set(cell);
    sub_board.propagate();
    if(sub_board.consistent()) {
      DeviceProblem<W> problem = {sub_board.known_on, sub_board.known_off};
      stack_push(stack, problem);
    }
  }
}

template <unsigned N, unsigned W>
__global__ void initialize_stack_kernel(DeviceStack<W> *stack, SolutionBuffer<W> *solution_buffer) {
  ThreeBoard<N, W> board;
  resolve_outcome_row<N, W, Axis::Horizontal>(board, N/2, stack);
}

template <unsigned N, unsigned W>
__launch_bounds__(32 * WARPS_PER_BLOCK, 12)
__global__ void work_kernel(DeviceStack<W> *stack, SolutionBuffer<W> *solution_buffer, unsigned batch_start, unsigned batch_size) {
  const unsigned problem_offset = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / 32);
  const unsigned problem_idx = batch_start + problem_offset;

  if(problem_offset >= batch_size)
    return;

  DeviceProblem<W> problem;
  problem.known_on = BitBoard<W>::load(stack->problems[problem_idx].known_on.data());
  problem.known_off = BitBoard<W>::load(stack->problems[problem_idx].known_off.data());

  ThreeBoard<N, W> board;
  board.known_on = problem.known_on;
  board.known_off = problem.known_off;

  if (!board.consistent())
    return;

  if (board.is_canonical_orientation() == LexStatus::Greater) {
    return;
  }

  if (board.unknown_pop() == 0) {
    solution_buffer_push(solution_buffer, board.known_on);
    return;
  }

  BitBoard<W> vulnerable = board.vulnerable();
  if(!vulnerable.empty()) {
    auto cell = vulnerable.template first_center_on<N>();
    resolve_outcome_cell<N, W>(board, cell, stack, solution_buffer);
    return;
  }

  auto [row, row_unknown] = board.most_constrained_row();
  resolve_outcome_row<N, W, Axis::Horizontal>(board, row, stack);
}

template <unsigned N, unsigned W>
int solve_with_device_stack() {
  init_lookup_tables_host();
  init_relevant_endpoint_host();
  init_relevant_endpoint_host_64();

  DeviceStack<W> *d_stack;
  SolutionBuffer<W> *d_solution_buffer;
  
  cudaMalloc((void**) &d_stack, sizeof(DeviceStack<W>));
  cudaMalloc((void**) &d_solution_buffer, sizeof(SolutionBuffer<W>));
  
  cudaMemset(d_stack, 0, sizeof(DeviceStack<W>));
  cudaMemset(d_solution_buffer, 0, sizeof(SolutionBuffer<W>));

  initialize_stack_kernel<N, W><<<1, 32>>>(d_stack, d_solution_buffer);

  unsigned start_size;
  cudaMemcpy(&start_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);

  while (start_size > 0) {
    unsigned batch_size = std::min(start_size, static_cast<unsigned>(MAX_BATCH_SIZE));
    unsigned batch_start = start_size - batch_size;

    unsigned blocks = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    work_kernel<N, W><<<blocks, WARPS_PER_BLOCK * 32>>>(d_stack, d_solution_buffer, batch_start, batch_size);

    unsigned new_size;
    cudaMemcpy(&new_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaMemcpy(&d_stack->problems[batch_start], &d_stack->problems[start_size],
               (new_size - start_size) * sizeof(Problem<W>), cudaMemcpyDeviceToDevice);

    start_size = new_size - batch_size;

    cudaMemcpy(&d_stack->size, &start_size, sizeof(unsigned), cudaMemcpyHostToDevice);
    
    unsigned solution_count;
    cudaMemcpy(&solution_count, &d_solution_buffer->size, sizeof(unsigned), cudaMemcpyDeviceToHost);
    
    if (solution_count > 0) {
      std::vector<board_array_t<W>> solutions(solution_count);
      for (unsigned i = 0; i < solution_count; i++) {
        cudaMemcpy(&solutions[i], &d_solution_buffer->solutions[i], sizeof(board_array_t<W>), cudaMemcpyDeviceToHost);
        std::cout << to_rle<N, W>(solutions[i]) << std::endl;
      }
      
      cudaMemset(&d_solution_buffer->size, 0, sizeof(unsigned));
    }
  }

  cudaFree(d_stack);
  cudaFree(d_solution_buffer);

  return 0;
}

// Explicitly instantiate the template to the N in params.hpp, or it doesn't get compiled at all.
template int solve_with_device_stack<N, 32>();

void init_lookup_tables_host() {
  unsigned char host_div_gcd_table[64][64];

  for (unsigned i = 1; i < 64; i++) {
    for (unsigned j = 1; j < 64; j++) {
      host_div_gcd_table[i][j] = i / std::gcd(i, j);
    }
  }

  cudaMemcpyToSymbol(div_gcd_table, host_div_gcd_table, sizeof(host_div_gcd_table));
}

bool relevant_endpoint(std::pair<unsigned, unsigned> q) {
  if (q.first == 0 || q.second == 0)
    return false;

  unsigned factor = std::gcd(q.first, q.second);

  if (factor > 1)
    // There is a point between that needs checking
    return true;

  if(q.first * 3 >= N || q.second * 3 >= N)
    // There is no way a fourth point can fit in the square
    return false;

  return true;
}

void init_relevant_endpoint_host() {
  uint64_t host_relevant_endpoint_table[64] = {0};

  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < N; j++) {
     bool relevant = relevant_endpoint({i, j});
     if(relevant) {
       host_relevant_endpoint_table[32+j] |= 1ULL << (32+i);
       host_relevant_endpoint_table[32+j] |= 1ULL << (32-i);
       host_relevant_endpoint_table[32-j] |= 1ULL << (32+i);
       host_relevant_endpoint_table[32-j] |= 1ULL << (32-i);
     }
    }
  }

  cudaMemcpyToSymbol(relevant_endpoint_table, host_relevant_endpoint_table, sizeof(host_relevant_endpoint_table));
}

// 128x128 grid from (-64, -64) to (63, 63)
// Layout A[0], A[1]
//        A[2], A[3]
//        A[4], A[5]
// Etc
// So (0, 0) is stored in A[129], awkwardly

void init_relevant_endpoint_host_64() {
  uint64_t host_relevant_endpoint_table_64[256] = {0};

  for (unsigned i = 1; i < N; i++) {
    for (unsigned j = 1; j < N; j++) {
     bool relevant = relevant_endpoint({i, j});
     if(relevant) {
       host_relevant_endpoint_table_64[(64+j)*2 + 0] |= 1ULL << (64-i);
       host_relevant_endpoint_table_64[(64+j)*2 + 1] |= 1ULL << i;
       host_relevant_endpoint_table_64[(64-j)*2 + 0] |= 1ULL << (64-i);
       host_relevant_endpoint_table_64[(64-j)*2 + 1] |= 1ULL << i;
     }
    }
  }

  cudaMemcpyToSymbol(relevant_endpoint_table_64, host_relevant_endpoint_table_64, sizeof(host_relevant_endpoint_table_64));
}
