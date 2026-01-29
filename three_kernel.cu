#include <array>
#include <vector>
#include <iostream>
#include <chrono>

#include "board.cu"
#include "three_board.cu"

#include "three_kernel.hpp"
#include "parsing.hpp"

#include "params.hpp"

#ifndef THREE_ENABLE_STATS
#define THREE_ENABLE_STATS 0
#endif

enum class StatId : unsigned {
  NodesVisited,
  VulnerableBranches,
  SemiVulnerableBranches,
  SymmetryForced,
  RowBranches,
  CanonicalSkips,
  Solutions,
  InconsistentNodes,
  Count
};

constexpr unsigned kStatCount = static_cast<unsigned>(StatId::Count);

template <unsigned N, unsigned W>
__device__ unsigned pick_center_col(board_row_t<W> bits) {
  constexpr int center_right = static_cast<int>(N / 2);
  constexpr int center_left = static_cast<int>((N - 1) / 2);
  for (int delta = 0; delta < static_cast<int>(N); ++delta) {
    int c = center_right + delta;
    if (c < static_cast<int>(N) && (bits & (board_row_t<W>(1) << c)))
      return static_cast<unsigned>(c);
    int c2 = center_left - delta;
    if (c2 >= 0 && c2 != c && (bits & (board_row_t<W>(1) << c2)))
      return static_cast<unsigned>(c2);
  }
  return static_cast<unsigned>(find_first_set<W>(bits));
}


#if THREE_ENABLE_STATS
struct SearchStats {
  unsigned long long counters[kStatCount];
};

__device__ SearchStats g_search_stats;

__device__ __forceinline__ void stats_record(StatId id, unsigned long long value = 1ULL) {
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&g_search_stats.counters[static_cast<unsigned>(id)], value);
  }
}

static inline void reset_search_stats() {
  SearchStats zero{};
  cudaMemcpyToSymbol(g_search_stats, &zero, sizeof(SearchStats));
}

static inline void print_stats_snapshot(const SearchStats &stats, unsigned stack_size) {
  std::cerr << "[stats] nodes=" << stats.counters[static_cast<unsigned>(StatId::NodesVisited)]
            << " sym_force=" << stats.counters[static_cast<unsigned>(StatId::SymmetryForced)]
            << " vuln_branches=" << stats.counters[static_cast<unsigned>(StatId::VulnerableBranches)]
            << " semivuln_branches=" << stats.counters[static_cast<unsigned>(StatId::SemiVulnerableBranches)]
            << " row_branches=" << stats.counters[static_cast<unsigned>(StatId::RowBranches)]
            << " canonical_skips=" << stats.counters[static_cast<unsigned>(StatId::CanonicalSkips)]
            << " inconsistent=" << stats.counters[static_cast<unsigned>(StatId::InconsistentNodes)]
            << " stack_size=" << stack_size
            << " solutions=" << stats.counters[static_cast<unsigned>(StatId::Solutions)]
            << std::endl;
}

static inline void maybe_print_stats(std::chrono::steady_clock::time_point &last_print, unsigned stack_size, bool force = false) {
  auto now = std::chrono::steady_clock::now();
  if (!force && now - last_print < std::chrono::seconds(1)) {
    return;
  }

  SearchStats snapshot;
  cudaMemcpyFromSymbol(&snapshot, g_search_stats, sizeof(SearchStats));
  print_stats_snapshot(snapshot, stack_size);
  last_print = now;
}
#else
struct SearchStats {};

__device__ __forceinline__ void stats_record(StatId, unsigned long long = 1ULL) {}
#endif

static inline void init_line_table_host_32() {
  if constexpr (N > 32) {
    return;
  }

  static bool initialized = false;
  static uint32_t *d_line_table = nullptr;
  if (initialized) {
    return;
  }
  initialized = true;

  const unsigned cell_count = N * N;
  const unsigned rows = LINE_TABLE_ROWS;
  const size_t entry_size = rows;
  const size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
  const size_t total_rows = total_entries * entry_size;

  std::vector<uint32_t> host_table(total_rows, 0);

  auto build_start = std::chrono::steady_clock::now();
  for (unsigned py = 0; py < N; ++py) {
    for (unsigned px = 0; px < N; ++px) {
      const unsigned p_idx = py * N + px;
      for (unsigned qy = 0; qy < N; ++qy) {
        for (unsigned qx = 0; qx < N; ++qx) {
          const unsigned q_idx = qy * N + qx;
          if (p_idx == q_idx) {
            continue;
          }

          unsigned pyy = py;
          unsigned pxx = px;
          unsigned qyy = qy;
          unsigned qxx = qx;
          if (pyy > qyy) {
            std::swap(pyy, qyy);
            std::swap(pxx, qxx);
          }

          int dx = static_cast<int>(qxx) - static_cast<int>(pxx);
          unsigned dy = qyy - pyy;
          if (dy == 0) {
            continue;
          }

          const unsigned adx = static_cast<unsigned>(std::abs(dx));
          const unsigned g = std::gcd(adx, dy);
          const unsigned delta_y = dy / g;
          const int delta_x = (dx < 0 ? -1 : 1) * static_cast<int>(adx / g);

          const unsigned p_quo = pyy / delta_y;
          const unsigned p_rem = pyy % delta_y;

          uint32_t *mask = &host_table[(static_cast<size_t>(p_idx) * cell_count + q_idx) * entry_size];
          for (unsigned r = 0; r < N; ++r) {
            if (r % delta_y == p_rem) {
              int col = static_cast<int>(pxx) + (static_cast<int>(r / delta_y) - static_cast<int>(p_quo)) * delta_x;
              if (col >= 0 && col < static_cast<int>(N)) {
                mask[r] |= (uint32_t(1) << col);
              }
            }
          }

          mask[py] &= ~(uint32_t(1) << px);
          mask[qy] &= ~(uint32_t(1) << qx);
        }
      }
    }
  }
  auto build_end = std::chrono::steady_clock::now();

  cudaMalloc((void **)&d_line_table, total_rows * sizeof(uint32_t));
  cudaMemcpy(d_line_table, host_table.data(), total_rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(g_line_table_32, &d_line_table, sizeof(d_line_table));

  auto upload_end = std::chrono::steady_clock::now();
  const auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
  const auto upload_ms = std::chrono::duration_cast<std::chrono::milliseconds>(upload_end - build_end).count();
  std::cerr << "[line_table] build_ms=" << build_ms << " upload_ms=" << upload_ms << std::endl;
}

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
    unsigned col = pick_center_col<N, W>(remaining);
    auto cell = make_cell(col);

    ThreeBoard<N, W> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      DeviceProblem<W> problem = {sub_board.known_on, sub_board.known_off};
      stack_push(stack, problem);
    } else {
      stats_record(StatId::InconsistentNodes);
    }

    tried_board.known_off.set(cell);
    remaining &= ~(board_row_t<W>(1) << col);
  }
}

template <unsigned N, unsigned W>
__device__ void resolve_outcome_cell(const ThreeBoard<N, W> board, cuda::std::pair<unsigned, unsigned> cell, DeviceStack<W> *stack, SolutionBuffer<W> *solution_buffer) {
  {
    ThreeBoard<N, W> sub_board = board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();
    if(sub_board.consistent()) {
      DeviceProblem<W> problem = {sub_board.known_on, sub_board.known_off};
      stack_push(stack, problem);
    } else {
      stats_record(StatId::InconsistentNodes);
    }
  }
  {
    ThreeBoard<N, W> sub_board = board;
    sub_board.known_off.set(cell);
    sub_board.propagate();
    if(sub_board.consistent()) {
      DeviceProblem<W> problem = {sub_board.known_on, sub_board.known_off};
      stack_push(stack, problem);
    } else {
      stats_record(StatId::InconsistentNodes);
    }
  }
}

template <unsigned N, unsigned W>
__global__ void initialize_stack_kernel(DeviceStack<W> *stack, SolutionBuffer<W> *solution_buffer) {
  ThreeBoard<N, W> board;
  resolve_outcome_row<N, W, Axis::Horizontal>(board, DEFAULT_SEED_ROW, stack);
}

template <unsigned N, unsigned W>
__launch_bounds__(32 * WARPS_PER_BLOCK, LAUNCH_MIN_BLOCKS)
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

  if (!board.consistent()) {
    stats_record(StatId::InconsistentNodes);
    return;
  }

  stats_record(StatId::NodesVisited);

  ForcedCell forced{};
  LexStatus canonical = board.is_canonical_orientation_with_forced(forced);
  if (canonical == LexStatus::Greater) {
    stats_record(StatId::CanonicalSkips);
    return;
  }

  if (board.complete()) {
    stats_record(StatId::Solutions);
    solution_buffer_push(solution_buffer, board.known_on);
    return;
  }

  const unsigned on_pop = board.known_on.pop();
  if (forced.has_force && on_pop >= SYM_FORCE_MIN_ON && on_pop <= SYM_FORCE_MAX_ON) {
    stats_record(StatId::SymmetryForced);
    resolve_outcome_cell<N, W>(board, forced.cell, stack, solution_buffer);
    return;
  }

  BitBoard<W> vulnerable = board.vulnerable();
  if (!vulnerable.empty()) {
    auto cell = vulnerable.template first_center_on<N>();
    stats_record(StatId::VulnerableBranches);
    resolve_outcome_cell<N, W>(board, cell, stack, solution_buffer);
    return;
  }

  BitBoard<W> semivulnerable = board.semivulnerable();
  if (!semivulnerable.empty()) {
    auto cell = semivulnerable.template first_center_on<N>();
    stats_record(StatId::SemiVulnerableBranches);
    resolve_outcome_cell<N, W>(board, cell, stack, solution_buffer);
    return;
  }

  auto [row, row_unknown] = board.most_constrained_row();
  stats_record(StatId::RowBranches);
  resolve_outcome_row<N, W, Axis::Horizontal>(board, row, stack);
}

template <unsigned N, unsigned W>
int solve_with_device_stack() {
  init_lookup_tables_host();
  init_relevant_endpoint_host(N);
  init_relevant_endpoint_host_64(N);
  init_line_table_host_32();

  DeviceStack<W> *d_stack;
  SolutionBuffer<W> *d_solution_buffer;
  
  cudaMalloc((void**) &d_stack, sizeof(DeviceStack<W>));
  cudaMalloc((void**) &d_solution_buffer, sizeof(SolutionBuffer<W>));
  
  cudaMemset(d_stack, 0, sizeof(DeviceStack<W>));
  cudaMemset(d_solution_buffer, 0, sizeof(SolutionBuffer<W>));

  initialize_stack_kernel<N, W><<<1, 32>>>(d_stack, d_solution_buffer);

#if THREE_ENABLE_STATS
  reset_search_stats();
  auto last_stats_print = std::chrono::steady_clock::now();
#endif

  Problem<W> *d_compact_tmp = nullptr;
  size_t compact_capacity = 0;

  unsigned start_size;
  cudaMemcpy(&start_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);

  while (start_size > 0) {
    unsigned batch_size = std::min(start_size, static_cast<unsigned>(MAX_BATCH_SIZE));
    unsigned batch_start = start_size - batch_size;

    unsigned blocks = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    work_kernel<N, W><<<blocks, WARPS_PER_BLOCK * 32>>>(d_stack, d_solution_buffer, batch_start, batch_size);

    unsigned new_size;
    cudaMemcpy(&new_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);
    unsigned pushes = new_size - start_size;

    if (pushes > 0) {
      if (pushes > batch_size) {
        if (pushes > compact_capacity) {
          if (d_compact_tmp) {
            cudaFree(d_compact_tmp);
          }
          cudaMalloc((void **)&d_compact_tmp, pushes * sizeof(Problem<W>));
          compact_capacity = pushes;
        }
        cudaMemcpy(d_compact_tmp, &d_stack->problems[start_size],
                   pushes * sizeof(Problem<W>), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&d_stack->problems[batch_start], d_compact_tmp,
                   pushes * sizeof(Problem<W>), cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpy(&d_stack->problems[batch_start], &d_stack->problems[start_size],
                   pushes * sizeof(Problem<W>), cudaMemcpyDeviceToDevice);
      }
    }

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

#if THREE_ENABLE_STATS
    maybe_print_stats(last_stats_print, start_size);
#endif
  }

  cudaDeviceSynchronize();

#if THREE_ENABLE_STATS
  // Print a final snapshot so the last interval isn't lost if the loop exits quickly.
  maybe_print_stats(last_stats_print, start_size, true);
#endif

  if (d_compact_tmp) {
    cudaFree(d_compact_tmp);
  }
  cudaFree(d_stack);
  cudaFree(d_solution_buffer);

  return 0;
}

// Explicitly instantiate the template to the N in params.hpp, or it doesn't get compiled at all.
template int solve_with_device_stack<N, 32>();
