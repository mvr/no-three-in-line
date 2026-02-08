#pragma once

#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>

#include "three_kernel.hpp"
#include "parsing.hpp"
#include "params.hpp"

#ifndef THREE_ENABLE_STATS
#define THREE_ENABLE_STATS 0
#endif

struct SearchStats {
  unsigned long long counters[static_cast<unsigned>(StatId::Count)];
};

#if THREE_ENABLE_STATS
static __device__ SearchStats g_search_stats;
static std::chrono::steady_clock::time_point g_last_stats_print;

__device__ __forceinline__ void stats_record(StatId id, unsigned long long value = 1ULL) {
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&g_search_stats.counters[static_cast<unsigned>(id)], value);
  }
}

inline void reset_search_stats() {
  SearchStats zero{};
  cudaMemcpyToSymbol(g_search_stats, &zero, sizeof(SearchStats));
  g_last_stats_print = std::chrono::steady_clock::now();
}

inline void maybe_print_stats(unsigned stack_size,
                              unsigned batch_size,
                              float batch_scale,
                              float push_ratio,
                              bool force = false) {
  auto now = std::chrono::steady_clock::now();
  if (!force && now - g_last_stats_print < std::chrono::seconds(10)) {
    return;
  }

  SearchStats snapshot;
  cudaMemcpyFromSymbol(&snapshot, g_search_stats, sizeof(SearchStats));
  std::cerr << "[stats] nodes=" << snapshot.counters[static_cast<unsigned>(StatId::NodesVisited)]
            << " sym_force=" << snapshot.counters[static_cast<unsigned>(StatId::SymmetryForced)]
            << " vuln_branches=" << snapshot.counters[static_cast<unsigned>(StatId::VulnerableBranches)]
            << " semivuln_branches=" << snapshot.counters[static_cast<unsigned>(StatId::SemiVulnerableBranches)]
            << " quasivuln_branches=" << snapshot.counters[static_cast<unsigned>(StatId::QuasiVulnerableBranches)]
            << " cell_branches=" << snapshot.counters[static_cast<unsigned>(StatId::CellBranches)]
            << " row_branches=" << snapshot.counters[static_cast<unsigned>(StatId::RowBranches)]
            << " canonical_skips=" << snapshot.counters[static_cast<unsigned>(StatId::CanonicalSkips)]
            << " inconsistent=" << snapshot.counters[static_cast<unsigned>(StatId::InconsistentNodes)]
            << " stack_size=" << stack_size
            << " batch_size=" << batch_size
            << " batch_scale=" << batch_scale
            << " push_ratio=" << push_ratio
            << " solutions=" << snapshot.counters[static_cast<unsigned>(StatId::Solutions)]
            << std::endl;

  g_last_stats_print = now;
}

inline void stats_final(unsigned stack_size, float batch_scale) {
  maybe_print_stats(stack_size, 1, batch_scale, 0.0f, true);
}
#else
__device__ __forceinline__ void stats_record(StatId, unsigned long long = 1ULL) {}
inline void reset_search_stats() {}
inline void maybe_print_stats(unsigned, unsigned, float, float, bool = false) {}
inline void stats_final(unsigned, float) {}
#endif

template <unsigned W>
__device__ bool stack_push(DeviceStack<W> *stack,
                           const BitBoard<W> &known_on,
                           const BitBoard<W> &known_off) {
  unsigned old_size;
  if ((threadIdx.x & 31) == 0) {
    old_size = atomicAdd(&stack->size, 1);
  }
  old_size = __shfl_sync(0xffffffff, old_size, 0);

  if (old_size >= STACK_CAPACITY) {
    if ((threadIdx.x & 31) == 0) {
      atomicAdd(&stack->overflow, 1);
    }
    return false;
  }

  known_on.save(stack->problems[old_size].known_on.data());
  known_off.save(stack->problems[old_size].known_off.data());
  return true;
}

template <unsigned W>
__device__ bool output_buffer_push(OutputBuffer<W> *buffer,
                                   const BitBoard<W> &known_on,
                                   const BitBoard<W> &known_off) {
  unsigned pos;
  if ((threadIdx.x & 31) == 0) {
    pos = atomicAdd(&buffer->size, 1);
  }
  pos = __shfl_sync(0xffffffff, pos, 0);

  if (pos >= buffer->capacity) {
    if ((threadIdx.x & 31) == 0) {
      atomicAdd(&buffer->overflow, 1);
    }
    return false;
  }

  known_on.save(buffer->entries[pos].known_on.data());
  known_off.save(buffer->entries[pos].known_off.data());
  return true;
}

template <typename Traits>
__device__ int cell_branch_score(const typename Traits::Board &board,
                                 unsigned x,
                                 unsigned y) {
  constexpr unsigned N = Traits::kN;
  constexpr unsigned W = Traits::kW;
  using Board = typename Traits::Board;

  board_row_t<W> row_on_bits = board.known_on.row(y);
  board_row_t<W> row_off_bits = board.known_off.row(y);
  unsigned row_on = popcount<W>(row_on_bits);
  unsigned row_off = popcount<W>(row_off_bits);
  unsigned row_unknown = N - row_on - row_off;

  board_row_t<W> col_on_mask = board.known_on.column(x);
  board_row_t<W> col_off_mask = board.known_off.column(x);
  unsigned col_on = popcount<W>(col_on_mask);
  unsigned col_off = popcount<W>(col_off_mask);
  unsigned col_unknown = N - col_on - col_off;

  auto cell = cuda::std::pair<unsigned, unsigned>{x, y};
  BitBoard<W> endpoint = Board::relevant_endpoint(cell);
  unsigned endpoint_on = (endpoint & board.known_on).pop();

  int score = 0;
  score += Traits::kCellBranchWColUnknown * static_cast<int>(col_unknown);
  score += Traits::kCellBranchWRowUnknown * static_cast<int>(row_unknown);
  score += Traits::kCellBranchWColOn * static_cast<int>(col_on);
  score -= Traits::kCellBranchWColOff * static_cast<int>(col_off);
  score -= Traits::kCellBranchWEndpointOn * static_cast<int>(endpoint_on);
  return score;
}

template <typename Traits>
__device__ typename Traits::Cell pick_best_branch_cell(const typename Traits::Board &board) {
  constexpr unsigned N = Traits::kN;
  constexpr unsigned W = Traits::kW;
  using Board = typename Traits::Board;
  BitBoard<W> bounds = Board::bounds();
  BitBoard<W> unknown = ~board.known_on & ~board.known_off & bounds;
  const int center = static_cast<int>((N - 1) / 2);

  int best_score = 0x7fffffff;
  int best_dist = 0x7fffffff;
  unsigned best_x = 0;
  unsigned best_y = 0;

  cuda::std::pair<int, int> cell;
  while (unknown.some_on_if_any(cell)) {
    unknown.erase(cell);
    unsigned x = cell.first;
    unsigned y = cell.second;

    int dx = static_cast<int>(x) - center;
    int dy = static_cast<int>(y) - center;
    int dist_center_l1 = (dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy);
    int score = cell_branch_score<Traits>(board, x, y);

    if (score < best_score || (score == best_score && dist_center_l1 < best_dist)) {
      best_score = score;
      best_dist = dist_center_l1;
      best_x = x;
      best_y = y;
    }
  }

  best_x = __shfl_sync(0xffffffff, best_x, 0);
  best_y = __shfl_sync(0xffffffff, best_y, 0);
  return {best_x, best_y};
}

template <unsigned N, unsigned W>
inline void emit_frontier_rle(const Problem<W> &problem) {
  std::cout << to_rle<N, W>(problem.known_on) << "|"
            << to_rle<N, W>(problem.known_off) << "\n";
}

template <typename Traits>
__device__ void resolve_outcome_cell(const typename Traits::Board &board,
                                     typename Traits::Cell cell,
                                     typename Traits::Stack *stack) {
  {
    typename Traits::Board sub_board = board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();
    if (sub_board.consistent()) {
      stack_push<Traits::kW>(stack, sub_board.known_on, sub_board.known_off);
    } else {
      stats_record(StatId::InconsistentNodes);
    }
  }
  {
    typename Traits::Board sub_board = board;
    sub_board.known_off.set(cell);
    sub_board.propagate();
    if (sub_board.consistent()) {
      stack_push<Traits::kW>(stack, sub_board.known_on, sub_board.known_off);
    } else {
      stats_record(StatId::InconsistentNodes);
    }
  }
}

template <typename Traits>
__global__ void initialize_stack_kernel(typename Traits::Stack *stack) {
  Traits::seed_initial(stack);
}

template <typename Traits>
__global__ void initialize_stack_seed_kernel(typename Traits::Stack *stack,
                                             typename Traits::Problem seed) {
  auto board = Traits::Board::load_from(seed.known_on, seed.known_off);
  board.propagate();
  if (!board.consistent()) {
    stats_record(StatId::InconsistentNodes);
    return;
  }
  stack_push<Traits::kW>(stack, board.known_on, board.known_off);
}

template <typename Traits, bool FrontierMode>
__global__ void work_kernel(typename Traits::Stack *stack,
                            typename Traits::Output *output,
                            unsigned batch_start,
                            unsigned batch_size,
                            unsigned processed_base,
                            unsigned max_steps,
                            unsigned min_on,
                            bool use_min_on) {
  const unsigned problem_offset = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / 32);
  const unsigned problem_idx = batch_start + problem_offset;

  if (problem_offset >= batch_size) {
    return;
  }

  const typename Traits::Problem &problem = stack->problems[problem_idx];
  auto board = Traits::Board::load_from(problem.known_on, problem.known_off);

  if (!board.consistent()) {
    stats_record(StatId::InconsistentNodes);
    return;
  }

  stats_record(StatId::NodesVisited);

  ForcedCell forced{};
  LexStatus canonical = board.canonical_with_forced(forced);
  if (canonical == LexStatus::Greater) {
    stats_record(StatId::CanonicalSkips);
    return;
  }

  const unsigned on_pop = board.known_on.pop();
  if constexpr (FrontierMode) {
    const unsigned global_idx = processed_base + problem_offset;
    const bool reached_steps = max_steps > 0 && global_idx >= max_steps;
    const bool above_min = !use_min_on || (on_pop >= min_on);
    const bool emit = above_min && reached_steps;

    if (board.complete()) {
      output_buffer_push<Traits::kW>(output, board.known_on, board.known_off);
      return;
    }

    if (emit) {
      output_buffer_push<Traits::kW>(output, board.known_on, board.known_off);
      return;
    }
  } else {
    if (board.complete()) {
      stats_record(StatId::Solutions);
      output_buffer_push<Traits::kW>(output, board.known_on, board.known_off);
      return;
    }
  }

  if (forced.has_force && on_pop <= Traits::kSymForceMaxOn) {
    stats_record(StatId::SymmetryForced);
    resolve_outcome_cell<Traits>(board, forced.cell, stack);
    return;
  }

  auto vulnerable = board.vulnerable();
  if (!vulnerable.empty()) {
    auto cell = Traits::pick_preferred_branch_cell(vulnerable);
    stats_record(StatId::VulnerableBranches);
    resolve_outcome_cell<Traits>(board, cell, stack);
    return;
  }

  auto semivulnerable = board.semivulnerable();
  if (!semivulnerable.empty()) {
    auto cell = Traits::pick_preferred_branch_cell(semivulnerable);
    stats_record(StatId::SemiVulnerableBranches);
    resolve_outcome_cell<Traits>(board, cell, stack);
    return;
  }

  auto quasivulnerable = board.quasivulnerable();
  if (!quasivulnerable.empty()) {
    auto cell = Traits::pick_preferred_branch_cell(quasivulnerable);
    stats_record(StatId::QuasiVulnerableBranches);
    resolve_outcome_cell<Traits>(board, cell, stack);
    return;
  }

  Traits::branch_fallback(board, stack);
}

template <typename Traits, bool FrontierMode>
int solve_with_device_stack_impl(const board_array_t<Traits::kW> *seed_on,
                                 const board_array_t<Traits::kW> *seed_off,
                                 const FrontierConfig *config) {
  const FrontierConfig default_config{};
  const FrontierConfig *cfg = config ? config : &default_config;

  Traits::init_host();

  using Stack = typename Traits::Stack;
  using Output = typename Traits::Output;
  using Problem = typename Traits::Problem;

  Stack *d_stack = nullptr;
  Output *d_output = nullptr;
  Problem *d_output_entries = nullptr;

  unsigned output_capacity = FrontierMode
    ? (cfg->buffer_capacity ? cfg->buffer_capacity : (BATCH_MAX_SIZE * 4))
    : SOLUTION_BUFFER_CAPACITY;

  cudaMalloc((void**) &d_stack, sizeof(Stack));
  cudaMalloc((void**) &d_output, sizeof(Output));
  cudaMalloc((void**) &d_output_entries, output_capacity * sizeof(Problem));

  cudaMemset(d_stack, 0, sizeof(Stack));
  cudaMemset(d_output, 0, sizeof(Output));

  Output host_output{};
  host_output.entries = d_output_entries;
  host_output.capacity = output_capacity;
  cudaMemcpy(d_output, &host_output, sizeof(Output), cudaMemcpyHostToDevice);

  if constexpr (!FrontierMode) {
    if (seed_on && seed_off) {
      Problem seed{};
      seed.known_on = *seed_on;
      seed.known_off = *seed_off;
      initialize_stack_seed_kernel<Traits><<<1, 32>>>(d_stack, seed);
    } else {
      initialize_stack_kernel<Traits><<<1, 32>>>(d_stack);
    }
  } else {
    initialize_stack_kernel<Traits><<<1, 32>>>(d_stack);
  }

  reset_search_stats();

  Problem *d_compact_tmp = nullptr;
  size_t compact_capacity = 0;

  unsigned start_size;
  cudaMemcpy(&start_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);
  float feedback_scale = 1 / static_cast<float>(BATCH_MAX_SIZE / BATCH_WARMUP_SIZE);

  unsigned processed_total = 0;

  while (start_size > 0) {
    unsigned batch_size = static_cast<unsigned>(feedback_scale * static_cast<float>(BATCH_MAX_SIZE));
    batch_size = std::clamp(batch_size, 0, std::min(start_size, BATCH_MAX_SIZE));
    unsigned batch_start = start_size - batch_size;

    unsigned overflow_count = 0;
    bool had_retry = false;
    while (true) {
      cudaMemset(&d_stack->overflow, 0, sizeof(unsigned));
      cudaMemset(&d_output->size, 0, sizeof(unsigned));
      cudaMemset(&d_output->overflow, 0, sizeof(unsigned));

      unsigned blocks = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
      work_kernel<Traits, FrontierMode><<<blocks, WARPS_PER_BLOCK * 32>>>(
          d_stack,
          d_output,
          batch_start,
          batch_size,
          processed_total,
          cfg->max_steps,
          cfg->min_on,
          cfg->use_min_on);

      cudaMemcpy(&overflow_count, &d_stack->overflow, sizeof(unsigned), cudaMemcpyDeviceToHost);
      if (overflow_count == 0) {
        break;
      }

      had_retry = true;
      cudaMemcpy(&d_stack->size, &start_size, sizeof(unsigned), cudaMemcpyHostToDevice);

      std::cerr << "[error] stack overflow (" << overflow_count
                << " pushes) capacity=" << STACK_CAPACITY << std::endl;;

      if (batch_size <= 1) {
        break;
      }

      batch_size = batch_size > 1 ? (batch_size / 2) : 1;
      batch_start = start_size - batch_size;
    }

    if (overflow_count > 0 && batch_size <= 1) {
      break;
    }

    unsigned new_size;
    cudaMemcpy(&new_size, &d_stack->size, sizeof(unsigned), cudaMemcpyDeviceToHost);
    unsigned pushes = new_size - start_size;

    if (pushes > 0) {
      if (pushes > batch_size) {
        if (pushes > compact_capacity) {
          if (d_compact_tmp) {
            cudaFree(d_compact_tmp);
          }
          cudaMalloc((void **)&d_compact_tmp, pushes * sizeof(Problem));
          compact_capacity = pushes;
        }
        cudaMemcpy(d_compact_tmp, &d_stack->problems[start_size],
                   pushes * sizeof(Problem), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&d_stack->problems[batch_start], d_compact_tmp,
                   pushes * sizeof(Problem), cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpy(&d_stack->problems[batch_start], &d_stack->problems[start_size],
                   pushes * sizeof(Problem), cudaMemcpyDeviceToDevice);
      }
    }

    start_size = new_size - batch_size;
    cudaMemcpy(&d_stack->size, &start_size, sizeof(unsigned), cudaMemcpyHostToDevice);

    float push_ratio = static_cast<float>(pushes) / static_cast<float>(batch_size);
    float adjust = 1.0f + (BATCH_FEEDBACK_GAIN_RATIO * (BATCH_FEEDBACK_TARGET_RATIO - push_ratio));
    feedback_scale *= adjust;
    if (had_retry) {
      feedback_scale *= 0.5f;
    }
    feedback_scale = std::clamp(feedback_scale, 0.05f, 1.0f);

    unsigned output_count = 0;
    unsigned output_overflow = 0;
    cudaMemcpy(&output_count, &d_output->size, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&output_overflow, &d_output->overflow, sizeof(unsigned), cudaMemcpyDeviceToHost);

    if constexpr (FrontierMode) {
      processed_total += batch_size;
      if (output_overflow > 0) {
        std::cerr << "[error] frontier buffer overflow (" << output_overflow
                  << " drops) capacity=" << output_capacity << std::endl;
        break;
      }
    } else {
      if (output_overflow > 0) {
        std::cerr << "[error] solution buffer overflow (" << output_overflow
                  << " drops) capacity=" << output_capacity << std::endl;
      }
    }

    if (output_count > 0) {
      std::vector<Problem> outputs(output_count);
      cudaMemcpy(outputs.data(), d_output_entries, output_count * sizeof(Problem), cudaMemcpyDeviceToHost);
      if constexpr (FrontierMode) {
        for (const auto &entry : outputs) {
          Traits::emit_frontier(entry);
        }
        std::cout.flush();
      } else {
        for (const auto &entry : outputs) {
          Traits::emit_solution(entry);
        }
      }
    }

    maybe_print_stats(start_size, batch_size, feedback_scale, push_ratio);
  }

  cudaDeviceSynchronize();
  stats_final(start_size, feedback_scale);

  if (d_compact_tmp) {
    cudaFree(d_compact_tmp);
  }
  cudaFree(d_output_entries);
  cudaFree(d_output);
  cudaFree(d_stack);

  return 0;
}
