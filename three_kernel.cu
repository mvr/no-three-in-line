#include <algorithm>
#include <vector>
#include <iostream>

#include "board.cu"
#include "three_board.cu"

#include "three_kernel.hpp"
#include "parsing.hpp"

#include "params.hpp"
#include "three_search.cuh"

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
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    } else {
      stats_record(StatId::InconsistentNodes);
    }

    tried_board.known_off.set(cell);
    remaining &= ~(board_row_t<W>(1) << col);
  }
}

template <unsigned N, unsigned W>
struct AsymTraits {
  static constexpr unsigned kN = N;
  static constexpr unsigned kW = W;
  static constexpr unsigned kSymForceMaxOn = N - 2;
  static constexpr unsigned kCellBranchRowSpan = N;
  static constexpr unsigned kCellBranchColSpan = N;

  static constexpr unsigned kCellBranchRowScoreThreshold = 20;

  static constexpr int kCellBranchWColUnknown = 3;
  static constexpr int kCellBranchWRowUnknown = 4;
  static constexpr int kCellBranchWColOn = 3;
  static constexpr int kCellBranchWColOff = 8;
  static constexpr int kCellBranchWEndpointOn = 10;

  using Board = ThreeBoard<N, W>;
  using Problem = Problem<W>;
  using Stack = DeviceStack<W>;
  using Output = OutputBuffer<W>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() {
    Board::init_tables_host();
  }

  _DI_ static Cell pick_preferred_branch_cell(const BitBoard<W> &mask) {
    return mask.template first_center_on<N>();
  }

  _DI_ static void seed_initial(Stack *stack) {
    Board board;
    resolve_outcome_row<N, W, Axis::Horizontal>(board, DEFAULT_SEED_ROW, stack);
  }

  _DI_ static void branch_fallback(const Board &board, Stack *stack) {
    auto [row, row_unknown] = board.most_constrained_row();
    if (row_unknown >= kCellBranchRowScoreThreshold) {
      auto cell = pick_best_branch_cell<AsymTraits>(board);
      stats_record(StatId::CellBranches);
      resolve_outcome_cell<AsymTraits>(board, cell, stack);
    } else {
      stats_record(StatId::RowBranches);
      resolve_outcome_row<N, W, Axis::Horizontal>(board, row, stack);
    }
  }

  static void emit_solution(const Problem &problem) {
    std::cout << to_rle<N, W>(problem.known_on) << std::endl;
  }

  static void emit_frontier(const Problem &problem) {
    emit_frontier_rle<N, W>(problem);
  }
};

template <unsigned N, unsigned W>
int solve_with_device_stack(unsigned frontier_min_on) {
  return solve_with_device_stack_impl<AsymTraits<N, W>, true>(nullptr, nullptr, frontier_min_on);
}

template <unsigned N, unsigned W>
int solve_with_device_stack() {
  return solve_with_device_stack_impl<AsymTraits<N, W>, false>(nullptr, nullptr, 0);
}

template <unsigned N, unsigned W>
int solve_with_device_stack(const board_array_t<W> *seed_on,
                            const board_array_t<W> *seed_off) {
  return solve_with_device_stack_impl<AsymTraits<N, W>, false>(seed_on, seed_off, 0);
}

// Explicitly instantiate the template to the N in params.hpp, or it doesn't get compiled at all.
template int solve_with_device_stack<N, 32>();
template int solve_with_device_stack<N, 32>(const board_array_t<32> *, const board_array_t<32> *);
template int solve_with_device_stack<N, 32>(unsigned);
