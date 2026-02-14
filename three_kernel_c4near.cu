#include <iostream>

#include "board.cu"
#include "three_board_c4near.cu"

#include "params.hpp"
#include "three_kernel_c4near.hpp"
#include "three_search.cuh"

template <unsigned N>
__device__ void resolve_outcome_row(const ThreeBoardC4Near<N, 32> &board,
                                    unsigned full_row,
                                    DeviceStack<32> *stack) {
  if (full_row == 0 || full_row >= N) {
    return;
  }

  ThreeBoardC4Near<N, 32> tried_board = board;
  constexpr board_row_t<32> row_mask =
      (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
  constexpr board_row_t<32> col_mask =
      (N <= 1) ? 0u : ((board_row_t<32>(1) << (N - 1)) - 1u);

  const unsigned row_idx = full_row - 1;
  const unsigned col_idx = full_row;

  board_row_t<32> row_known_on = board.known_on.row(row_idx) & row_mask;
  board_row_t<32> row_known_off = board.known_off.row(row_idx) & row_mask;
  board_row_t<32> col_known_on = board.known_on.column(col_idx) & col_mask;
  board_row_t<32> col_known_off = board.known_off.column(col_idx) & col_mask;

  board_row_t<32> row_remaining = ~row_known_on & ~row_known_off & row_mask;
  board_row_t<32> col_remaining = ~col_known_on & ~col_known_off & col_mask;

  const board_row_t<32> pivot_row_bit = board_row_t<32>(1) << col_idx;
  const board_row_t<32> pivot_col_bit = board_row_t<32>(1) << row_idx;
  const bool pivot_unknown =
      ((row_remaining & pivot_row_bit) != 0u) && ((col_remaining & pivot_col_bit) != 0u);
  row_remaining &= ~pivot_row_bit;
  col_remaining &= ~pivot_col_bit;

  if ((row_known_on | col_known_on) == 0u) {
    if (row_remaining != 0u) {
      unsigned keep = find_last_set<32>(row_remaining);
      row_remaining &= ~(board_row_t<32>(1) << keep);
    } else if (col_remaining != 0u) {
      unsigned keep = find_last_set<32>(col_remaining);
      col_remaining &= ~(board_row_t<32>(1) << keep);
    }
  }

  while (col_remaining != 0u) {
    unsigned bit = find_first_set<32>(col_remaining);
    cuda::std::pair<unsigned, unsigned> cell = {col_idx, bit};

    ThreeBoardC4Near<N, 32> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<32>(stack, sub_board.known_on, sub_board.known_off);
    }

    tried_board.known_off.set(cell);
    col_remaining &= ~(board_row_t<32>(1) << bit);
  }

  while (row_remaining != 0u) {
    unsigned bit = find_first_set<32>(row_remaining);
    cuda::std::pair<unsigned, unsigned> cell = {bit, row_idx};

    ThreeBoardC4Near<N, 32> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<32>(stack, sub_board.known_on, sub_board.known_off);
    }

    tried_board.known_off.set(cell);
    row_remaining &= ~(board_row_t<32>(1) << bit);
  }

  if (pivot_unknown) {
    cuda::std::pair<unsigned, unsigned> cell = {col_idx, row_idx};
    ThreeBoardC4Near<N, 32> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<32>(stack, sub_board.known_on, sub_board.known_off);
    }
  }
}

template <unsigned N>
struct C4NearTraits {
  static constexpr unsigned kN = N;
  static constexpr unsigned kW = 32;
  static constexpr unsigned kSymForceMaxOn = N / 2;
  static constexpr unsigned kRowOnZeroUnknownNum = 6;
  static constexpr unsigned kRowOnZeroUnknownDen = 4;

  using Board = ThreeBoardC4Near<N, 32>;
  using Problem = Problem<32>;
  using Stack = DeviceStack<32>;
  using Output = OutputBuffer<32>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() { Board::init_tables_host(); }

  _DI_ static Cell pick_preferred_branch_cell(const BitBoard<32> &mask) {
    const unsigned lane = threadIdx.x & 31;
    unsigned best_score = 0xffffffffu;
    unsigned best_x = 0;
    unsigned best_y = 0;

    constexpr int center2x = static_cast<int>(N) - 1;
    constexpr int center2y = static_cast<int>(N) - 2;
    if (lane < (N - 1) && mask.state != 0) {
      const unsigned x = pick_center_col<N, 32>(mask.state);
      int dx2 = 2 * static_cast<int>(x) - center2x;
      if (dx2 < 0) {
        dx2 = -dx2;
      }
      int dy2 = 2 * static_cast<int>(lane) - center2y;
      if (dy2 < 0) {
        dy2 = -dy2;
      }
      best_score = static_cast<unsigned>(dx2 * dx2 + dy2 * dy2);
      best_x = x;
      best_y = lane;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      const unsigned other_score = __shfl_down_sync(0xffffffff, best_score, offset);
      const unsigned other_x = __shfl_down_sync(0xffffffff, best_x, offset);
      const unsigned other_y = __shfl_down_sync(0xffffffff, best_y, offset);
      if (other_score < best_score) {
        best_score = other_score;
        best_x = other_x;
        best_y = other_y;
      }
    }

    best_score = __shfl_sync(0xffffffff, best_score, 0);
    best_x = __shfl_sync(0xffffffff, best_x, 0);
    best_y = __shfl_sync(0xffffffff, best_y, 0);
    if (best_score == 0xffffffffu) {
      return {0u, 0u};
    }
    return {best_x, best_y};
  }

  _DI_ static unsigned pick_row_on_priority(const Board &board) {
    constexpr board_row_t<32> row_mask =
        (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
    constexpr board_row_t<32> col_mask =
        (N <= 1) ? 0u : ((board_row_t<32>(1) << (N - 1)) - 1u);

    unsigned row0 = 0;
    unsigned row1 = 0;
    unsigned best0_unknown = 0xffffffffu;
    unsigned best1_unknown = 0xffffffffu;

    for (unsigned full_row = 1; full_row < N; ++full_row) {
      const unsigned row_idx = full_row - 1;
      board_row_t<32> row_on = board.known_on.row(row_idx) & row_mask;
      board_row_t<32> row_off = board.known_off.row(row_idx) & row_mask;
      board_row_t<32> col_on = board.known_on.column(full_row) & col_mask;
      board_row_t<32> col_off = board.known_off.column(full_row) & col_mask;

      board_row_t<32> row_unknown = ~(row_on | row_off) & row_mask;
      board_row_t<32> col_unknown = ~(col_on | col_off) & col_mask;

      const unsigned pivot_unknown = (row_unknown >> full_row) & 1u;
      const bool row_empty = (row_on | col_on) == 0u;
      const unsigned unknown_count =
          popcount<32>(row_unknown) + popcount<32>(col_unknown) - pivot_unknown;

      if ((threadIdx.x & 31) == 0 && unknown_count != 0u) {
        if (row_empty) {
          if (unknown_count < best0_unknown ||
              (unknown_count == best0_unknown && full_row < row0)) {
            row0 = full_row;
            best0_unknown = unknown_count;
          }
        } else {
          if (unknown_count < best1_unknown ||
              (unknown_count == best1_unknown && full_row < row1)) {
            row1 = full_row;
            best1_unknown = unknown_count;
          }
        }
      }
    }

    row0 = __shfl_sync(0xffffffff, row0, 0);
    row1 = __shfl_sync(0xffffffff, row1, 0);
    best0_unknown = __shfl_sync(0xffffffff, best0_unknown, 0);
    best1_unknown = __shfl_sync(0xffffffff, best1_unknown, 0);

    if (best1_unknown == 0xffffffffu) {
      return row0;
    }
    if (best0_unknown == 0xffffffffu) {
      return row1;
    }
    if ((best0_unknown * kRowOnZeroUnknownDen) <=
        (best1_unknown * kRowOnZeroUnknownNum)) {
      return row0;
    }
    return row1;
  }

  _DI_ static void seed_initial(Stack *stack) {
    Board board;
    stack_push<32>(stack, board.known_on, board.known_off);
  }

  _DI_ static void branch_fallback(const Board &board, Stack *stack) {
    const unsigned row = pick_row_on_priority(board);
    if (row != 0u) {
      stats_record(StatId::RowBranches);
      resolve_outcome_row<N>(board, row, stack);
      return;
    }

    const BitBoard<32> unknown = (~board.known_on & ~board.known_off) & Board::bounds();
    const Cell cell = pick_preferred_branch_cell(unknown);
    stats_record(StatId::CellBranches);
    resolve_outcome_cell<C4NearTraits<N>>(board, cell, stack);
  }

  static void emit_solution(const Problem &problem) {
    board_array_t<Board::FULL_W> expanded{};
    constexpr board_row_t<32> row_mask =
        (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
    auto set_full = [&](int fx, int fy) {
      const int ox = static_cast<int>(N) - 1;
      const int oy = static_cast<int>(N) - 1;
      const int ix_i = fx + ox;
      const int iy_i = fy + oy;
      if (ix_i < 0 || ix_i >= static_cast<int>(Board::FULL_N) ||
          iy_i < 0 || iy_i >= static_cast<int>(Board::FULL_N)) {
        return;
      }
      const unsigned ix = static_cast<unsigned>(ix_i);
      const unsigned iy = static_cast<unsigned>(iy_i);
      if constexpr (Board::FULL_W == 32) {
        expanded[iy] |= (uint32_t(1) << ix);
      } else {
        expanded[iy] |= (uint64_t(1) << ix);
      }
    };

    for (unsigned ly = 0; ly < (N - 1); ++ly) {
      board_row_t<32> row = problem.known_on[ly] & row_mask;
      while (row != 0) {
        const unsigned lx = count_trailing_zeros<32>(row);
        const int x = static_cast<int>(lx);
        const int y = static_cast<int>(ly) + 1;

        if (x == y) {
          set_full(x, y);
          set_full(-x, -y);
        } else {
          cuda::std::pair<int, int> p{x, y};
          auto rotate90_host = [](cuda::std::pair<int, int> c) {
            return cuda::std::pair<int, int>{-c.second, c.first};
          };
          for (int r = 0; r < 4; ++r) {
            set_full(p.first, p.second);
            p = rotate90_host(p);
          }
        }

        row &= (row - 1);
      }
    }

    std::cout << to_rle<Board::FULL_N, Board::FULL_W>(expanded) << std::endl;
  }

  static void emit_frontier(const Problem &problem) {
    std::cout << to_rle<N, 32>(problem.known_on) << "|"
              << to_rle<N, 32>(problem.known_off) << "\n";
  }
};

template <unsigned N>
int solve_with_device_stack_c4near() {
  return solve_with_device_stack_impl<C4NearTraits<N>, false>(nullptr, nullptr, 0);
}

template <unsigned N>
int solve_with_device_stack_c4near(const board_array_t<32> *seed_on,
                                   const board_array_t<32> *seed_off) {
  return solve_with_device_stack_impl<C4NearTraits<N>, false>(seed_on, seed_off, 0);
}

template <unsigned N>
int solve_with_device_stack_c4near(unsigned frontier_min_on) {
  return solve_with_device_stack_impl<C4NearTraits<N>, true>(nullptr, nullptr, frontier_min_on);
}

template int solve_with_device_stack_c4near<N>();
template int solve_with_device_stack_c4near<N>(const board_array_t<32> *, const board_array_t<32> *);
template int solve_with_device_stack_c4near<N>(unsigned);
