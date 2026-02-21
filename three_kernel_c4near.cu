#include <iostream>

#include "board.cu"
#include "three_board_c4near.cu"

#include "params.hpp"
#include "three_kernel_c4near.hpp"
#include "three_search.cuh"

template <unsigned N, unsigned W>
__device__ void resolve_outcome_row(const ThreeBoardC4Near<N, W> &board,
                                    unsigned full_row,
                                    DeviceStack<W> *stack) {
  if (full_row == 0 || full_row >= N) {
    return;
  }

  ThreeBoardC4Near<N, W> tried_board = board;
  constexpr board_row_t<W> row_mask =
      (N == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << N) - 1u);
  constexpr board_row_t<W> col_mask =
      ((N - 1) == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << (N - 1)) - 1u);

  const unsigned row_idx = full_row - 1;
  const unsigned col_idx = full_row;

  board_row_t<W> row_known_on = board.known_on.row(row_idx) & row_mask;
  board_row_t<W> row_known_off = board.known_off.row(row_idx) & row_mask;
  board_row_t<W> col_known_on = board.known_on.column(col_idx) & col_mask;
  board_row_t<W> col_known_off = board.known_off.column(col_idx) & col_mask;

  board_row_t<W> row_remaining = ~row_known_on & ~row_known_off & row_mask;
  board_row_t<W> col_remaining = ~col_known_on & ~col_known_off & col_mask;

  const board_row_t<W> pivot_row_bit = board_row_t<W>(1) << col_idx;
  const board_row_t<W> pivot_col_bit = board_row_t<W>(1) << row_idx;
  const bool pivot_unknown =
      ((row_remaining & pivot_row_bit) != 0u) && ((col_remaining & pivot_col_bit) != 0u);
  row_remaining &= ~pivot_row_bit;
  col_remaining &= ~pivot_col_bit;

  if ((row_known_on | col_known_on) == 0u) {
    if (row_remaining != 0u) {
      const unsigned keep = find_last_set<W>(row_remaining);
      row_remaining &= ~(board_row_t<W>(1) << keep);
    } else if (col_remaining != 0u) {
      const unsigned keep = find_last_set<W>(col_remaining);
      col_remaining &= ~(board_row_t<W>(1) << keep);
    }
  }

  while (col_remaining != 0u) {
    const unsigned bit = find_first_set<W>(col_remaining);
    const cuda::std::pair<unsigned, unsigned> cell = {col_idx, bit};

    ThreeBoardC4Near<N, W> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    }

    tried_board.known_off.set(cell);
    col_remaining &= ~(board_row_t<W>(1) << bit);
  }

  while (row_remaining != 0u) {
    const unsigned bit = find_first_set<W>(row_remaining);
    const cuda::std::pair<unsigned, unsigned> cell = {bit, row_idx};

    ThreeBoardC4Near<N, W> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    }

    tried_board.known_off.set(cell);
    row_remaining &= ~(board_row_t<W>(1) << bit);
  }

  if (pivot_unknown) {
    const cuda::std::pair<unsigned, unsigned> cell = {col_idx, row_idx};
    ThreeBoardC4Near<N, W> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    }
  }
}

template <unsigned N, unsigned W>
struct C4NearTraits {
  static_assert(N <= 64, "C4Near solver currently supports N <= 64");
  static_assert((W == 32 && N <= 32) || (W == 64 && N <= 64),
                "Invalid C4Near width/size combination");

  static constexpr unsigned kN = N;
  static constexpr unsigned kW = W;
  static constexpr unsigned kSymForceMaxOn = N / 2;
  static constexpr unsigned kRowOnZeroUnknownNum = 6;
  static constexpr unsigned kRowOnZeroUnknownDen = 4;

  using Board = ThreeBoardC4Near<N, W>;
  using Problem = ::Problem<W>;
  using Stack = DeviceStack<W>;
  using Output = OutputBuffer<W>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() { Board::init_tables_host(); }

  _DI_ static Cell pick_preferred_branch_cell(const BitBoard<W> &mask) {
    const unsigned lane = threadIdx.x & 31;
    unsigned best_score = 0xffffffffu;
    unsigned best_x = 0;
    unsigned best_y = 0;

    constexpr int center2x = static_cast<int>(N) - 1;
    constexpr int center2y = static_cast<int>(N) - 2;

    auto consider = [&](unsigned y, board_row_t<W> bits) {
      if (y >= (N - 1) || bits == 0) {
        return;
      }
      const unsigned x = pick_center_col<N, W>(bits);
      int dx2 = 2 * static_cast<int>(x) - center2x;
      if (dx2 < 0) {
        dx2 = -dx2;
      }
      int dy2 = 2 * static_cast<int>(y) - center2y;
      if (dy2 < 0) {
        dy2 = -dy2;
      }
      const unsigned score = static_cast<unsigned>(dx2 * dx2 + dy2 * dy2);
      if (score < best_score) {
        best_score = score;
        best_x = x;
        best_y = y;
      }
    };

    if constexpr (W == 32) {
      consider(lane, mask.state);
    } else {
      const unsigned row_even = 2 * lane;
      const unsigned row_odd = row_even + 1;
      const board_row_t<64> bits_even =
          (static_cast<board_row_t<64>>(mask.state.y) << 32) | mask.state.x;
      const board_row_t<64> bits_odd =
          (static_cast<board_row_t<64>>(mask.state.w) << 32) | mask.state.z;
      consider(row_even, bits_even);
      consider(row_odd, bits_odd);
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
    constexpr board_row_t<W> row_mask =
        (N == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << N) - 1u);
    constexpr board_row_t<W> col_mask =
        ((N - 1) == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << (N - 1)) - 1u);

    unsigned row0 = 0;
    unsigned row1 = 0;
    unsigned best0_unknown = 0xffffffffu;
    unsigned best1_unknown = 0xffffffffu;

    for (unsigned full_row = 1; full_row < N; ++full_row) {
      const unsigned row_idx = full_row - 1;
      const board_row_t<W> row_on = board.known_on.row(row_idx) & row_mask;
      const board_row_t<W> row_off = board.known_off.row(row_idx) & row_mask;
      const board_row_t<W> col_on = board.known_on.column(full_row) & col_mask;
      const board_row_t<W> col_off = board.known_off.column(full_row) & col_mask;

      const board_row_t<W> row_unknown = ~(row_on | row_off) & row_mask;
      const board_row_t<W> col_unknown = ~(col_on | col_off) & col_mask;

      const unsigned pivot_unknown = (row_unknown >> full_row) & 1u;
      const bool row_empty = (row_on | col_on) == 0u;
      const unsigned unknown_count =
          popcount<W>(row_unknown) + popcount<W>(col_unknown) - pivot_unknown;

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
    stack_push<W>(stack, board.known_on, board.known_off);
  }

  _DI_ static void branch_fallback(const Board &board, Stack *stack) {
    const unsigned row = pick_row_on_priority(board);
    if (row != 0u) {
      stats_record(StatId::RowBranches);
      resolve_outcome_row<N, W>(board, row, stack);
      return;
    }

    const BitBoard<W> unknown = (~board.known_on & ~board.known_off) & Board::bounds();
    const Cell cell = pick_preferred_branch_cell(unknown);
    stats_record(StatId::CellBranches);
    resolve_outcome_cell<C4NearTraits<N, W>>(board, cell, stack);
  }

  static void emit_solution(const Problem &problem) {
    std::array<std::array<uint8_t, Board::FULL_N>, Board::FULL_N> expanded{};
    constexpr board_row_t<W> row_mask =
        (N == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << N) - 1u);
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
      expanded[iy][ix] = 1u;
    };

    for (unsigned ly = 0; ly < (N - 1); ++ly) {
      board_row_t<W> row = problem.known_on[ly] & row_mask;
      while (row != 0) {
        const unsigned lx = count_trailing_zeros<W>(row);
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

    std::cout << to_rle_dense<Board::FULL_N>(expanded) << std::endl;
  }

  static void emit_frontier(const Problem &problem) { emit_frontier_rle<N, W>(problem); }
};

template <unsigned N, unsigned W>
int solve_with_device_stack_c4near(const SearchOptions<W> &options) {
  if constexpr (W == 32 && N > 32) {
    (void)options;
    std::cerr << "[c4near] W=32 is invalid for N > 32\n";
    return 1;
  } else {
    if (options.mode == SearchMode::Frontier) {
      return solve_with_device_stack_impl<C4NearTraits<N, W>, true>(options);
    }
    return solve_with_device_stack_impl<C4NearTraits<N, W>, false>(options);
  }
}

template int solve_with_device_stack_c4near<N, 32>(const SearchOptions<32> &);
template int solve_with_device_stack_c4near<N, 64>(const SearchOptions<64> &);
