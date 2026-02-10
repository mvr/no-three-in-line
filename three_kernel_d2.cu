#include <iostream>

#include "board.cu"
#include "three_board_d2.cu"

#include "three_kernel_d2.hpp"
#include "params.hpp"
#include "three_search.cuh"

template <unsigned N, unsigned W>
__device__ void resolve_outcome_row(const ThreeBoardD2<N, W> &board,
                                    unsigned row,
                                    DeviceStack<W> *stack) {
  constexpr unsigned full_n = 2 * N;
  constexpr board_row_t<W> row_mask = (full_n == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << full_n) - 1);

  board_row_t<W> line_known_on = board.known_on.row(row) & row_mask;
  board_row_t<W> line_known_off = board.known_off.row(row) & row_mask;
  board_row_t<W> remaining = ~line_known_on & ~line_known_off & row_mask;

  ThreeBoardD2<N, W> tried_board = board;
  if (line_known_on == 0) {
    unsigned keep = find_last_set<W>(remaining);
    remaining &= ~(board_row_t<W>(1) << keep);
  }

  while (remaining != 0) {
    const unsigned col = pick_center_col<full_n, W>(remaining);

    ThreeBoardD2<N, W> sub_board = tried_board;
    sub_board.known_on.set({col, row});
    sub_board.eliminate_all_lines({col, row});
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    } else {
      stats_record(StatId::InconsistentNodes);
    }

    tried_board.known_off.set({col, row});
    remaining &= ~(board_row_t<W>(1) << col);
  }
}

template <unsigned N, unsigned W>
__device__ void resolve_outcome_col(const ThreeBoardD2<N, W> &board,
                                    unsigned col,
                                    DeviceStack<W> *stack) {
  constexpr board_row_t<W> row_mask =
      (N == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << N) - 1);

  board_row_t<W> col_known_on = board.known_on.column(col) & row_mask;
  board_row_t<W> col_known_off = board.known_off.column(col) & row_mask;
  board_row_t<W> remaining = ~col_known_on & ~col_known_off & row_mask;

  ThreeBoardD2<N, W> tried_board = board;

  while (remaining != 0) {
    const unsigned row = pick_center_col<N, W>(remaining);

    ThreeBoardD2<N, W> sub_board = tried_board;
    sub_board.known_on.set({col, row});
    sub_board.eliminate_all_lines({col, row});
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    } else {
      stats_record(StatId::InconsistentNodes);
    }

    tried_board.known_off.set({col, row});
    remaining &= ~(board_row_t<W>(1) << row);
  }
}

template <unsigned N, unsigned W>
struct D2Traits {
  static constexpr unsigned kN = ThreeBoardD2<N, W>::FULL_N;
  static constexpr unsigned kW = W;
  static constexpr unsigned kSymForceMaxOn = N / 2;
  static constexpr unsigned kCellBranchRowSpan = ThreeBoardD2<N, W>::FULL_N;
  static constexpr unsigned kCellBranchColSpan = N;
  static constexpr unsigned kRowOnZeroUnknownNum = 7;
  static constexpr unsigned kRowOnZeroUnknownDen = 4;
  static constexpr unsigned kRowBranchUnknownThreshold = 3;
  static constexpr unsigned kColumnBranchUnknownThreshold = 5;
  static constexpr unsigned kColumnVsRowUnknownNum = 3;
  static constexpr unsigned kColumnVsRowUnknownDen = 2;
  static constexpr int kCellBranchWColUnknown = 8;
  static constexpr int kCellBranchWRowUnknown = 0;
  static constexpr int kCellBranchWColOn = 4;
  static constexpr int kCellBranchWColOff = 8;
  static constexpr int kCellBranchWEndpointOn = 0;
  static constexpr bool kEnableSemiQuasi = true;

  using Board = ThreeBoardD2<N, W>;
  using Problem = Problem<W>;
  using Stack = DeviceStack<W>;
  using Output = OutputBuffer<W>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() {
    Board::init_tables_host();
  }

  _DI_ static Cell pick_preferred_branch_cell(const BitBoard<W> &mask) {
    const unsigned lane = threadIdx.x & 31;
    constexpr unsigned full_n = Board::FULL_N;
    unsigned best_score = 0xffffffffu;
    unsigned best_x = 0;
    unsigned best_y = 0;

    constexpr int center2 = static_cast<int>(Board::FULL_N) - 1;  // 2*(N-0.5)
    auto consider = [&](unsigned x, unsigned y) {
      int dx2 = 2 * static_cast<int>(x) - center2;
      if (dx2 < 0) {
        dx2 = -dx2;
      }
      int dy2 = 2 * static_cast<int>(y) - center2;
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
      if (lane < N && mask.state != 0) {
        const unsigned x = pick_center_col<full_n, W>(mask.state);
        consider(x, lane);
      }
    } else {
      const unsigned row_even = 2 * lane;
      if (row_even < N) {
        const uint64_t row_even_bits =
            static_cast<uint64_t>(mask.state.x) |
            (static_cast<uint64_t>(mask.state.y) << 32);
        if (row_even_bits != 0) {
          const unsigned x = pick_center_col<full_n, W>(row_even_bits);
          consider(x, row_even);
        }
      }
      const unsigned row_odd = 2 * lane + 1;
      if (row_odd < N) {
        const uint64_t row_odd_bits =
            static_cast<uint64_t>(mask.state.z) |
            (static_cast<uint64_t>(mask.state.w) << 32);
        if (row_odd_bits != 0) {
          const unsigned x = pick_center_col<full_n, W>(row_odd_bits);
          consider(x, row_odd);
        }
      }
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

  struct BranchRowChoice {
    unsigned row;
    unsigned unknown;
  };

  _DI_ static BranchRowChoice pick_branch_row(const Board &board) {
    const unsigned lane = threadIdx.x & 31;

    unsigned best_row0 = 0;
    unsigned best_unknown0 = 0xffffffffu;
    unsigned best_row1 = 0;
    unsigned best_unknown1 = 0xffffffffu;
    constexpr unsigned full_n = Board::FULL_N;

    auto update_best = [&](unsigned row, unsigned on, unsigned unknown) {
      if (unknown == 0 || on > 1) {
        return;
      }
      if (on == 0) {
        if (unknown < best_unknown0 || (unknown == best_unknown0 && row < best_row0)) {
          best_unknown0 = unknown;
          best_row0 = row;
        }
      } else {
        if (unknown < best_unknown1 || (unknown == best_unknown1 && row < best_row1)) {
          best_unknown1 = unknown;
          best_row1 = row;
        }
      }
    };

    if constexpr (W == 32) {
      if (lane < N) {
        const unsigned on = popcount<32>(board.known_on.state);
        const unsigned off = popcount<32>(board.known_off.state);
        const unsigned unknown = full_n - on - off;
        update_best(lane, on, unknown);
      }
    } else {
      if (lane < ((N + 1) >> 1)) {
        const unsigned row_even = 2 * lane;
        if (row_even < N) {
          const unsigned on_even = popcount<32>(board.known_on.state.x) + popcount<32>(board.known_on.state.y);
          const unsigned off_even = popcount<32>(board.known_off.state.x) + popcount<32>(board.known_off.state.y);
          const unsigned unk_even = full_n - on_even - off_even;
          update_best(row_even, on_even, unk_even);
        }
      }
      if (lane < (N >> 1)) {
        const unsigned row_odd = 2 * lane + 1;
        const unsigned on_odd = popcount<32>(board.known_on.state.z) + popcount<32>(board.known_on.state.w);
        const unsigned off_odd = popcount<32>(board.known_off.state.z) + popcount<32>(board.known_off.state.w);
        const unsigned unk_odd = full_n - on_odd - off_odd;
        update_best(row_odd, on_odd, unk_odd);
      }
    }

    auto reduce_best = [&](unsigned &best_row, unsigned &best_unknown) {
      for (int offset = 16; offset > 0; offset /= 2) {
        unsigned other_row = __shfl_down_sync(0xffffffff, best_row, offset);
        unsigned other_unknown = __shfl_down_sync(0xffffffff, best_unknown, offset);
        if (other_unknown < best_unknown ||
            (other_unknown == best_unknown && other_row < best_row)) {
          best_row = other_row;
          best_unknown = other_unknown;
        }
      }
      best_row = __shfl_sync(0xffffffff, best_row, 0);
      best_unknown = __shfl_sync(0xffffffff, best_unknown, 0);
    };

    reduce_best(best_row0, best_unknown0);
    reduce_best(best_row1, best_unknown1);

    if (best_unknown0 != 0xffffffffu) {
      if (best_unknown1 == 0xffffffffu) {
        return {best_row0, best_unknown0};
      }
      if ((best_unknown0 * kRowOnZeroUnknownDen) <= (best_unknown1 * kRowOnZeroUnknownNum)) {
        return {best_row0, best_unknown0};
      }
      return {best_row1, best_unknown1};
    }
    if (best_unknown1 != 0xffffffffu) {
      return {best_row1, best_unknown1};
    }
    return {0u, 0u};
  }

  struct BranchColChoice {
    unsigned col;
    unsigned unknown;
  };

  _DI_ static BranchColChoice pick_branch_col(const Board &board) {
    constexpr unsigned full_n = Board::FULL_N;
    constexpr board_row_t<W> row_mask =
        (N == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << N) - 1);

    unsigned best_col = 0;
    unsigned best_unknown = 0xffffffffu;

    for (unsigned col = 0; col < full_n; ++col) {
      const unsigned col_on = board.known_on.column_pop(col);
      const unsigned col_off = board.known_off.column_pop(col);
      const unsigned col_unknown = N - col_on - col_off;
      if (col_on == 0 && col_unknown > 0) {
        if (col_unknown < best_unknown ||
            (col_unknown == best_unknown && col < best_col)) {
          best_col = col;
          best_unknown = col_unknown;
        }
      }
    }

    return {best_col, best_unknown};
  }

  _DI_ static void seed_initial(Stack *stack) {
    Board board;
    stack_push<W>(stack, board.known_on, board.known_off);
  }

  _DI_ static void branch_fallback(const Board &board, Stack *stack) {
    const BranchColChoice col_choice = pick_branch_col(board);
    const BranchRowChoice row_choice = pick_branch_row(board);

    const bool row_meets = (row_choice.unknown != 0xffffffffu) &&
                           (row_choice.unknown <= kRowBranchUnknownThreshold);
    const bool col_meets = (col_choice.unknown != 0xffffffffu) &&
                           (col_choice.unknown <= kColumnBranchUnknownThreshold);

    if (row_meets && col_meets) {
      const bool choose_col = (col_choice.unknown * kColumnVsRowUnknownDen) <=
                              (row_choice.unknown * kColumnVsRowUnknownNum);
      if (choose_col) {
        stats_record(StatId::RowBranches);
        resolve_outcome_col<N, W>(board, col_choice.col, stack);
      } else {
        stats_record(StatId::RowBranches);
        resolve_outcome_row<N, W>(board, row_choice.row, stack);
      }
      return;
    }

    if (col_meets) {
      stats_record(StatId::RowBranches);
      resolve_outcome_col<N, W>(board, col_choice.col, stack);
      return;
    }

    if (row_meets) {
      stats_record(StatId::RowBranches);
      resolve_outcome_row<N, W>(board, row_choice.row, stack);
      return;
    }

    auto cell = pick_best_branch_cell<D2Traits>(board);
    stats_record(StatId::CellBranches);
    resolve_outcome_cell<D2Traits>(board, cell, stack);
  }

  static void expand_half_to_full(const board_array_t<W> &half, board_array_t<Board::FULL_W> &expanded) {
    for (unsigned y = 0; y < N; ++y) {
      board_row_t<W> row = half[y];
      const unsigned my = Board::FULL_N - 1 - y;
      while (row != 0) {
        const unsigned x = find_first_set<W>(row);
        if constexpr (Board::FULL_W == 32) {
          expanded[y] |= (uint32_t(1) << x);
          expanded[my] |= (uint32_t(1) << x);
        } else {
          expanded[y] |= (uint64_t(1) << x);
          expanded[my] |= (uint64_t(1) << x);
        }
        row &= (row - 1);
      }
    }
  }

  static void emit_solution(const Problem &problem) {
    board_array_t<Board::FULL_W> expanded{};
    expand_half_to_full(problem.known_on, expanded);
    std::cout << to_rle<Board::FULL_N, Board::FULL_W>(expanded) << std::endl;
  }

  static void emit_frontier(const Problem &problem) {
    board_array_t<Board::FULL_W> expanded_on{};
    board_array_t<Board::FULL_W> expanded_off{};
    expand_half_to_full(problem.known_on, expanded_on);
    expand_half_to_full(problem.known_off, expanded_off);
    std::cout << to_rle<Board::FULL_N, Board::FULL_W>(expanded_on)
              << "|" << to_rle<Board::FULL_N, Board::FULL_W>(expanded_off) << "\n";
  }
};

template <unsigned N>
int solve_with_device_stack_d2() {
  if constexpr (N <= 16) {
    return solve_with_device_stack_impl<D2Traits<N, 32>, false>(nullptr, nullptr, 0);
  } else {
    return solve_with_device_stack_impl<D2Traits<N, 64>, false>(nullptr, nullptr, 0);
  }
}

template <unsigned N>
int solve_with_device_stack_d2(unsigned frontier_min_on) {
  if constexpr (N <= 16) {
    return solve_with_device_stack_impl<D2Traits<N, 32>, true>(nullptr, nullptr, frontier_min_on);
  } else {
    return solve_with_device_stack_impl<D2Traits<N, 64>, true>(nullptr, nullptr, frontier_min_on);
  }
}

template <unsigned N>
int solve_with_device_stack_d2(const board_array_t<32> *seed_on,
                               const board_array_t<32> *seed_off) {
  if constexpr (N <= 16) {
    return solve_with_device_stack_impl<D2Traits<N, 32>, false>(seed_on, seed_off, 0);
  } else {
    (void)seed_on;
    (void)seed_off;
    std::cerr << "[d2] 32-bit seeds are invalid for N > 16\n";
    return 1;
  }
}

template <unsigned N>
int solve_with_device_stack_d2(const board_array_t<64> *seed_on,
                               const board_array_t<64> *seed_off) {
  if constexpr (N > 16) {
    return solve_with_device_stack_impl<D2Traits<N, 64>, false>(seed_on, seed_off, 0);
  } else {
    (void)seed_on;
    (void)seed_off;
    std::cerr << "[d2] 64-bit seeds are invalid for N <= 16\n";
    return 1;
  }
}

template int solve_with_device_stack_d2<N>();
template int solve_with_device_stack_d2<N>(unsigned);
template int solve_with_device_stack_d2<N>(const board_array_t<32> *, const board_array_t<32> *);
template int solve_with_device_stack_d2<N>(const board_array_t<64> *, const board_array_t<64> *);
