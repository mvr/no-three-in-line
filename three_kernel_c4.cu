#include <iostream>
#include <cstdlib>

#include "board.cu"
#include "three_board_c4.cu"

#include "three_kernel_c4.hpp"
#include "parsing.hpp"
#include "params.hpp"
#include "three_search.cuh"

template <unsigned N, unsigned W>
__device__ void resolve_outcome_row(const ThreeBoardC4<N, W> &board,
                                    unsigned ix,
                                    DeviceStack<W> *stack) {
  ThreeBoardC4<N, W> tried_board = board;

  constexpr board_row_t<W> row_mask = (N == W) ? ~board_row_t<W>(0) : ((board_row_t<W>(1) << N) - 1);

  board_row_t<W> row_known_on = board.known_on.row(ix) & row_mask;
  board_row_t<W> row_known_off = board.known_off.row(ix) & row_mask;
  board_row_t<W> col_known_on = board.known_on.column(ix) & row_mask;
  board_row_t<W> col_known_off = board.known_off.column(ix) & row_mask;

  board_row_t<W> row_remaining = ~row_known_on & ~row_known_off & row_mask;
  board_row_t<W> col_remaining = ~col_known_on & ~col_known_off & row_mask;

  const board_row_t<W> pivot_bit = board_row_t<W>(1) << ix;
  const bool pivot_unknown = (row_remaining & pivot_bit) != 0;
  row_remaining &= ~pivot_bit;
  col_remaining &= ~pivot_bit;

  if ((row_known_on | col_known_on) == 0) {
    if (row_remaining != 0) {
      unsigned keep = find_last_set<W>(row_remaining);
      row_remaining &= ~(board_row_t<W>(1) << keep);
    } else if (col_remaining != 0) {
      unsigned keep = find_last_set<W>(col_remaining);
      col_remaining &= ~(board_row_t<W>(1) << keep);
    }
  }

  while (col_remaining != 0) {
    unsigned bit = find_first_set<W>(col_remaining);
    cuda::std::pair<unsigned, unsigned> cell = {ix, bit};

    ThreeBoardC4<N, W> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    }

    tried_board.known_off.set(cell);
    col_remaining &= ~(board_row_t<W>(1) << bit);
  }

  while (row_remaining != 0) {
    unsigned bit = find_first_set<W>(row_remaining);
    cuda::std::pair<unsigned, unsigned> cell = {bit, ix};

    ThreeBoardC4<N, W> sub_board = tried_board;
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
    cuda::std::pair<unsigned, unsigned> cell = {ix, ix};
    ThreeBoardC4<N, W> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<W>(stack, sub_board.known_on, sub_board.known_off);
    }
  }
}

template <unsigned N, unsigned W>
struct C4Traits {
  static_assert(N <= 64, "C4 solver currently supports N <= 64");
  static constexpr unsigned kN = N;
  static constexpr unsigned kW = W;
  static constexpr unsigned kSymForceMaxOn = (N / 2);

  static constexpr unsigned kRowOnZeroUnknownNum = 7;
  static constexpr unsigned kRowOnZeroUnknownDen = 4;

  using Board = ThreeBoardC4<N, W>;
  using Problem = ::Problem<W>;
  using Stack = DeviceStack<W>;
  using Output = OutputBuffer<W>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() { Board::init_tables_host(); }

  _DI_ static Cell pick_preferred_branch_cell(const BitBoard<W> &mask) {
    auto cell = mask.template first_origin_on<N>();
    return {static_cast<unsigned>(cell.first), static_cast<unsigned>(cell.second)};
  }

  _DI_ static unsigned pick_row_on_priority(const Board &board) {
    if constexpr (W == 32) {
      const unsigned lane = threadIdx.x & 31;
      const board_row_t<32> row_mask = (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);

      BitBoard<32> on_t = board.known_on.flip_diagonal();
      BitBoard<32> off_t = board.known_off.flip_diagonal();

      board_row_t<32> row_on = board.known_on.state & row_mask;
      board_row_t<32> row_off = board.known_off.state & row_mask;
      board_row_t<32> col_on = on_t.state & row_mask;
      board_row_t<32> col_off = off_t.state & row_mask;

      board_row_t<32> row_unknown = ~(row_on | row_off) & row_mask;
      board_row_t<32> col_unknown = ~(col_on | col_off) & row_mask;

      unsigned pivot_unknown = (row_unknown >> lane) & 1u;
      bool row_empty = (row_on | col_on) == 0;
      unsigned unknown_count = popcount<32>(row_unknown) + popcount<32>(col_unknown) - pivot_unknown;

      unsigned row0 = lane;
      unsigned row1 = lane;
      unsigned best0_unknown = 0xffffffffu;
      unsigned best1_unknown = 0xffffffffu;

      if (lane < N && unknown_count != 0) {
        if (row_empty) {
          best0_unknown = unknown_count;
        } else {
          best1_unknown = unknown_count;
        }
      }

      for (int offset = 16; offset > 0; offset /= 2) {
        unsigned other_row0 = __shfl_down_sync(0xffffffff, row0, offset);
        unsigned other0_unknown = __shfl_down_sync(0xffffffff, best0_unknown, offset);
        if (other0_unknown < best0_unknown ||
            (other0_unknown == best0_unknown && other_row0 < row0)) {
          row0 = other_row0;
          best0_unknown = other0_unknown;
        }

        unsigned other_row1 = __shfl_down_sync(0xffffffff, row1, offset);
        unsigned other1_unknown = __shfl_down_sync(0xffffffff, best1_unknown, offset);
        if (other1_unknown < best1_unknown ||
            (other1_unknown == best1_unknown && other_row1 < row1)) {
          row1 = other_row1;
          best1_unknown = other1_unknown;
        }
      }

      row0 = __shfl_sync(0xffffffff, row0, 0);
      row1 = __shfl_sync(0xffffffff, row1, 0);
      best0_unknown = __shfl_sync(0xffffffff, best0_unknown, 0);
      best1_unknown = __shfl_sync(0xffffffff, best1_unknown, 0);

      if (best1_unknown == 0xffffffffu)
        return row0;

      if (best0_unknown == 0xffffffffu)
        return row1;

      if ((best0_unknown * kRowOnZeroUnknownDen) <= (best1_unknown * kRowOnZeroUnknownNum)) {
        return row0;
      }
      return row1;
    } else {
      constexpr board_row_t<64> row_mask = (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1);
      const unsigned lane = threadIdx.x & 31;
      const BitBoard<64> on_t = board.known_on.flip_diagonal();
      const BitBoard<64> off_t = board.known_off.flip_diagonal();

      const unsigned row_even = 2 * lane;
      const unsigned row_odd = row_even + 1;

      const board_row_t<64> row_on_even =
          (((board_row_t<64>)board.known_on.state.y << 32) | board.known_on.state.x) & row_mask;
      const board_row_t<64> row_on_odd =
          (((board_row_t<64>)board.known_on.state.w << 32) | board.known_on.state.z) & row_mask;
      const board_row_t<64> row_off_even =
          (((board_row_t<64>)board.known_off.state.y << 32) | board.known_off.state.x) & row_mask;
      const board_row_t<64> row_off_odd =
          (((board_row_t<64>)board.known_off.state.w << 32) | board.known_off.state.z) & row_mask;

      const board_row_t<64> col_on_even =
          (((board_row_t<64>)on_t.state.y << 32) | on_t.state.x) & row_mask;
      const board_row_t<64> col_on_odd =
          (((board_row_t<64>)on_t.state.w << 32) | on_t.state.z) & row_mask;
      const board_row_t<64> col_off_even =
          (((board_row_t<64>)off_t.state.y << 32) | off_t.state.x) & row_mask;
      const board_row_t<64> col_off_odd =
          (((board_row_t<64>)off_t.state.w << 32) | off_t.state.z) & row_mask;

      unsigned row0 = row_even;
      unsigned row1 = row_even;
      unsigned best0_unknown = 0xffffffffu;
      unsigned best1_unknown = 0xffffffffu;

      if (row_even < N) {
        const unsigned row_on = popcount<64>(row_on_even);
        const unsigned row_off = popcount<64>(row_off_even);
        const unsigned col_on = popcount<64>(col_on_even);
        const unsigned col_off = popcount<64>(col_off_even);
        const unsigned row_unknown = N - row_on - row_off;
        const unsigned col_unknown = N - col_on - col_off;
        const unsigned pivot_unknown =
            (((row_on_even | row_off_even) & (board_row_t<64>(1) << row_even)) == 0) ? 1u : 0u;
        const unsigned unknown_count = row_unknown + col_unknown - pivot_unknown;

        if (unknown_count != 0) {
          const bool row_empty = (row_on + col_on) == 0;
          if (row_empty) {
            best0_unknown = unknown_count;
            row0 = row_even;
          } else {
            best1_unknown = unknown_count;
            row1 = row_even;
          }
        }
      }

      if (row_odd < N) {
        const unsigned row_on = popcount<64>(row_on_odd);
        const unsigned row_off = popcount<64>(row_off_odd);
        const unsigned col_on = popcount<64>(col_on_odd);
        const unsigned col_off = popcount<64>(col_off_odd);
        const unsigned row_unknown = N - row_on - row_off;
        const unsigned col_unknown = N - col_on - col_off;
        const unsigned pivot_unknown =
            (((row_on_odd | row_off_odd) & (board_row_t<64>(1) << row_odd)) == 0) ? 1u : 0u;
        const unsigned unknown_count = row_unknown + col_unknown - pivot_unknown;

        if (unknown_count != 0) {
          const bool row_empty = (row_on + col_on) == 0;
          if (row_empty) {
            if (unknown_count < best0_unknown || (unknown_count == best0_unknown && row_odd < row0)) {
              best0_unknown = unknown_count;
              row0 = row_odd;
            }
          } else {
            if (unknown_count < best1_unknown || (unknown_count == best1_unknown && row_odd < row1)) {
              best1_unknown = unknown_count;
              row1 = row_odd;
            }
          }
        }
      }

      for (int offset = 16; offset > 0; offset /= 2) {
        unsigned other_row0 = __shfl_down_sync(0xffffffff, row0, offset);
        unsigned other0_unknown = __shfl_down_sync(0xffffffff, best0_unknown, offset);
        if (other0_unknown < best0_unknown ||
            (other0_unknown == best0_unknown && other_row0 < row0)) {
          row0 = other_row0;
          best0_unknown = other0_unknown;
        }

        unsigned other_row1 = __shfl_down_sync(0xffffffff, row1, offset);
        unsigned other1_unknown = __shfl_down_sync(0xffffffff, best1_unknown, offset);
        if (other1_unknown < best1_unknown ||
            (other1_unknown == best1_unknown && other_row1 < row1)) {
          row1 = other_row1;
          best1_unknown = other1_unknown;
        }
      }

      row0 = __shfl_sync(0xffffffff, row0, 0);
      row1 = __shfl_sync(0xffffffff, row1, 0);
      best0_unknown = __shfl_sync(0xffffffff, best0_unknown, 0);
      best1_unknown = __shfl_sync(0xffffffff, best1_unknown, 0);

      if (best1_unknown == 0xffffffffu)
        return row0;

      if (best0_unknown == 0xffffffffu)
        return row1;

      if ((best0_unknown * kRowOnZeroUnknownDen) <= (best1_unknown * kRowOnZeroUnknownNum)) {
        return row0;
      }
      return row1;
    }
  }

  _DI_ static void seed_initial(Stack *stack) {
    constexpr unsigned seed_row = N / 2;
    Board board;
    resolve_outcome_row<N, W>(board, seed_row, stack);
  }

  _DI_ static void branch_fallback(const Board &board, Stack *stack) {
    unsigned row = pick_row_on_priority(board);
    stats_record(StatId::RowBranches);
    resolve_outcome_row<N, W>(board, row, stack);
  }

  static void emit_solution(const Problem &problem) {
    board_array_t<Board::FULL_W> expanded{};

    for (unsigned y = 0; y < N; ++y) {
      board_row_t<W> row = problem.known_on[y];
      while (row != 0) {
        unsigned x = count_trailing_zeros<W>(row);
        int px = static_cast<int>(x);
        int py = static_cast<int>(y);

        for (int r = 0; r < 4; ++r) {
          int fx = px + static_cast<int>(N);
          int fy = py + static_cast<int>(N);
          if constexpr (Board::FULL_W == 32) {
            expanded[fy] |= (1U << fx);
          } else {
            expanded[fy] |= (1ULL << fx);
          }

          int nx = -py - 1;
          int ny = px;
          px = nx;
          py = ny;
        }

        row &= (row - 1);
      }
    }

    std::cout << to_rle<Board::FULL_N, Board::FULL_W>(expanded) << std::endl;
  }

  static void emit_frontier(const Problem &problem) {
    emit_frontier_rle<N, W>(problem);
  }
};

template <unsigned N, unsigned W>
int solve_with_device_stack_c4_w() {
  return solve_with_device_stack_impl<C4Traits<N, W>, false>(nullptr, nullptr, 0);
}

template <unsigned N, unsigned W>
int solve_with_device_stack_c4_w(const board_array_t<W> *seed_on,
                                 const board_array_t<W> *seed_off) {
  return solve_with_device_stack_impl<C4Traits<N, W>, false>(seed_on, seed_off, 0);
}

template <unsigned N, unsigned W>
int solve_with_device_stack_c4_w(unsigned frontier_min_on) {
  return solve_with_device_stack_impl<C4Traits<N, W>, true>(nullptr, nullptr, frontier_min_on);
}

template <unsigned N>
int solve_with_device_stack_c4() {
  if constexpr (N <= 32) {
    return solve_with_device_stack_c4_w<N, 32>();
  } else {
    return solve_with_device_stack_c4_w<N, 64>();
  }
}

template <unsigned N>
int solve_with_device_stack_c4(unsigned frontier_min_on) {
  if constexpr (N <= 32) {
    return solve_with_device_stack_c4_w<N, 32>(frontier_min_on);
  } else {
    return solve_with_device_stack_c4_w<N, 64>(frontier_min_on);
  }
}

template <unsigned N>
int solve_with_device_stack_c4(const board_array_t<32> *seed_on,
                               const board_array_t<32> *seed_off) {
  if constexpr (N <= 32) {
    return solve_with_device_stack_c4_w<N, 32>(seed_on, seed_off);
  } else {
    (void)seed_on;
    (void)seed_off;
    std::cerr << "[c4] 32-bit seeds are invalid for N > 32\n";
    return 1;
  }
}

template <unsigned N>
int solve_with_device_stack_c4(const board_array_t<64> *seed_on,
                               const board_array_t<64> *seed_off) {
  if constexpr (N > 32) {
    return solve_with_device_stack_c4_w<N, 64>(seed_on, seed_off);
  } else {
    (void)seed_on;
    (void)seed_off;
    std::cerr << "[c4] 64-bit seeds are invalid for N <= 32\n";
    return 1;
  }
}

template int solve_with_device_stack_c4<N>();
template int solve_with_device_stack_c4<N>(unsigned);
template int solve_with_device_stack_c4<N>(const board_array_t<32> *, const board_array_t<32> *);
template int solve_with_device_stack_c4<N>(const board_array_t<64> *, const board_array_t<64> *);
