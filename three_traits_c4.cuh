#pragma once

#include "three_search.cuh"
#include "parsing.hpp"
#include "compare_with_unknowns.cuh"

template <unsigned N>
void init_line_table_c4_host();
template <unsigned N>
__device__ void resolve_outcome_row(const ThreeBoardC4<N> &board,
                                    unsigned ix,
                                    DeviceStack<32> *stack) {
  ThreeBoardC4<N> tried_board = board;

  board_row_t<32> row_known_on = board.known_on.row(ix);
  board_row_t<32> row_known_off = board.known_off.row(ix);
  board_row_t<32> col_known_on = board.known_on.column(ix);
  board_row_t<32> col_known_off = board.known_off.column(ix);

  const board_row_t<32> row_mask = (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
  board_row_t<32> row_remaining = ~row_known_on & ~row_known_off & row_mask;
  board_row_t<32> col_remaining = ~col_known_on & ~col_known_off & row_mask;

  const bool pivot_unknown = (row_remaining & (board_row_t<32>(1) << ix)) != 0;
  row_remaining &= ~(board_row_t<32>(1) << ix);
  col_remaining &= ~(board_row_t<32>(1) << ix);

  uint64_t remaining = (uint64_t(row_remaining) << N) | col_remaining;

  if ((row_known_on | col_known_on) == 0) {
    if (remaining != 0) {
      unsigned keep = find_last_set<64>(remaining);
      remaining &= ~(uint64_t(1) << keep);
    }
  }

  while (remaining != 0) {
    unsigned bit = find_first_set<64>(remaining);
    cuda::std::pair<unsigned, unsigned> cell;

    if (bit >= N)
      cell = {bit - N, ix};
    else
      cell = {ix, bit};

    ThreeBoardC4<N> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<32>(stack, sub_board.known_on, sub_board.known_off);
    }

    tried_board.known_off.set(cell);
    remaining &= (remaining - 1);
  }

  if (pivot_unknown) {
    cuda::std::pair<unsigned, unsigned> cell = {ix, ix};
    ThreeBoardC4<N> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      stack_push<32>(stack, sub_board.known_on, sub_board.known_off);
    }
  }
}

template <unsigned N>
struct C4Traits {
  static constexpr unsigned kN = N;
  static constexpr unsigned kW = 32;
  static constexpr unsigned kSymForceMaxOn = (N / 2);

  // These are currently unused
  static constexpr unsigned kCellBranchRowScoreThreshold = 60;
  static constexpr int kCellBranchWColUnknown = 3;
  static constexpr int kCellBranchWRowUnknown = 4;
  static constexpr int kCellBranchWColOn = 3;
  static constexpr int kCellBranchWColOff = 8;
  static constexpr int kCellBranchWEndpointOff = 0;
  static constexpr int kCellBranchWEndpointOn = 10;
  static constexpr unsigned kRowOnZeroUnknownNum = 7;
  static constexpr unsigned kRowOnZeroUnknownDen = 4;

  using Board = ThreeBoardC4<N>;
  using Problem = Problem<32>;
  using Stack = DeviceStack<32>;
  using Output = OutputBuffer<32>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() {
    init_lookup_tables_host();
    init_relevant_endpoint_host(N);
    init_relevant_endpoint_host_64(ThreeBoardC4<N>::FULL_N);
    init_line_table_c4_host<N>();
  }

  _DI_ static LexStatus canonical_with_forced(Board &board, ForcedCell &forced) {
    BitBoard<32> diag_on = board.known_on.flip_diagonal();
    BitBoard<32> diag_off = board.known_off.flip_diagonal();
    BitBoard<32> bounds = Board::bounds();
    diag_on &= bounds;
    diag_off &= bounds;
    ForceCandidate local_force{};
    LexStatus order = compare_with_unknowns_forced<32>(board.known_on,
                                                       board.known_off,
                                                       diag_on,
                                                       diag_off,
                                                       bounds,
                                                       local_force);
    if (order == LexStatus::Unknown && local_force.has_force) {
      forced.has_force = true;
      forced.force_on = local_force.force_on;
      auto cell = local_force.cell;
      if (local_force.on_b) {
        cell = {cell.second, cell.first};
      }
      forced.cell = cell;
    }
    return order;
  }

  _DI_ static Board load_board(const Problem &problem) {
    Board board;
    board.known_on = BitBoard<32>::load(problem.known_on.data());
    board.known_off = BitBoard<32>::load(problem.known_off.data());
    board.apply_bounds();
    return board;
  }

  _DI_ static bool complete(const Board &board) {
    BitBoard<32> unknown = ~(board.known_on | board.known_off) & Board::bounds();
    return unknown.empty();
  }

  _DI_ static BitBoard<32> vulnerable(const Board &board) {
    return board.vulnerable();
  }

  _DI_ static BitBoard<32> semivulnerable(const Board &board) {
    return board.semivulnerable();
  }

  _DI_ static BitBoard<32> quasivulnerable(const Board &board) {
    return board.quasivulnerable();
  }

  _DI_ static Cell pick_preferred_branch_cell(const BitBoard<32> &mask) {
    auto cell = mask.template first_origin_on<N>();
    return {static_cast<unsigned>(cell.first), static_cast<unsigned>(cell.second)};
  }

  _DI_ static unsigned pick_row_on_priority(const Board &board) {
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
    } else {
      return row1;
    }
  }

  _DI_ static void seed_initial(Stack *stack) {
    constexpr unsigned seed_row = N / 2;
    Board board;
    resolve_outcome_row<N>(board, seed_row, stack);
  }

  _DI_ static void branch_fallback(const Board &board, Stack *stack) {
    unsigned row = pick_row_on_priority(board);
    stats_record(StatId::RowBranches);
    resolve_outcome_row<N>(board, row, stack);
  }

  static void emit_solution(const Problem &problem) {
    board_array_t<Board::FULL_W> expanded{};

    for (unsigned y = 0; y < N; ++y) {
      board_row_t<32> row = problem.known_on[y];
      while (row != 0) {
        unsigned x = static_cast<unsigned>(__builtin_ctz(row));
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
    emit_frontier_rle<N, 32>(problem);
  }
};
