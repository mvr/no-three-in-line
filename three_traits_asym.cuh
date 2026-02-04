#pragma once

#include "three_search.cuh"
#include "three_kernel.hpp"
#include "params.hpp"

template <unsigned N, unsigned W>
struct AsymTraits {
  static constexpr unsigned kN = N;
  static constexpr unsigned kW = W;
  static constexpr unsigned kSymForceMaxOn = N - 2;

  static constexpr unsigned kCellBranchRowScoreThreshold = 20;

  static constexpr int kCellBranchWColUnknown = 3;
  static constexpr int kCellBranchWRowUnknown = 4;
  static constexpr int kCellBranchWColOn = 3;
  static constexpr int kCellBranchWColOff = 8;
  static constexpr int kCellBranchWEndpointOff = 0;
  static constexpr int kCellBranchWEndpointOn = 10;

  using Board = ThreeBoard<N, W>;
  using Problem = Problem<W>;
  using Stack = DeviceStack<W>;
  using Output = OutputBuffer<W>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() {
    init_lookup_tables_host();
    init_relevant_endpoint_host(N);
    init_relevant_endpoint_host_64(N);
    init_line_table_host_32();
  }

  _DI_ static Board load_board(const Problem &problem) {
    Board board;
    board.known_on = BitBoard<W>::load(problem.known_on.data());
    board.known_off = BitBoard<W>::load(problem.known_off.data());
    return board;
  }

  _DI_ static bool complete(const Board &board) {
    return board.complete();
  }

  _DI_ static LexStatus canonical_with_forced(Board &board, ForcedCell &forced) {
    return board.is_canonical_orientation_with_forced(forced);
  }

  _DI_ static BitBoard<W> vulnerable(const Board &board) {
    return board.vulnerable();
  }

  _DI_ static BitBoard<W> semivulnerable(const Board &board) {
    return board.semivulnerable();
  }

  _DI_ static BitBoard<W> quasivulnerable(const Board &board) {
    return board.quasivulnerable();
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
