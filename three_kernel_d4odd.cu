#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>

#include "board.cu"
#include "three_board_d4odd.cu"

#include "params.hpp"
#include "three_kernel_d4odd.hpp"
#include "three_search.cuh"

static constexpr unsigned D4ODD_FAMILY_MAX = 32;
static constexpr unsigned D4ODD_FAMILY_ORDER_MAX = 2 * D4ODD_FAMILY_MAX;

__device__ __constant__ uint16_t
    g_d4odd_family_visit_order[D4ODD_FAMILY_MAX * D4ODD_FAMILY_ORDER_MAX];
__device__ __constant__ uint8_t g_d4odd_family_visit_count[D4ODD_FAMILY_MAX];

template <unsigned N>
inline void init_d4odd_family_visit_order_host() {
  using Board = ThreeBoardD4Odd<N, 32>;
  static_assert(Board::STORE_W <= D4ODD_FAMILY_MAX, "D4Odd family table exceeds max family count");
  static_assert((2 * Board::STORE_W) <= D4ODD_FAMILY_ORDER_MAX,
                "D4Odd family table exceeds max family order length");

  struct Candidate {
    uint16_t packed;
    unsigned score;
    unsigned tiebreak;
  };

  std::array<uint16_t, D4ODD_FAMILY_MAX * D4ODD_FAMILY_ORDER_MAX> host_order{};
  std::array<uint8_t, D4ODD_FAMILY_MAX> host_count{};

  for (unsigned family = 0; family < Board::STORE_W; ++family) {
    std::array<Candidate, D4ODD_FAMILY_ORDER_MAX> candidates{};
    unsigned candidate_count = 0u;

    for (unsigned ly = 0; ly < Board::STORE_H; ++ly) {
      for (unsigned lx = 0; lx < Board::STORE_W; ++lx) {
        const bool row_family_match =
            ((lx <= ly) && (family == ly)) ||
            ((lx > ly) && (ly < Board::H) && (family == (ly + 1u)));
        const bool col_family_match = (Board::storage_col_family(lx, ly) == family);
        if (!row_family_match && !col_family_match) {
          continue;
        }

        const auto full = Board::local_to_full({lx, ly});
        const int fx = full.first;
        const int fy = full.second;
        const int abs_fx = (fx < 0) ? -fx : fx;
        Candidate candidate{};
        candidate.packed = static_cast<uint16_t>((ly << 8) | lx);
        candidate.score = static_cast<unsigned>(fx * fx + fy * fy);
        candidate.tiebreak = (static_cast<unsigned>(fy) << 8) |
                             (static_cast<unsigned>(abs_fx) << 5) |
                             lx;

        candidates[candidate_count++] = candidate;
      }
    }

    std::sort(candidates.begin(),
              candidates.begin() + candidate_count,
              [](const Candidate &a, const Candidate &b) {
                if (a.score != b.score) {
                  return a.score < b.score;
                }
                return a.tiebreak < b.tiebreak;
              });

    host_count[family] = static_cast<uint8_t>(candidate_count);
    const unsigned base = family * D4ODD_FAMILY_ORDER_MAX;
    for (unsigned idx = 0; idx < candidate_count; ++idx) {
      host_order[base + idx] = candidates[idx].packed;
    }
  }

  cudaMemcpyToSymbol(g_d4odd_family_visit_order,
                     host_order.data(),
                     host_order.size() * sizeof(uint16_t));
  cudaMemcpyToSymbol(g_d4odd_family_visit_count,
                     host_count.data(),
                     host_count.size() * sizeof(uint8_t));
}

template <unsigned N>
struct D4OddTraits {
  using Board = ThreeBoardD4Odd<N, 32>;
  static constexpr unsigned kN = Board::STORE_W;
  static constexpr unsigned kW = 32;
  static constexpr unsigned kSymForceMaxOn = (Board::H / 2);

  using Problem = Problem<32>;
  using Stack = DeviceStack<32>;
  using Output = OutputBuffer<32>;
  using Cell = cuda::std::pair<unsigned, unsigned>;

  static void init_host() {
    Board::init_tables_host();
    init_d4odd_family_visit_order_host<N>();
  }

  _DI_ static Cell pick_preferred_branch_cell(const BitBoard<32> &mask) {
    const unsigned lane = threadIdx.x & 31;
    unsigned best_score = 0xffffffffu;
    unsigned best_tiebreak = 0xffffffffu;
    unsigned best_x = 0;
    unsigned best_y = 0;

    if (lane < Board::STORE_H) {
      board_row_t<32> row = mask.state & Board::row_mask();
      while (row != 0u) {
        const unsigned x = count_trailing_zeros<32>(row);
        const auto full = Board::local_to_full({x, lane});
        const int fx = full.first;
        const int fy = full.second;
        const int abs_fx = (fx < 0) ? -fx : fx;
        const unsigned score = static_cast<unsigned>(fx * fx + fy * fy);
        const unsigned tiebreak = (static_cast<unsigned>(fy) << 8) |
                                  (static_cast<unsigned>(abs_fx) << 5) |
                                  x;
        if (score < best_score ||
            (score == best_score && tiebreak < best_tiebreak)) {
          best_score = score;
          best_tiebreak = tiebreak;
          best_x = x;
          best_y = lane;
        }
        row &= (row - 1u);
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      const unsigned other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
      const unsigned other_tiebreak = __shfl_down_sync(0xffffffffu, best_tiebreak, offset);
      const unsigned other_x = __shfl_down_sync(0xffffffffu, best_x, offset);
      const unsigned other_y = __shfl_down_sync(0xffffffffu, best_y, offset);
      if (other_score < best_score ||
          (other_score == best_score && other_tiebreak < best_tiebreak)) {
        best_score = other_score;
        best_tiebreak = other_tiebreak;
        best_x = other_x;
        best_y = other_y;
      }
    }

    best_score = __shfl_sync(0xffffffffu, best_score, 0);
    best_tiebreak = __shfl_sync(0xffffffffu, best_tiebreak, 0);
    best_x = __shfl_sync(0xffffffffu, best_x, 0);
    best_y = __shfl_sync(0xffffffffu, best_y, 0);
    if (best_score == 0xffffffffu) {
      return {0u, 0u};
    }
    return {best_x, best_y};
  }

  _DI_ static void seed_initial(Stack *stack) {
    Board board;
    stack_push<32>(stack, board.known_on, board.known_off);
  }

  _DI_ static unsigned pick_family_on_priority(const Board &board) {
    const unsigned lane = threadIdx.x & 31;
    const bool active = lane < Board::STORE_W;

    const BitBoard<32> unknown = (~board.known_on & ~board.known_off) & Board::bounds();
    const BinaryCountSaturating3<32> unknown_counter =
        Board::template family_on_counts_impl<BinaryCountSaturating3<32>>(unknown);
    unsigned earliest_family = 0xffffffffu;

    if (active) {
      const unsigned unknown_nonzero =
          ((unknown_counter.bit0 >> lane) & 1u) |
          ((unknown_counter.bit1 >> lane) & 1u) |
          ((unknown_counter.bit2 >> lane) & 1u);
      if (unknown_nonzero != 0u) {
        earliest_family = lane;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      const unsigned other_earliest = __shfl_down_sync(0xffffffffu, earliest_family, offset);
      if (other_earliest < earliest_family) {
        earliest_family = other_earliest;
      }
    }

    earliest_family = __shfl_sync(0xffffffffu, earliest_family, 0);

    if (earliest_family != 0xffffffffu) {
      return earliest_family;
    }
    return 0u;
  }

  _DI_ static void resolve_outcome_family(const Board &board,
                                          unsigned family,
                                          Stack *stack) {
    if (family >= Board::STORE_W) {
      return;
    }

    Board tried_board = board;
    BitBoard<32> remaining =
        ((~board.known_on & ~board.known_off) & Board::bounds()) &
        Board::logical_family_mask(family);
    if (remaining.empty()) {
      return;
    }

    const unsigned family_count =
        static_cast<unsigned>(g_d4odd_family_visit_count[family]);
    const uint16_t *visit =
        &g_d4odd_family_visit_order[family * D4ODD_FAMILY_ORDER_MAX];
    for (unsigned idx = 0; idx < family_count; ++idx) {
      const uint16_t packed = *visit++;
      const unsigned x = static_cast<unsigned>(packed & 0xffu);
      const unsigned y = static_cast<unsigned>(packed >> 8);
      if (!remaining.get(static_cast<int>(x), static_cast<int>(y))) {
        continue;
      }

      Board sub_board = tried_board;
      sub_board.known_on.set(static_cast<int>(x), static_cast<int>(y));
      sub_board.eliminate_all_lines({x, y});
      sub_board.propagate();
      if (sub_board.consistent()) {
        stack_push<32>(stack, sub_board.known_on, sub_board.known_off);
      } else {
        stats_record(StatId::InconsistentNodes);
      }

      tried_board.known_off.set(static_cast<int>(x), static_cast<int>(y));
      remaining.erase(static_cast<int>(x), static_cast<int>(y));
      if (remaining.empty()) {
        break;
      }
    }
  }

  _DI_ static void branch_fallback(const Board &board, Stack *stack) {
    stats_record(StatId::RowBranches);
    const unsigned family = pick_family_on_priority(board);
    resolve_outcome_family(board, family, stack);
  }

  static void emit_solution(const Problem &problem) {
    board_array_t<Board::FULL_W> expanded{};
    auto set_full = [&](int fx, int fy) {
      const int ox = static_cast<int>(Board::H);
      const int oy = static_cast<int>(Board::H);
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

    constexpr board_row_t<32> row_mask = Board::row_mask();
    for (unsigned ly = 0; ly < Board::STORE_H; ++ly) {
      board_row_t<32> row = problem.known_on[ly] & row_mask;
      while (row != 0u) {
        const unsigned lx = count_trailing_zeros<32>(row);
        const auto p = Board::local_to_full({lx, ly});

        set_full(p.first, p.second);
        set_full(p.second, p.first);
        set_full(-p.first, -p.second);
        set_full(-p.second, -p.first);

        row &= (row - 1u);
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
int solve_with_device_stack_d4odd(const SearchOptions<32> &options) {
  if (options.mode == SearchMode::Frontier) {
    return solve_with_device_stack_impl<D4OddTraits<N>, true>(options);
  }
  return solve_with_device_stack_impl<D4OddTraits<N>, false>(options);
}

template int solve_with_device_stack_d4odd<N>(const SearchOptions<32> &);
