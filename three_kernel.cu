#include <algorithm>
#include <vector>
#include <iostream>

#include "board.cu"
#include "three_board.cu"

#include "three_kernel.hpp"
#include "parsing.hpp"

#include "params.hpp"
#include "three_search.cuh"

template <unsigned N, unsigned W>
__device__ unsigned pick_center_col(board_row_t<W> bits) {
  constexpr int center_right = static_cast<int>(N / 2);
  constexpr int center_left = static_cast<int>((N - 1) / 2);
  board_row_t<W> right_mask = bits & (~((board_row_t<W>(1) << center_right) - 1));
  board_row_t<W> left_mask = bits & ((board_row_t<W>(1) << (center_left + 1)) - 1);

  int right = find_first_set<W>(right_mask);
  int left = find_last_set<W>(left_mask);

  bool has_right = right_mask != 0;
  bool has_left = left_mask != 0;

  if (!has_left && has_right)
    return right;

  if (!has_right && has_left)
    return left;

  int dist_right = right - center_right;
  int dist_left = center_left - left;
  return static_cast<unsigned>(dist_right <= dist_left ? right : left);
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

  static void init_line_table_host_32() {
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

    cudaMalloc((void **)&d_line_table, total_rows * sizeof(uint32_t));
    cudaMemcpy(d_line_table, host_table.data(), total_rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_line_table_32, &d_line_table, sizeof(d_line_table));
  }

  static void init_host() {
    init_lookup_tables_host();
    init_relevant_endpoint_host(N);
    init_relevant_endpoint_host_64(N);
    init_line_table_host_32();
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
int solve_with_device_stack(const FrontierConfig &config) {
  return solve_with_device_stack_impl<AsymTraits<N, W>, true>(nullptr, nullptr, &config);
}

template <unsigned N, unsigned W>
int solve_with_device_stack() {
  return solve_with_device_stack_impl<AsymTraits<N, W>, false>(nullptr, nullptr, nullptr);
}

template <unsigned N, unsigned W>
int solve_with_device_stack(const board_array_t<W> *seed_on,
                            const board_array_t<W> *seed_off) {
  return solve_with_device_stack_impl<AsymTraits<N, W>, false>(seed_on, seed_off, nullptr);
}

// Explicitly instantiate the template to the N in params.hpp, or it doesn't get compiled at all.
template int solve_with_device_stack<N, 32>();
template int solve_with_device_stack<N, 32>(const board_array_t<32> *, const board_array_t<32> *);
template int solve_with_device_stack<N, 32>(const FrontierConfig &);
