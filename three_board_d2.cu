#pragma once

#include "common.hpp"
#include "board.cu"
#include "three_board.cu"

#include <cuda/std/utility>

__device__ const uint32_t *__restrict__ g_d2_line_table_32 = nullptr;
__device__ const ulonglong2 *__restrict__ g_d2_line_table_64 = nullptr;

// D2 reflection-symmetric board.
// Full board is 2N x 2N, represented by storing the top half: N rows x 2N cols.
// Reflection is across the horizontal midline: (x, y) <-> (x, 2N - 1 - y).
template <unsigned N, unsigned W>
struct ThreeBoardD2 {
  static_assert(W == 32 || W == 64, "ThreeBoardD2 supports W=32 or W=64");
  static_assert(N <= 32, "ThreeBoardD2 requires N <= 32");
  static_assert((W == 32 && N <= 16) || (W == 64), "W=32 requires N <= 16");

  static constexpr unsigned FULL_N = 2 * N;
  static constexpr unsigned FULL_W = (FULL_N <= 32) ? 32 : 64;
  static constexpr unsigned LINE_ROWS =
      LINE_TABLE_FULL_WARP_LOAD ? 32
                                : ((W == 32) ? ((N + 7u) & ~7u) : ((((N + 1u) >> 1) + 7u) & ~7u));
  static_assert(LINE_ROWS <= 32, "D2 line table rows must fit one warp");

  BitBoard<W> known_on;
  BitBoard<W> known_off;

  _DI_ ThreeBoardD2() : known_on{}, known_off{} {}
  _DI_ ThreeBoardD2(BitBoard<W> on, BitBoard<W> off) : known_on{on}, known_off{off} {}

  static _DI_ BitBoard<W> bounds();
  static _DI_ BitBoard<W> relevant_endpoint(cuda::std::pair<unsigned, unsigned> p);
  static void init_line_table_host();
  static void init_tables_host();

  _DI_ bool consistent() const;
  _DI_ unsigned unknown_pop() const;
  _DI_ bool complete() const;
  _DI_ LexStatus canonical_with_forced(ForcedCell &forced) const;
  static _DI_ ThreeBoardD2<N, W> load_from(const board_array_t<W> &on,
                                           const board_array_t<W> &off);
  _DI_ bool operator==(const ThreeBoardD2<N, W> &other) const;

  _DI_ ThreeBoardD2<N, W> force_orthogonal() const;
  _DI_ BitBoard<W> vulnerable() const;
  _DI_ BitBoard<W> semivulnerable() const;
  _DI_ BitBoard<W> quasivulnerable() const;
  template <unsigned UnknownTarget>
  _DI_ BitBoard<W> semivulnerable_like() const;
  _DI_ void apply_bounds();

  static _DI_ int mirror_y(int y);

  _DI_ BitBoard<W> eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                  cuda::std::pair<unsigned, unsigned> q) const;
  _DI_ BitBoard<W> eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                       cuda::std::pair<unsigned, unsigned> q) const;
  _DI_ BitBoard<W> eliminate_pair(cuda::std::pair<int, int> p,
                                  cuda::std::pair<int, int> q) const;
  _DI_ BitBoard<W> eliminate_pair_steps(cuda::std::pair<int, int> p,
                                        cuda::std::pair<int, int> q,
                                        int step_x,
                                        int step_y) const;

  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<W> seed);

  _DI_ void propagate();
};

template <unsigned N>
__global__ void init_d2_line_table_kernel_32(uint32_t *__restrict__ table) {
  constexpr unsigned cell_count = N * (2 * N);
  constexpr unsigned line_rows = ThreeBoardD2<N, 32>::LINE_ROWS;
  const unsigned pair_idx = blockIdx.x;
  if (pair_idx >= cell_count * cell_count) {
    return;
  }
  const unsigned lane = threadIdx.x & 31;

  const unsigned p_idx = pair_idx / cell_count;
  const unsigned q_idx = pair_idx - p_idx * cell_count;
  const unsigned px = p_idx % (2 * N);
  const unsigned py = p_idx / (2 * N);
  const unsigned qx = q_idx % (2 * N);
  const unsigned qy = q_idx / (2 * N);

  ThreeBoardD2<N, 32> board;
  BitBoard<32> mask = board.eliminate_line_slow({px, py}, {qx, qy});

  if (lane < line_rows) {
    table[static_cast<size_t>(pair_idx) * line_rows + lane] = mask.state;
  }
}

template <unsigned N>
__global__ void init_d2_line_table_kernel_64(ulonglong2 *__restrict__ table) {
  constexpr unsigned cell_count = N * (2 * N);
  constexpr unsigned line_rows = ThreeBoardD2<N, 64>::LINE_ROWS;
  const unsigned pair_idx = blockIdx.x;
  if (pair_idx >= cell_count * cell_count) {
    return;
  }
  const unsigned lane = threadIdx.x & 31;

  const unsigned p_idx = pair_idx / cell_count;
  const unsigned q_idx = pair_idx - p_idx * cell_count;
  const unsigned px = p_idx % (2 * N);
  const unsigned py = p_idx / (2 * N);
  const unsigned qx = q_idx % (2 * N);
  const unsigned qy = q_idx / (2 * N);

  ThreeBoardD2<N, 64> board;
  BitBoard<64> mask = board.eliminate_line_slow({px, py}, {qx, qy});

  if (lane < line_rows) {
    const size_t idx = static_cast<size_t>(pair_idx) * line_rows + lane;
    const uint64_t even = (static_cast<uint64_t>(mask.state.y) << 32) |
                          static_cast<uint64_t>(mask.state.x);
    const uint64_t odd = (static_cast<uint64_t>(mask.state.w) << 32) |
                         static_cast<uint64_t>(mask.state.z);
    table[idx] = make_ulonglong2(even, odd);
  }
}

template <unsigned N, unsigned W>
inline void ThreeBoardD2<N, W>::init_line_table_host() {
  static uint32_t *d_table_32 = nullptr;
  static ulonglong2 *d_table_64 = nullptr;

  constexpr unsigned cell_count = N * (2 * N);
  constexpr size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
  constexpr size_t total_rows = total_entries * LINE_ROWS;

  if constexpr (W == 32) {
    if (d_table_32 != nullptr) {
      cudaFree(d_table_32);
      d_table_32 = nullptr;
    }
    cudaMalloc((void **)&d_table_32, total_rows * sizeof(uint32_t));
    init_d2_line_table_kernel_32<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_32);
    cudaGetLastError();
    cudaDeviceSynchronize();
  } else {
    if (d_table_64 != nullptr) {
      cudaFree(d_table_64);
      d_table_64 = nullptr;
    }
    cudaMalloc((void **)&d_table_64, total_rows * sizeof(ulonglong2));
    init_d2_line_table_kernel_64<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_64);
    cudaGetLastError();
    cudaDeviceSynchronize();
  }

  if constexpr (W == 32) {
    cudaMemcpyToSymbol(g_d2_line_table_32, &d_table_32, sizeof(d_table_32));
  } else {
    cudaMemcpyToSymbol(g_d2_line_table_64, &d_table_64, sizeof(d_table_64));
  }
}

template <unsigned N, unsigned W>
inline void ThreeBoardD2<N, W>::init_tables_host() {
  init_lookup_tables_host();
  if constexpr (W == 32) {
    init_relevant_endpoint_host(N);
  } else {
    init_relevant_endpoint_host_64(N);
  }
  init_line_table_host();
}

// --- Inline implementation -------------------------------------------------

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::bounds() {
  return BitBoard<W>::rect(FULL_N, N);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::relevant_endpoint(cuda::std::pair<unsigned, unsigned>) {
  return bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardD2<N, W>::apply_bounds() {
  const BitBoard<W> b = bounds();
  known_on &= b;
  known_off &= b;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardD2<N, W>::consistent() const {
  return (known_on & known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ unsigned ThreeBoardD2<N, W>::unknown_pop() const {
  return N * FULL_N - (known_on | known_off).pop();
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardD2<N, W>::complete() const {
  BitBoard<W> unknown = ~(known_on | known_off) & bounds();
  return unknown.empty();
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoardD2<N, W>::canonical_with_forced(ForcedCell &forced) const {
  BitBoard<W> flip_h_on = known_on.flip_horizontal().rotate_torus(FULL_N, 0);
  BitBoard<W> flip_h_off = known_off.flip_horizontal().rotate_torus(FULL_N, 0);
  const BitBoard<W> bds = bounds();
  flip_h_on &= bds;
  flip_h_off &= bds;

  ForceCandidate local_force{};
  LexStatus order = compare_with_unknowns_forced<W>(known_on,
                                                     known_off,
                                                     flip_h_on,
                                                     flip_h_off,
                                                     bds,
                                                     local_force);

  if (order == LexStatus::Unknown && local_force.has_force) {
    forced.has_force = true;
    forced.force_on = local_force.force_on;
    auto cell = local_force.cell;
    if (local_force.on_b) {
      cell = {static_cast<int>(FULL_N) - 1 - cell.first, cell.second};
    }
    forced.cell = cell;
  }

  return order;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoardD2<N, W> ThreeBoardD2<N, W>::load_from(const board_array_t<W> &on,
                                                      const board_array_t<W> &off) {
  ThreeBoardD2<N, W> board;
  board.known_on = BitBoard<W>::load(on.data());
  board.known_off = BitBoard<W>::load(off.data());
  board.apply_bounds();
  return board;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardD2<N, W>::operator==(const ThreeBoardD2<N, W> &other) const {
  return known_on == other.known_on && known_off == other.known_off;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoardD2<N, W> ThreeBoardD2<N, W>::force_orthogonal() const {
  ThreeBoardD2<N, W> result = *this;

  if constexpr (W == 32) {
    const board_row_t<32> row_mask = (FULL_N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << FULL_N) - 1u);
    const unsigned lane = threadIdx.x & 31;

    // Row target: exactly 2 ON per stored row.
    if (lane < N) {
      const unsigned on_row = popcount<32>(known_on.state);
      const unsigned off_row = popcount<32>(known_off.state);

      if (on_row == 2) {
        result.known_off.state |= (~known_on.state) & row_mask;
      }
      if (on_row > 2) {
        result.known_on.state = ~0u;
        result.known_off.state = ~0u;
      }

      if (off_row == FULL_N - 2) {
        result.known_on.state |= (~known_off.state) & row_mask;
      }
      if (off_row > FULL_N - 2) {
        result.known_on.state = ~0u;
        result.known_off.state = ~0u;
      }
    }

    // Column target: exactly 1 ON per stored column.
    const BinaryCountSaturating<32> col_on = BinaryCountSaturating<32>::vertical(known_on.state);
    const board_row_t<32> col_on_eq_1 = col_on.bit0 & ~col_on.bit1;
    const board_row_t<32> col_on_gt_1 = col_on.bit1;

    result.known_off.state |= (~known_on.state) & col_on_eq_1;
    result.known_on.state |= col_on_gt_1;
    result.known_off.state |= col_on_gt_1;

    const BitBoard<32> not_known_off = (~known_off) & bounds();
    const BinaryCountSaturating<32> col_not_off = BinaryCountSaturating<32>::vertical(not_known_off.state);
    const board_row_t<32> col_not_off_eq_1 = col_not_off.bit0 & ~col_not_off.bit1;
    const board_row_t<32> col_not_off_lt_1 = ~col_not_off.bit0 & ~col_not_off.bit1;

    result.known_on.state |= (~known_off.state) & col_not_off_eq_1;
    result.known_on.state |= col_not_off_lt_1;
    result.known_off.state |= col_not_off_lt_1;
  } else {
    const board_row_t<32> row_mask_x = (FULL_N >= 32) ? 0xffffffffu : ((board_row_t<32>(1) << FULL_N) - 1u);
    board_row_t<32> row_mask_y;
    if constexpr (FULL_N <= 32) {
      row_mask_y = 0;
    } else if constexpr (FULL_N >= 64) {
      row_mask_y = 0xffffffffu;
    } else {
      row_mask_y = (board_row_t<32>(1) << (FULL_N - 32)) - 1u;
    }

    // Row target: exactly 2 ON per stored row.
    {
      const unsigned on_even = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
      const unsigned off_even = popcount<32>(known_off.state.x) + popcount<32>(known_off.state.y);

      if (on_even == 2) {
        result.known_off.state.x |= (~known_on.state.x) & row_mask_x;
        result.known_off.state.y |= (~known_on.state.y) & row_mask_y;
      }
      if (on_even > 2) {
        result.known_on.state.x = ~0u;
        result.known_on.state.y = ~0u;
        result.known_off.state.x = ~0u;
        result.known_off.state.y = ~0u;
      }

      if (off_even == FULL_N - 2) {
        result.known_on.state.x |= (~known_off.state.x) & row_mask_x;
        result.known_on.state.y |= (~known_off.state.y) & row_mask_y;
      }
      if (off_even > FULL_N - 2) {
        result.known_on.state.x = ~0u;
        result.known_on.state.y = ~0u;
        result.known_off.state.x = ~0u;
        result.known_off.state.y = ~0u;
      }
    }

    {
      const unsigned on_odd = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
      const unsigned off_odd = popcount<32>(known_off.state.z) + popcount<32>(known_off.state.w);

      if (on_odd == 2) {
        result.known_off.state.z |= (~known_on.state.z) & row_mask_x;
        result.known_off.state.w |= (~known_on.state.w) & row_mask_y;
      }
      if (on_odd > 2) {
        result.known_on.state.z = ~0u;
        result.known_on.state.w = ~0u;
        result.known_off.state.z = ~0u;
        result.known_off.state.w = ~0u;
      }

      if (off_odd == FULL_N - 2) {
        result.known_on.state.z |= (~known_off.state.z) & row_mask_x;
        result.known_on.state.w |= (~known_off.state.w) & row_mask_y;
      }
      if (off_odd > FULL_N - 2) {
        result.known_on.state.z = ~0u;
        result.known_on.state.w = ~0u;
        result.known_off.state.z = ~0u;
        result.known_off.state.w = ~0u;
      }
    }

    // Column target: exactly 1 ON per stored column.
    {
      const BinaryCountSaturating<32> col_on_low = BinaryCountSaturating<32>::vertical(known_on.state.x) +
                                                   BinaryCountSaturating<32>::vertical(known_on.state.z);
      const board_row_t<32> col_on_low_eq_1 = col_on_low.bit0 & ~col_on_low.bit1;
      const board_row_t<32> col_on_low_gt_1 = col_on_low.bit1;

      result.known_off.state.x |= (~known_on.state.x) & col_on_low_eq_1;
      result.known_off.state.z |= (~known_on.state.z) & col_on_low_eq_1;
      result.known_on.state.x |= col_on_low_gt_1;
      result.known_on.state.z |= col_on_low_gt_1;
      result.known_off.state.x |= col_on_low_gt_1;
      result.known_off.state.z |= col_on_low_gt_1;

      const BinaryCountSaturating<32> col_on_high = BinaryCountSaturating<32>::vertical(known_on.state.y) +
                                                    BinaryCountSaturating<32>::vertical(known_on.state.w);
      const board_row_t<32> col_on_high_eq_1 = col_on_high.bit0 & ~col_on_high.bit1;
      const board_row_t<32> col_on_high_gt_1 = col_on_high.bit1;

      result.known_off.state.y |= (~known_on.state.y) & col_on_high_eq_1;
      result.known_off.state.w |= (~known_on.state.w) & col_on_high_eq_1;
      result.known_on.state.y |= col_on_high_gt_1;
      result.known_on.state.w |= col_on_high_gt_1;
      result.known_off.state.y |= col_on_high_gt_1;
      result.known_off.state.w |= col_on_high_gt_1;
    }

    {
      const BitBoard<64> not_known_off = (~known_off) & bounds();

      const BinaryCountSaturating<32> col_not_off_low = BinaryCountSaturating<32>::vertical(not_known_off.state.x) +
                                                        BinaryCountSaturating<32>::vertical(not_known_off.state.z);
      const board_row_t<32> col_not_off_low_eq_1 = col_not_off_low.bit0 & ~col_not_off_low.bit1;
      const board_row_t<32> col_not_off_low_lt_1 = ~col_not_off_low.bit0 & ~col_not_off_low.bit1;

      result.known_on.state.x |= (~known_off.state.x) & col_not_off_low_eq_1;
      result.known_on.state.z |= (~known_off.state.z) & col_not_off_low_eq_1;
      result.known_on.state.x |= col_not_off_low_lt_1;
      result.known_on.state.z |= col_not_off_low_lt_1;
      result.known_off.state.x |= col_not_off_low_lt_1;
      result.known_off.state.z |= col_not_off_low_lt_1;

      const BinaryCountSaturating<32> col_not_off_high = BinaryCountSaturating<32>::vertical(not_known_off.state.y) +
                                                         BinaryCountSaturating<32>::vertical(not_known_off.state.w);
      const board_row_t<32> col_not_off_high_eq_1 = col_not_off_high.bit0 & ~col_not_off_high.bit1;
      const board_row_t<32> col_not_off_high_lt_1 = ~col_not_off_high.bit0 & ~col_not_off_high.bit1;

      result.known_on.state.y |= (~known_off.state.y) & col_not_off_high_eq_1;
      result.known_on.state.w |= (~known_off.state.w) & col_not_off_high_eq_1;
      result.known_on.state.y |= col_not_off_high_lt_1;
      result.known_on.state.w |= col_not_off_high_lt_1;
      result.known_off.state.y |= col_not_off_high_lt_1;
      result.known_off.state.w |= col_not_off_high_lt_1;
    }
  }

  result.apply_bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::vulnerable() const {
  BitBoard<W> result;
  BitBoard<W> unknown = (~known_on & ~known_off) & bounds();

  if constexpr (W == 32) {
    const unsigned row_on = popcount<32>(known_on.state);
    const unsigned row_unknown = popcount<32>(unknown.state);
    if ((row_on == 1 && row_unknown == 2) || (row_on == 0 && row_unknown == 3)) {
      result.state = ~0u;
    }

    // Full-board column unknown=4 corresponds to stored unknown=2.
    // One additional OFF then forces the remaining cell ON in that column.
    const BinaryCountSaturating<32> col_on = BinaryCountSaturating<32>::vertical(known_on.state);
    const BinaryCountSaturating<32> col_unknown = BinaryCountSaturating<32>::vertical(unknown.state);
    const board_row_t<32> vulnerable_cols = col_on.template eq_target<0>() & col_unknown.template eq_target<2>();
    result.state |= unknown.state & vulnerable_cols;
  } else {
    const unsigned row_on_even = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
    const unsigned row_on_odd = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
    const unsigned row_unknown_even = popcount<32>(unknown.state.x) + popcount<32>(unknown.state.y);
    const unsigned row_unknown_odd = popcount<32>(unknown.state.z) + popcount<32>(unknown.state.w);

    if ((row_on_even == 1 && row_unknown_even == 2) || (row_on_even == 0 && row_unknown_even == 3)) {
      result.state.x = ~0u;
      result.state.y = ~0u;
    }
    if ((row_on_odd == 1 && row_unknown_odd == 2) || (row_on_odd == 0 && row_unknown_odd == 3)) {
      result.state.z = ~0u;
      result.state.w = ~0u;
    }

    const BinaryCountSaturating<32> col_on_even_low = BinaryCountSaturating<32>::vertical(known_on.state.x);
    const BinaryCountSaturating<32> col_on_odd_low = BinaryCountSaturating<32>::vertical(known_on.state.z);
    const BinaryCountSaturating<32> col_unknown_even_low = BinaryCountSaturating<32>::vertical(unknown.state.x);
    const BinaryCountSaturating<32> col_unknown_odd_low = BinaryCountSaturating<32>::vertical(unknown.state.z);

    const board_row_t<32> col_on_low_eq_0 = col_on_even_low.template eq_target<0>() & col_on_odd_low.template eq_target<0>();
    const board_row_t<32> col_unknown_low_eq_2 =
        (col_unknown_even_low.template eq_target<2>() & col_unknown_odd_low.template eq_target<0>()) |
        (col_unknown_even_low.template eq_target<1>() & col_unknown_odd_low.template eq_target<1>()) |
        (col_unknown_even_low.template eq_target<0>() & col_unknown_odd_low.template eq_target<2>());
    const board_row_t<32> vulnerable_cols_low = col_on_low_eq_0 & col_unknown_low_eq_2;

    result.state.x |= unknown.state.x & vulnerable_cols_low;
    result.state.z |= unknown.state.z & vulnerable_cols_low;

    const BinaryCountSaturating<32> col_on_even_high = BinaryCountSaturating<32>::vertical(known_on.state.y);
    const BinaryCountSaturating<32> col_on_odd_high = BinaryCountSaturating<32>::vertical(known_on.state.w);
    const BinaryCountSaturating<32> col_unknown_even_high = BinaryCountSaturating<32>::vertical(unknown.state.y);
    const BinaryCountSaturating<32> col_unknown_odd_high = BinaryCountSaturating<32>::vertical(unknown.state.w);

    const board_row_t<32> col_on_high_eq_0 = col_on_even_high.template eq_target<0>() & col_on_odd_high.template eq_target<0>();
    const board_row_t<32> col_unknown_high_eq_2 =
        (col_unknown_even_high.template eq_target<2>() & col_unknown_odd_high.template eq_target<0>()) |
        (col_unknown_even_high.template eq_target<1>() & col_unknown_odd_high.template eq_target<1>()) |
        (col_unknown_even_high.template eq_target<0>() & col_unknown_odd_high.template eq_target<2>());
    const board_row_t<32> vulnerable_cols_high = col_on_high_eq_0 & col_unknown_high_eq_2;

    result.state.y |= unknown.state.y & vulnerable_cols_high;
    result.state.w |= unknown.state.w & vulnerable_cols_high;
  }

  result &= unknown & bounds();
  return result;
}

template <unsigned N, unsigned W>
template <unsigned UnknownTarget>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::semivulnerable_like() const {
  static_assert(UnknownTarget < 8, "semivulnerable_like expects a target < 8");
  BitBoard<W> result;
  const BitBoard<W> unknown = (~known_on & ~known_off) & bounds();

  if constexpr (W == 32) {
    const unsigned row_unknown = popcount<32>(unknown.state);
    if (known_on.state == 0 && row_unknown == UnknownTarget) {
      result.state = ~0u;
    }
  } else {
    const unsigned row_unknown_even = popcount<32>(unknown.state.x) + popcount<32>(unknown.state.y);
    if ((known_on.state.x | known_on.state.y) == 0 && row_unknown_even == UnknownTarget) {
      result.state.x = ~0u;
      result.state.y = ~0u;
    }

    const unsigned row_unknown_odd = popcount<32>(unknown.state.z) + popcount<32>(unknown.state.w);
    if ((known_on.state.z | known_on.state.w) == 0 && row_unknown_odd == UnknownTarget) {
      result.state.z = ~0u;
      result.state.w = ~0u;
    }
  }

  result &= unknown & bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::semivulnerable() const {
  return semivulnerable_like<4>();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::quasivulnerable() const {
  return semivulnerable_like<5>();
}

template <unsigned N, unsigned W>
_DI_ int ThreeBoardD2<N, W>::mirror_y(int y) {
  return static_cast<int>(FULL_N) - 1 - y;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::eliminate_pair_steps(cuda::std::pair<int, int> p,
                                                          cuda::std::pair<int, int> q,
                                                          int step_x,
                                                          int step_y) const {
  BitBoard<W> result;

  if constexpr (W == 32) {
    const int row = static_cast<int>(threadIdx.x & 31);
    if (row >= static_cast<int>(N)) {
      return result;
    }
    if (p.second == row || q.second == row) {
      return result;
    }

    const int diff = row - p.second;
    if (diff % step_y != 0) {
      return result;
    }

    const int k = diff / step_y;
    const int col = p.first + step_x * k;
    if (col < 0 || col >= static_cast<int>(FULL_N)) {
      return result;
    }

    result.state |= (1u << static_cast<unsigned>(col));
  } else {
    const int lane = static_cast<int>(threadIdx.x & 31);

    auto process_row = [&](int row, uint32_t &out_lo, uint32_t &out_hi) {
      if (row >= static_cast<int>(N)) {
        return;
      }
      if (p.second == row || q.second == row) {
        return;
      }

      const int diff = row - p.second;
      if (diff % step_y != 0) {
        return;
      }

      const int k = diff / step_y;
      const int col = p.first + step_x * k;
      if (col < 0 || col >= static_cast<int>(FULL_N)) {
        return;
      }

      if (col < 32) {
        out_lo |= (1u << static_cast<unsigned>(col));
      } else {
        out_hi |= (1u << static_cast<unsigned>(col - 32));
      }
    };

    process_row(2 * lane, result.state.x, result.state.y);
    process_row(2 * lane + 1, result.state.z, result.state.w);
  }

  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::eliminate_pair(cuda::std::pair<int, int> p,
                                                    cuda::std::pair<int, int> q) const {
  BitBoard<W> result;

  if (p == q) {
    return result;
  }

  const int dx = q.first - p.first;
  const int dy = q.second - p.second;

  if (dx == 0 || dy == 0) {
    return result;
  }

  const unsigned abs_dx = static_cast<unsigned>(dx < 0 ? -dx : dx);
  const unsigned abs_dy = static_cast<unsigned>(dy < 0 ? -dy : dy);

  int step_x = (dx < 0 ? -1 : 1) * static_cast<int>(div_gcd_table[abs_dx][abs_dy]);
  int step_y = (dy < 0 ? -1 : 1) * static_cast<int>(div_gcd_table[abs_dy][abs_dx]);

  if (step_y < 0) {
    step_y = -step_y;
    step_x = -step_x;
  }

  return eliminate_pair_steps(p, q, step_x, step_y);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                                    cuda::std::pair<unsigned, unsigned> q) const {
  constexpr unsigned cell_count = FULL_N * N;
  const unsigned p_idx = p.second * FULL_N + p.first;
  const unsigned q_idx = q.second * FULL_N + q.first;
  const size_t base = (static_cast<size_t>(p_idx) * cell_count + q_idx) * LINE_ROWS;
  const unsigned lane = threadIdx.x & 31;

  if constexpr (W == 32) {
    const uint32_t *__restrict__ table = g_d2_line_table_32;
    const uint32_t row = (lane < LINE_ROWS) ? __ldg(table + base + lane) : 0u;
    return BitBoard<32>(row);
  } else {
    const ulonglong2 *__restrict__ table = g_d2_line_table_64;
    BitBoard<64> result;
    const ulonglong2 row =
        (lane < LINE_ROWS) ? __ldg(table + base + lane) : make_ulonglong2(0ull, 0ull);
    const uint64_t even_row = row.x;
    const uint64_t odd_row = row.y;

    result.state.x = static_cast<uint32_t>(even_row);
    result.state.y = static_cast<uint32_t>(even_row >> 32);
    result.state.z = static_cast<uint32_t>(odd_row);
    result.state.w = static_cast<uint32_t>(odd_row >> 32);
    return result;
  }
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD2<N, W>::eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                                         cuda::std::pair<unsigned, unsigned> q) const {
  BitBoard<W> result;

  const cuda::std::pair<int, int> p0{p.first, p.second};
  const cuda::std::pair<int, int> p1{p.first, mirror_y(p.second)};
  const cuda::std::pair<int, int> q0{q.first, q.second};
  const cuda::std::pair<int, int> q1{q.first, mirror_y(q.second)};

  result |= eliminate_pair(p0, q0);
  result |= eliminate_pair(p0, q1);
  result |= eliminate_pair(p1, q0);
  result |= eliminate_pair(p1, q1);

  result &= bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardD2<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = known_on;
  cuda::std::pair<int, int> q;
  while (qs.pop_on_if_any(q)) {
    known_off |= eliminate_line(p, q);
    if (!consistent()) {
      return;
    }
  }
  apply_bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardD2<N, W>::eliminate_all_lines(BitBoard<W> seed) {
  cuda::std::pair<int, int> p;
  while (seed.pop_on_if_any(p)) {

    BitBoard<W> qs = known_on & ~seed;
    cuda::std::pair<int, int> q;
    while (qs.pop_on_if_any(q)) {
      known_off |= eliminate_line(p, q);
      if (!consistent()) {
        return;
      }
    }
  }
  apply_bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardD2<N, W>::propagate() {
  ThreeBoardD2<N, W> prev;
  BitBoard<W> done_ons = known_on;

  do {
    do {
      prev = *this;
      *this = force_orthogonal();
      if (!consistent()) {
        return;
      }
    } while (!(*this == prev));

    prev = *this;
    eliminate_all_lines(known_on & ~done_ons);
    if (!consistent()) {
      return;
    }
    done_ons = known_on;
  } while (!(*this == prev));
}
