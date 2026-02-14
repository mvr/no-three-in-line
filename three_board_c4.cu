#pragma once

#include "common.hpp"
#include "binary_count.cuh"
#include "lookup_tables.cuh"
#include "board.cu"
#include "three_board.cu"

#include <cuda/std/array>
#include <cuda/std/utility>

__device__ const uint32_t *__restrict__ g_c4_line_table_32 = nullptr;
__device__ const ulonglong2 *__restrict__ g_c4_line_table_64 = nullptr;

// C4-symmetric board 2N × 2N for N up to 64. Stores only the
// fundamental domain [0, N) × [0, N); the remaining three quadrants are
// implied by fourfold rotational symmetry around (-0.5, -0.5).
template <unsigned N, unsigned W = 32>
struct ThreeBoardC4 {
  static_assert(W == 32 || W == 64, "ThreeBoardC4 supports W=32 or W=64");
  static_assert(N <= 64, "ThreeBoardC4 currently supports N <= 64");
  static_assert((W == 32 && N <= 32) || (W == 64 && N <= 64),
                "Invalid ThreeBoardC4 width/size combination");

  static constexpr unsigned FULL_N = 2 * N;
  static constexpr unsigned FULL_W = (FULL_N <= 32) ? 32 : 64;
  static constexpr unsigned LINE_ROWS =
      LINE_TABLE_FULL_WARP_LOAD ? 32
                                : ((W == 32) ? ((N + 7u) & ~7u) : ((((N + 1u) >> 1) + 7u) & ~7u));
  static_assert(LINE_ROWS <= 32, "C4 line table rows must fit one warp");

  BitBoard<W> known_on;
  BitBoard<W> known_off;

  _DI_ ThreeBoardC4() : known_on{}, known_off{} {}
  _DI_ ThreeBoardC4(BitBoard<W> on, BitBoard<W> off) : known_on{on}, known_off{off} {}

  static _DI_ BitBoard<W> bounds();
  static _DI_ BitBoard<W> relevant_endpoint(cuda::std::pair<unsigned, unsigned> p);
  static void init_line_table_host();
  static void init_tables_host();

  _DI_ bool consistent() const;
  _DI_ unsigned unknown_pop() const;
  _DI_ bool complete() const;
  _DI_ LexStatus canonical_with_forced(ForcedCell &forced) const;
  static _DI_ ThreeBoardC4<N, W> load_from(const board_array_t<W> &on,
                                           const board_array_t<W> &off);
  _DI_ bool operator==(const ThreeBoardC4<N, W> &other) const;

  _DI_ ThreeBoardC4<N, W> force_orthogonal() const;
  _DI_ BitBoard<W> vulnerable() const;
  _DI_ BitBoard<W> semivulnerable() const;
  _DI_ BitBoard<W> quasivulnerable() const;
  _DI_ BitBoard<W> preferred_branch_cells() const;
  template <unsigned UnknownTarget>
  _DI_ BitBoard<W> semivulnerable_like() const;
  _DI_ void apply_bounds();

  // Helpers for reasoning about orbits.
  static _DI_ cuda::std::pair<int, int> rotate90(cuda::std::pair<int, int> p);
  static _DI_ cuda::std::array<cuda::std::pair<int, int>, 4> orbit(cuda::std::pair<int, int> p);
  template <typename Visitor>
  static _DI_ void for_each_orbit_point(cuda::std::pair<int, int> p, Visitor &&visitor);

  _DI_ BitBoard<W> eliminate_line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
  _DI_ BitBoard<W> eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                       cuda::std::pair<unsigned, unsigned> q) const;
  _DI_ BitBoard<W> eliminate_pair(cuda::std::pair<int, int> pi, cuda::std::pair<int, int> qj) const;
  _DI_ BitBoard<W> eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                        cuda::std::pair<int, int> qj,
                                        int step_x, int step_y) const;
  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<W> seed);
  _DI_ void eliminate_all_lines_slow(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines_slow(BitBoard<W> seed);

  _DI_ void propagate();
  _DI_ void propagate_slow();
};

template <unsigned N>
__global__ void init_c4_line_table_kernel_32(uint32_t *__restrict__ table) {
  constexpr unsigned cell_count = N * N;
  constexpr unsigned line_rows = ThreeBoardC4<N, 32>::LINE_ROWS;
  const unsigned pair_idx = blockIdx.x;
  if (pair_idx >= cell_count * cell_count) {
    return;
  }
  const unsigned lane = threadIdx.x & 31;

  const unsigned p_idx = pair_idx / cell_count;
  const unsigned q_idx = pair_idx - p_idx * cell_count;
  const unsigned px = p_idx % N;
  const unsigned py = p_idx / N;
  const unsigned qx = q_idx % N;
  const unsigned qy = q_idx / N;

  ThreeBoardC4<N, 32> board;
  board.known_on.set(px, py);
  board.known_on.set(qx, qy);
  board.eliminate_all_lines_slow({px, py});
  board.eliminate_all_lines_slow({qx, qy});
  board.propagate_slow();

  if (lane < line_rows) {
    table[static_cast<size_t>(pair_idx) * line_rows + lane] = board.known_off.state;
  }
}

template <unsigned N>
__global__ void init_c4_line_table_kernel_64(ulonglong2 *__restrict__ table) {
  constexpr unsigned cell_count = N * N;
  constexpr unsigned line_rows = ThreeBoardC4<N, 64>::LINE_ROWS;
  const unsigned pair_idx = blockIdx.x;
  if (pair_idx >= cell_count * cell_count) {
    return;
  }
  const unsigned lane = threadIdx.x & 31;

  const unsigned p_idx = pair_idx / cell_count;
  const unsigned q_idx = pair_idx - p_idx * cell_count;
  const unsigned px = p_idx % N;
  const unsigned py = p_idx / N;
  const unsigned qx = q_idx % N;
  const unsigned qy = q_idx / N;

  ThreeBoardC4<N, 64> board;
  board.known_on.set(px, py);
  board.known_on.set(qx, qy);
  board.eliminate_all_lines_slow({px, py});
  board.eliminate_all_lines_slow({qx, qy});
  board.propagate_slow();

  if (lane < line_rows) {
    const size_t idx = static_cast<size_t>(pair_idx) * line_rows + lane;
    const uint64_t even = (static_cast<uint64_t>(board.known_off.state.y) << 32) |
                          static_cast<uint64_t>(board.known_off.state.x);
    const uint64_t odd = (static_cast<uint64_t>(board.known_off.state.w) << 32) |
                         static_cast<uint64_t>(board.known_off.state.z);
    table[idx] = make_ulonglong2(even, odd);
  }
}

template <unsigned N, unsigned W>
inline void ThreeBoardC4<N, W>::init_line_table_host() {
  static uint32_t *d_table_32 = nullptr;
  static ulonglong2 *d_table_64 = nullptr;

  constexpr unsigned cell_count = N * N;
  constexpr size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
  constexpr size_t total_rows = total_entries * LINE_ROWS;

  if constexpr (W == 32) {
    if (d_table_32 != nullptr) {
      cudaFree(d_table_32);
      d_table_32 = nullptr;
    }
    cudaMalloc((void **)&d_table_32, total_rows * sizeof(uint32_t));
    init_c4_line_table_kernel_32<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_32);
    cudaGetLastError();
    cudaDeviceSynchronize();
  } else {
    if (d_table_64 != nullptr) {
      cudaFree(d_table_64);
      d_table_64 = nullptr;
    }
    cudaMalloc((void **)&d_table_64, total_rows * sizeof(ulonglong2));
    init_c4_line_table_kernel_64<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_64);
    cudaGetLastError();
    cudaDeviceSynchronize();
  }

  if constexpr (W == 32) {
    cudaMemcpyToSymbol(g_c4_line_table_32, &d_table_32, sizeof(d_table_32));
  } else {
    cudaMemcpyToSymbol(g_c4_line_table_64, &d_table_64, sizeof(d_table_64));
  }
}

template <unsigned N, unsigned W>
inline void ThreeBoardC4<N, W>::init_tables_host() {
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
_DI_ BitBoard<W> ThreeBoardC4<N, W>::bounds() {
  return BitBoard<W>::rect(N, N);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::relevant_endpoint(cuda::std::pair<unsigned, unsigned> p) {
  if constexpr (W == 32) {
    uint64_t fullrow = relevant_endpoint_table[32 - p.second + (threadIdx.x & 31)];
    uint32_t moved_row = fullrow >> (32 - p.first);
    return BitBoard<W>(moved_row);
  } else {
    BitBoard<W> result;

    const unsigned lane = threadIdx.x & 31;
    const unsigned even_row_idx = 64 - p.second + lane * 2;
    const unsigned odd_row_idx = even_row_idx + 1;

    uint64_t full_low_bits_even = relevant_endpoint_table_64[even_row_idx * 2];
    uint64_t full_high_bits_even = relevant_endpoint_table_64[even_row_idx * 2 + 1];
    uint64_t full_low_bits_odd = relevant_endpoint_table_64[odd_row_idx * 2];
    uint64_t full_high_bits_odd = relevant_endpoint_table_64[odd_row_idx * 2 + 1];

    if (p.first < 32) {
      result.state.x = (full_low_bits_even >> (64 - p.first)) | (full_high_bits_even << p.first);
      result.state.y = full_high_bits_even >> (32 - p.first);
      result.state.z = (full_low_bits_odd >> (64 - p.first)) | (full_high_bits_odd << p.first);
      result.state.w = full_high_bits_odd >> (32 - p.first);
    } else {
      result.state.x = full_low_bits_even >> (64 - p.first);
      result.state.y =
          (full_low_bits_even >> (64 - (p.first - 32))) | (full_high_bits_even << (p.first - 32));
      result.state.z = full_low_bits_odd >> (64 - p.first);
      result.state.w =
          (full_low_bits_odd >> (64 - (p.first - 32))) | (full_high_bits_odd << (p.first - 32));
    }

    return result;
  }
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4<N, W>::apply_bounds() {
  const BitBoard<W> b = bounds();
  known_on &= b;
  known_off &= b;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardC4<N, W>::consistent() const {
  return (known_on & known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ unsigned ThreeBoardC4<N, W>::unknown_pop() const {
  return N * N - (known_on | known_off).pop();
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardC4<N, W>::complete() const {
  BitBoard<W> unknown = ~(known_on | known_off) & bounds();
  return unknown.empty();
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoardC4<N, W>::canonical_with_forced(ForcedCell &forced) const {
  BitBoard<W> diag_on = known_on.flip_diagonal();
  BitBoard<W> diag_off = known_off.flip_diagonal();
  BitBoard<W> bds = bounds();
  diag_on &= bds;
  diag_off &= bds;
  ForceCandidate local_force{};
  LexStatus order = compare_with_unknowns_forced<W>(known_on,
                                                     known_off,
                                                     diag_on,
                                                     diag_off,
                                                     bds,
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

template <unsigned N, unsigned W>
_DI_ ThreeBoardC4<N, W> ThreeBoardC4<N, W>::load_from(const board_array_t<W> &on,
                                                      const board_array_t<W> &off) {
  ThreeBoardC4<N, W> board;
  board.known_on = BitBoard<W>::load(on.data());
  board.known_off = BitBoard<W>::load(off.data());
  board.apply_bounds();
  return board;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardC4<N, W>::operator==(const ThreeBoardC4<N, W> &other) const {
  return known_on == other.known_on && known_off == other.known_off;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoardC4<N, W> ThreeBoardC4<N, W>::force_orthogonal() const {
  ThreeBoardC4<N, W> result = *this;

  if constexpr (W == 32) {
    const board_row_t<32> lane_bit = 1u << (threadIdx.x & 31);

    const BinaryCountSaturating<32> row_on_counter = BinaryCountSaturating<32>::horizontal(known_on.state);
    const BinaryCountSaturating<32> col_on_counter = BinaryCountSaturating<32>::vertical(known_on.state);
    const BinaryCountSaturating<32> total_on_counter = row_on_counter + col_on_counter;

    const board_row_t<32> total_on_eq_2 = total_on_counter.bit1 & ~total_on_counter.bit0;
    const board_row_t<32> total_on_gt_2 = total_on_counter.bit1 & total_on_counter.bit0;

    result.known_off.state |= (~known_on.state) & total_on_eq_2;
    result.known_on.state |= total_on_gt_2;
    result.known_off.state |= total_on_gt_2;

    if (total_on_eq_2 & lane_bit) {
      result.known_off.state |= ~known_on.state;
    }
    if (total_on_gt_2 & lane_bit) {
      result.known_on.state = ~0u;
      result.known_off.state = ~0u;
    }

    BitBoard<W> not_known_off = (~known_off) & bounds();

    const BinaryCountSaturating<32> row_not_off_counter =
        BinaryCountSaturating<32>::horizontal(not_known_off.state);
    const BinaryCountSaturating<32> col_not_off_counter =
        BinaryCountSaturating<32>::vertical(not_known_off.state);
    const BinaryCountSaturating<32> total_not_off_counter = row_not_off_counter + col_not_off_counter;

    const board_row_t<32> total_not_off_eq_2 = total_not_off_counter.bit1 & ~total_not_off_counter.bit0;
    const board_row_t<32> total_not_off_lt_2 = ~total_not_off_counter.bit1;

    result.known_on.state |= (~known_off.state) & total_not_off_eq_2;
    result.known_on.state |= total_not_off_lt_2;
    result.known_off.state |= total_not_off_lt_2;

    if (total_not_off_eq_2 & lane_bit) {
      result.known_on.state |= ~known_off.state;
    }
    if (total_not_off_lt_2 & lane_bit) {
      result.known_on.state = ~0u;
      result.known_off.state = ~0u;
    }
  } else {
    const unsigned lane = threadIdx.x & 31;
    constexpr board_row_t<64> row_mask = (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
    const uint32_t row_mask_lo = static_cast<uint32_t>(row_mask);
    const uint32_t row_mask_hi = static_cast<uint32_t>(row_mask >> 32);
    const board_row_t<64> lane_even_bit = board_row_t<64>(1) << (2 * lane);
    const board_row_t<64> lane_odd_bit = lane_even_bit << 1;

    const board_row_t<64> known_on_even =
        ((static_cast<board_row_t<64>>(known_on.state.y) << 32) | known_on.state.x) & row_mask;
    const board_row_t<64> known_on_odd =
        ((static_cast<board_row_t<64>>(known_on.state.w) << 32) | known_on.state.z) & row_mask;

    const BinaryCountSaturating<64> row_on_counter =
        BinaryCountSaturating<64>::horizontal_interleave(known_on_even, known_on_odd);
    const BinaryCountSaturating<64> col_on_counter =
        BinaryCountSaturating<64>::vertical(known_on_even) + BinaryCountSaturating<64>::vertical(known_on_odd);
    const BinaryCountSaturating<64> total_on_counter = row_on_counter + col_on_counter;

    const board_row_t<64> total_on_eq_2 = total_on_counter.template eq_target<2>();
    const board_row_t<64> total_on_gt_2 = total_on_counter.bit1 & total_on_counter.bit0;

    const uint32_t eq2_lo = static_cast<uint32_t>(total_on_eq_2);
    const uint32_t eq2_hi = static_cast<uint32_t>(total_on_eq_2 >> 32);
    const uint32_t gt2_lo = static_cast<uint32_t>(total_on_gt_2);
    const uint32_t gt2_hi = static_cast<uint32_t>(total_on_gt_2 >> 32);

    result.known_off.state.x |= (~known_on.state.x) & eq2_lo;
    result.known_off.state.y |= (~known_on.state.y) & eq2_hi;
    result.known_off.state.z |= (~known_on.state.z) & eq2_lo;
    result.known_off.state.w |= (~known_on.state.w) & eq2_hi;

    result.known_on.state.x |= gt2_lo;
    result.known_on.state.y |= gt2_hi;
    result.known_on.state.z |= gt2_lo;
    result.known_on.state.w |= gt2_hi;
    result.known_off.state.x |= gt2_lo;
    result.known_off.state.y |= gt2_hi;
    result.known_off.state.z |= gt2_lo;
    result.known_off.state.w |= gt2_hi;

    if (total_on_eq_2 & lane_even_bit) {
      result.known_off.state.x |= (~known_on.state.x) & row_mask_lo;
      result.known_off.state.y |= (~known_on.state.y) & row_mask_hi;
    }
    if (total_on_eq_2 & lane_odd_bit) {
      result.known_off.state.z |= (~known_on.state.z) & row_mask_lo;
      result.known_off.state.w |= (~known_on.state.w) & row_mask_hi;
    }
    if (total_on_gt_2 & lane_even_bit) {
      result.known_on.state.x |= row_mask_lo;
      result.known_on.state.y |= row_mask_hi;
      result.known_off.state.x |= row_mask_lo;
      result.known_off.state.y |= row_mask_hi;
    }
    if (total_on_gt_2 & lane_odd_bit) {
      result.known_on.state.z |= row_mask_lo;
      result.known_on.state.w |= row_mask_hi;
      result.known_off.state.z |= row_mask_lo;
      result.known_off.state.w |= row_mask_hi;
    }

    const BitBoard<W> not_known_off = (~known_off) & bounds();
    const board_row_t<64> not_known_off_even =
        ((static_cast<board_row_t<64>>(not_known_off.state.y) << 32) | not_known_off.state.x);
    const board_row_t<64> not_known_off_odd =
        ((static_cast<board_row_t<64>>(not_known_off.state.w) << 32) | not_known_off.state.z);

    const BinaryCountSaturating<64> row_not_off_counter =
        BinaryCountSaturating<64>::horizontal_interleave(not_known_off_even, not_known_off_odd);
    const BinaryCountSaturating<64> col_not_off_counter = BinaryCountSaturating<64>::vertical(not_known_off_even) +
                                                          BinaryCountSaturating<64>::vertical(not_known_off_odd);
    const BinaryCountSaturating<64> total_not_off_counter = row_not_off_counter + col_not_off_counter;

    const board_row_t<64> total_not_off_eq_2 = total_not_off_counter.template eq_target<2>();
    const board_row_t<64> total_not_off_lt_2 = ~total_not_off_counter.bit1;

    const uint32_t not_eq2_lo = static_cast<uint32_t>(total_not_off_eq_2);
    const uint32_t not_eq2_hi = static_cast<uint32_t>(total_not_off_eq_2 >> 32);
    const uint32_t not_lt2_lo = static_cast<uint32_t>(total_not_off_lt_2);
    const uint32_t not_lt2_hi = static_cast<uint32_t>(total_not_off_lt_2 >> 32);

    result.known_on.state.x |= not_known_off.state.x & not_eq2_lo;
    result.known_on.state.y |= not_known_off.state.y & not_eq2_hi;
    result.known_on.state.z |= not_known_off.state.z & not_eq2_lo;
    result.known_on.state.w |= not_known_off.state.w & not_eq2_hi;

    result.known_on.state.x |= not_lt2_lo;
    result.known_on.state.y |= not_lt2_hi;
    result.known_on.state.z |= not_lt2_lo;
    result.known_on.state.w |= not_lt2_hi;
    result.known_off.state.x |= not_lt2_lo;
    result.known_off.state.y |= not_lt2_hi;
    result.known_off.state.z |= not_lt2_lo;
    result.known_off.state.w |= not_lt2_hi;

    if (total_not_off_eq_2 & lane_even_bit) {
      result.known_on.state.x |= not_known_off.state.x;
      result.known_on.state.y |= not_known_off.state.y;
    }
    if (total_not_off_eq_2 & lane_odd_bit) {
      result.known_on.state.z |= not_known_off.state.z;
      result.known_on.state.w |= not_known_off.state.w;
    }
    if (total_not_off_lt_2 & lane_even_bit) {
      result.known_on.state.x |= row_mask_lo;
      result.known_on.state.y |= row_mask_hi;
      result.known_off.state.x |= row_mask_lo;
      result.known_off.state.y |= row_mask_hi;
    }
    if (total_not_off_lt_2 & lane_odd_bit) {
      result.known_on.state.z |= row_mask_lo;
      result.known_on.state.w |= row_mask_hi;
      result.known_off.state.z |= row_mask_lo;
      result.known_off.state.w |= row_mask_hi;
    }
  }

  result.apply_bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::vulnerable() const {
  BitBoard<W> result;
  BitBoard<W> unknown = (~known_on & ~known_off) & bounds();

  if constexpr (W == 32) {
    const BinaryCountSaturating<32> row_on_counter = BinaryCountSaturating<32>::horizontal(known_on.state);
    const BinaryCountSaturating<32> col_on_counter = BinaryCountSaturating<32>::vertical(known_on.state);
    const BinaryCountSaturating<32> total_on_counter = row_on_counter + col_on_counter;

    const BinaryCountSaturating3<32> row_unknown_counter = BinaryCountSaturating3<32>::horizontal(unknown.state);
    const BinaryCountSaturating3<32> col_unknown_counter = BinaryCountSaturating3<32>::vertical(unknown.state);
    const BinaryCountSaturating3<32> total_unknown_counter = row_unknown_counter + col_unknown_counter;

    const board_row_t<32> total_on_eq_0 = total_on_counter.template eq_target<0>();
    const board_row_t<32> total_on_eq_1 = total_on_counter.template eq_target<1>();
    const board_row_t<32> total_unknown_eq_2 = total_unknown_counter.template eq_target<2>();
    const board_row_t<32> total_unknown_eq_3 = total_unknown_counter.template eq_target<3>();

    const board_row_t<32> vulnerable_rows =
        (total_on_eq_1 & total_unknown_eq_2) | (total_on_eq_0 & total_unknown_eq_3);

    const board_row_t<32> lane_bit = 1u << (threadIdx.x & 31);
    if (vulnerable_rows & lane_bit) {
      result.state = ~0u;
    }
    result.state |= vulnerable_rows;
  } else {
    const unsigned lane = threadIdx.x & 31;
    constexpr board_row_t<64> row_mask = (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
    const uint32_t row_mask_lo = static_cast<uint32_t>(row_mask);
    const uint32_t row_mask_hi = static_cast<uint32_t>(row_mask >> 32);
    const board_row_t<64> lane_even_bit = board_row_t<64>(1) << (2 * lane);
    const board_row_t<64> lane_odd_bit = lane_even_bit << 1;

    const board_row_t<64> known_on_even =
        ((static_cast<board_row_t<64>>(known_on.state.y) << 32) | known_on.state.x) & row_mask;
    const board_row_t<64> known_on_odd =
        ((static_cast<board_row_t<64>>(known_on.state.w) << 32) | known_on.state.z) & row_mask;
    const board_row_t<64> unknown_even =
        ((static_cast<board_row_t<64>>(unknown.state.y) << 32) | unknown.state.x) & row_mask;
    const board_row_t<64> unknown_odd =
        ((static_cast<board_row_t<64>>(unknown.state.w) << 32) | unknown.state.z) & row_mask;

    const BinaryCountSaturating<64> row_on_counter =
        BinaryCountSaturating<64>::horizontal_interleave(known_on_even, known_on_odd);
    const BinaryCountSaturating<64> col_on_counter =
        BinaryCountSaturating<64>::vertical(known_on_even) + BinaryCountSaturating<64>::vertical(known_on_odd);
    const BinaryCountSaturating<64> total_on_counter = row_on_counter + col_on_counter;

    const BinaryCountSaturating3<64> row_unknown_counter =
        BinaryCountSaturating3<64>::horizontal_interleave(unknown_even, unknown_odd);
    const BinaryCountSaturating3<64> col_unknown_counter =
        BinaryCountSaturating3<64>::vertical(unknown_even) + BinaryCountSaturating3<64>::vertical(unknown_odd);
    const BinaryCountSaturating3<64> total_unknown_counter = row_unknown_counter + col_unknown_counter;

    const board_row_t<64> total_on_eq_0 = total_on_counter.template eq_target<0>();
    const board_row_t<64> total_on_eq_1 = total_on_counter.template eq_target<1>();
    const board_row_t<64> total_unknown_eq_2 = total_unknown_counter.template eq_target<2>();
    const board_row_t<64> total_unknown_eq_3 = total_unknown_counter.template eq_target<3>();
    const board_row_t<64> vulnerable_rows = (total_on_eq_1 & total_unknown_eq_2) |
                                            (total_on_eq_0 & total_unknown_eq_3);

    const uint32_t vuln_lo = static_cast<uint32_t>(vulnerable_rows);
    const uint32_t vuln_hi = static_cast<uint32_t>(vulnerable_rows >> 32);
    result.state.x |= vuln_lo;
    result.state.y |= vuln_hi;
    result.state.z |= vuln_lo;
    result.state.w |= vuln_hi;

    if (vulnerable_rows & lane_even_bit) {
      result.state.x |= row_mask_lo;
      result.state.y |= row_mask_hi;
    }
    if (vulnerable_rows & lane_odd_bit) {
      result.state.z |= row_mask_lo;
      result.state.w |= row_mask_hi;
    }
  }

  result &= unknown & bounds();
  return result;
}

template <unsigned N, unsigned W>
template <unsigned UnknownTarget>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::semivulnerable_like() const {
  static_assert(UnknownTarget < 8, "semivulnerable_like expects a target < 8");
  BitBoard<W> result;

  BitBoard<W> unknown = (~known_on & ~known_off) & bounds();
  if constexpr (W == 32) {
    const BinaryCountSaturating3<32> row_unknown_counter = BinaryCountSaturating3<32>::horizontal(unknown.state);
    const BinaryCountSaturating3<32> col_unknown_counter = BinaryCountSaturating3<32>::vertical(unknown.state);
    const BinaryCountSaturating3<32> total_unknown_counter = row_unknown_counter + col_unknown_counter;

    const uint32_t col_on_any = __reduce_or_sync(0xffffffff, known_on.state);
    const board_row_t<32> row_on_eq_0 = (known_on.state == 0) ? ~0u : 0u;
    const board_row_t<32> total_on_eq_0 = row_on_eq_0 & ~col_on_any;
    const board_row_t<32> total_unknown_eq = total_unknown_counter.template eq_target<UnknownTarget>();

    const board_row_t<32> semivuln_rows = total_on_eq_0 & total_unknown_eq;

    const board_row_t<32> lane_bit = 1u << (threadIdx.x & 31);
    if (semivuln_rows & lane_bit) {
      result.state = ~0u;
    }
    result.state |= semivuln_rows;
  } else {
    const unsigned lane = threadIdx.x & 31;
    constexpr board_row_t<64> row_mask = (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
    const uint32_t row_mask_lo = static_cast<uint32_t>(row_mask);
    const uint32_t row_mask_hi = static_cast<uint32_t>(row_mask >> 32);
    const board_row_t<64> lane_even_bit = board_row_t<64>(1) << (2 * lane);
    const board_row_t<64> lane_odd_bit = lane_even_bit << 1;

    const board_row_t<64> known_on_even =
        ((static_cast<board_row_t<64>>(known_on.state.y) << 32) | known_on.state.x) & row_mask;
    const board_row_t<64> known_on_odd =
        ((static_cast<board_row_t<64>>(known_on.state.w) << 32) | known_on.state.z) & row_mask;
    const board_row_t<64> unknown_even =
        ((static_cast<board_row_t<64>>(unknown.state.y) << 32) | unknown.state.x) & row_mask;
    const board_row_t<64> unknown_odd =
        ((static_cast<board_row_t<64>>(unknown.state.w) << 32) | unknown.state.z) & row_mask;

    const BinaryCountSaturating3<64> row_unknown_counter =
        BinaryCountSaturating3<64>::horizontal_interleave(unknown_even, unknown_odd);
    const BinaryCountSaturating3<64> col_unknown_counter =
        BinaryCountSaturating3<64>::vertical(unknown_even) + BinaryCountSaturating3<64>::vertical(unknown_odd);
    const BinaryCountSaturating3<64> total_unknown_counter = row_unknown_counter + col_unknown_counter;

    const board_row_t<64> row_on_eq_0 =
        ((((known_on.state.x | known_on.state.y) == 0) ? lane_even_bit : board_row_t<64>(0)) |
         (((known_on.state.z | known_on.state.w) == 0) ? lane_odd_bit : board_row_t<64>(0))) &
        row_mask;
    const uint32_t col_on_any_lo = __reduce_or_sync(0xffffffff, known_on.state.x | known_on.state.z);
    const uint32_t col_on_any_hi = __reduce_or_sync(0xffffffff, known_on.state.y | known_on.state.w);
    const board_row_t<64> col_on_eq_0 =
        (((static_cast<board_row_t<64>>(~col_on_any_hi) << 32) | ~col_on_any_lo) & row_mask);
    const board_row_t<64> total_on_eq_0 = row_on_eq_0 & col_on_eq_0;
    const board_row_t<64> total_unknown_eq = total_unknown_counter.template eq_target<UnknownTarget>();
    const board_row_t<64> mask = total_on_eq_0 & total_unknown_eq;

    const uint32_t mask_lo = static_cast<uint32_t>(mask);
    const uint32_t mask_hi = static_cast<uint32_t>(mask >> 32);
    result.state.x |= mask_lo;
    result.state.y |= mask_hi;
    result.state.z |= mask_lo;
    result.state.w |= mask_hi;

    if (mask & lane_even_bit) {
      result.state.x |= row_mask_lo;
      result.state.y |= row_mask_hi;
    }
    if (mask & lane_odd_bit) {
      result.state.z |= row_mask_lo;
      result.state.w |= row_mask_hi;
    }
  }

  result &= unknown & bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::semivulnerable() const {
  return semivulnerable_like<4>();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::quasivulnerable() const {
  return semivulnerable_like<5>();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::preferred_branch_cells() const {
  BitBoard<W> cells = vulnerable();
  if (!cells.empty()) {
    return cells;
  }
  cells = semivulnerable();
  if (!cells.empty()) {
    return cells;
  }
  cells = quasivulnerable();
  if (!cells.empty()) {
    return cells;
  }
  return BitBoard<W>{};
}

// --- Orbit helpers ---------------------------------------------------------

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<int, int> ThreeBoardC4<N, W>::rotate90(cuda::std::pair<int, int> p) {
  return {-p.second - 1, p.first};
}

template <unsigned N, unsigned W>
_DI_ cuda::std::array<cuda::std::pair<int, int>, 4> ThreeBoardC4<N, W>::orbit(cuda::std::pair<int, int> p) {
  cuda::std::array<cuda::std::pair<int, int>, 4> result{};
  result[0] = p;
  for (int r = 1; r < 4; ++r) {
    result[r] = rotate90(result[r - 1]);
  }
  return result;
}

template <unsigned N, unsigned W>
template <typename Visitor>
_DI_ void ThreeBoardC4<N, W>::for_each_orbit_point(cuda::std::pair<int, int> p, Visitor &&visitor) {
  const auto pts = orbit(p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    visitor(pts[r]);
  }
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::eliminate_pair(cuda::std::pair<int, int> pi,
                                                  cuda::std::pair<int, int> qj) const {
  BitBoard<W> result;

  if (pi == qj)
    return result;

  int dx = qj.first - pi.first;
  int dy = qj.second - pi.second;
  if (dx == 0 || dy == 0)
    return result;

  int abs_dx = dx < 0 ? -dx : dx;
  int abs_dy = dy < 0 ? -dy : dy;

  const unsigned step_x_mag = div_gcd_table[abs_dx][abs_dy];
  const unsigned step_y_mag = div_gcd_table[abs_dy][abs_dx];
  int step_x = (dx < 0 ? -1 : 1) * static_cast<int>(step_x_mag);
  int step_y = (dy < 0 ? -1 : 1) * static_cast<int>(step_y_mag);
  if (step_y < 0) {
    step_y = -step_y;
    step_x = -step_x;
  }
  return eliminate_pair_steps(pi, qj, step_x, step_y);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                                        cuda::std::pair<int, int> qj,
                                                        int step_x, int step_y) const {
  BitBoard<W> result;

  auto process_row = [&](int row, auto &&set_col) {
    if (row < 0 || row >= static_cast<int>(N))
      return;
    if (pi.second == row || qj.second == row)
      return;

    int diff = row - pi.second;
    if (diff % step_y != 0)
      return;

    int k = diff / step_y;
    int col = pi.first + step_x * k;
    if (col < 0 || col >= static_cast<int>(N))
      return;

    set_col(static_cast<unsigned>(col));
  };

  if constexpr (W == 32) {
    int row = static_cast<int>(threadIdx.x & 31);
    process_row(row, [&](unsigned col) { result.state |= board_row_t<32>(1) << col; });
  } else {
    int lane = static_cast<int>(threadIdx.x & 31);
    process_row(2 * lane, [&](unsigned col) {
      if (col < 32) {
        result.state.x |= 1u << col;
      } else {
        result.state.y |= 1u << (col - 32);
      }
    });
    process_row(2 * lane + 1, [&](unsigned col) {
      if (col < 32) {
        result.state.z |= 1u << col;
      } else {
        result.state.w |= 1u << (col - 32);
      }
    });
  }
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4<N, W>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                                  cuda::std::pair<unsigned, unsigned> q) {
  constexpr unsigned cell_count = N * N;
  const unsigned p_idx = p.second * N + p.first;
  const unsigned q_idx = q.second * N + q.first;
  const size_t base = (static_cast<size_t>(p_idx) * cell_count + q_idx) * LINE_ROWS;
  const unsigned lane = threadIdx.x & 31;

  if constexpr (W == 32) {
    const uint32_t *__restrict__ table = g_c4_line_table_32;
    const uint32_t row = (lane < LINE_ROWS) ? __ldg(table + base + lane) : 0u;
    return BitBoard<32>(row);
  } else {
    const ulonglong2 *__restrict__ table = g_c4_line_table_64;
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
_DI_ BitBoard<W> ThreeBoardC4<N, W>::eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                                       cuda::std::pair<unsigned, unsigned> q) const {
  BitBoard<W> result;

  const auto orbit_p = orbit(p);
  const auto orbit_q = orbit(q);

  auto process_family = [&](int q_offset) {
    const auto &p_base = orbit_p[0];
    const auto &q_base = orbit_q[q_offset];

    int dx = q_base.first - p_base.first;
    int dy = q_base.second - p_base.second;

    if (dx == 0 || dy == 0)
      return;

    const unsigned abs_dx = std::abs(dx);
    const unsigned abs_dy = std::abs(dy);

    int current_step_x = (dx < 0 ? -1 : 1) * static_cast<int>(div_gcd_table[abs_dx][abs_dy]);
    int current_step_y = (dy < 0 ? -1 : 1) * static_cast<int>(div_gcd_table[abs_dy][abs_dx]);

    if (current_step_y < 0) {
      current_step_y = -current_step_y;
      current_step_x = -current_step_x;
    }

    for (int r = 0; r < 4; ++r) {
      const auto &pi = orbit_p[r & 3];
      const auto &qj = orbit_q[(q_offset + r) & 3];

      result |= eliminate_pair_steps(pi, qj, current_step_x, current_step_y);

      int next_step_x = -current_step_y;
      int next_step_y = current_step_x;

      // Normalise so that the y-step is always positive
      if (next_step_y < 0) {
        next_step_y = -next_step_y;
        next_step_x = -next_step_x;
      }

      current_step_x = next_step_x;
      current_step_y = next_step_y;
    }
  };

  process_family(0);
  process_family(1);
  process_family(2);
  process_family(3);

  result &= bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
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
_DI_ void ThreeBoardC4<N, W>::eliminate_all_lines_slow(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = known_on;
  cuda::std::pair<int, int> q;
  while (qs.pop_on_if_any(q)) {
    known_off |= eliminate_line_slow(p, q);
    if (!consistent())
      return;
  }
  apply_bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4<N, W>::eliminate_all_lines(BitBoard<W> ps) {
  cuda::std::pair<int, int> p;
  while (ps.pop_on_if_any(p)) {

    BitBoard<W> qs = known_on & ~ps;
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
_DI_ void ThreeBoardC4<N, W>::eliminate_all_lines_slow(BitBoard<W> ps) {
  cuda::std::pair<int, int> p;
  while (ps.pop_on_if_any(p)) {

    BitBoard<W> qs = known_on & ~ps;
    cuda::std::pair<int, int> q;
    while (qs.pop_on_if_any(q)) {
      known_off |= eliminate_line_slow(p, q);
      if (!consistent())
        return;
    }
  }
  apply_bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4<N, W>::propagate() {
  ThreeBoardC4<N, W> prev;
  BitBoard<W> done_ons = known_on;

  do {
    do {
      prev = *this;
      *this = force_orthogonal();
      if (!consistent())
        return;
    } while (!(*this == prev));

    prev = *this;
    eliminate_all_lines(known_on & ~done_ons);
    if (!consistent())
      return;
    done_ons = known_on;
  } while (!(*this == prev));
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4<N, W>::propagate_slow() {
  ThreeBoardC4<N, W> prev;
  BitBoard<W> done_ons = known_on;

  do {
    do {
      prev = *this;
      *this = force_orthogonal();
      if (!consistent())
        return;
    } while (!(*this == prev));

    prev = *this;
    eliminate_all_lines_slow(known_on & ~done_ons);
    if (!consistent())
      return;
    done_ons = known_on;
  } while (!(*this == prev));
}
