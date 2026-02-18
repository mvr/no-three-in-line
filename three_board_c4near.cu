#pragma once

#include "board.cu"
#include "common.hpp"
#include "three_board.cu"

#include <cuda/std/array>
#include <cuda/std/utility>

__device__ const uint32_t *__restrict__ g_c4near_line_table_32 = nullptr;
__device__ const ulonglong2 *__restrict__ g_c4near_line_table_64 = nullptr;

// C4-near board for odd full size (2N-1)x(2N-1).
// Stored domain is x in [0, N), y in [1, N), represented as local:
//   lx = x in [0, N), ly = y-1 in [0, N-1)
// Active long diagonal (x=y, y>0) may contain exactly one ON pair.
// Opposite long diagonal (x=-y) is forced OFF.
template <unsigned N, unsigned W = 32>
struct ThreeBoardC4Near {
  static_assert(W == 32 || W == 64, "ThreeBoardC4Near supports W=32 or W=64");
  static_assert(N >= 2 && N <= 64, "ThreeBoardC4Near requires 2 <= N <= 64");
  static_assert((W == 32 && N <= 32) || (W == 64 && N <= 64),
                "Invalid ThreeBoardC4Near width/size combination");

  static constexpr unsigned FULL_N = 2 * N - 1;
  static constexpr unsigned FULL_W = (FULL_N <= 32) ? 32 : 64;
  static constexpr unsigned STORE_H = N - 1;
  static constexpr unsigned STORE_W = N;
  static constexpr unsigned LINE_ROWS =
      LINE_TABLE_FULL_WARP_LOAD ? 32
                                : ((W == 32) ? ((STORE_H + 7u) & ~7u)
                                             : ((((STORE_H + 1u) >> 1) + 7u) & ~7u));
  static_assert(LINE_ROWS <= 32, "C4Near line table rows must fit one warp");

  BitBoard<W> known_on;
  BitBoard<W> known_off;

  _DI_ ThreeBoardC4Near() : known_on{}, known_off{} {}
  _DI_ ThreeBoardC4Near(BitBoard<W> on, BitBoard<W> off) : known_on{on}, known_off{off} {}

  static _DI_ BitBoard<W> bounds();
  static _DI_ BitBoard<W> active_diagonal();
  template <typename CounterT>
  static _DI_ CounterT orthogonal_counts_excluding_active(BitBoard<W> board,
                                                          BitBoard<W> active_points);
  static _DI_ BinaryCountSaturating<W> orthogonal_counts(BitBoard<W> board,
                                                         BitBoard<W> active_points,
                                                         board_row_t<W> diagonal_mask);
  static _DI_ BinaryCountSaturating3<W> orthogonal_counts3(BitBoard<W> board,
                                                           BitBoard<W> active_points,
                                                           board_row_t<W> diagonal_mask);
  static _DI_ BitBoard<W> canonical_reflect(BitBoard<W> board);
  static _DI_ BitBoard<W> relevant_endpoint(cuda::std::pair<unsigned, unsigned>);
  static void init_line_table_host();
  static void init_tables_host();

  _DI_ bool consistent() const;
  _DI_ unsigned unknown_pop() const;
  _DI_ bool complete() const;
  _DI_ LexStatus canonical_with_forced(ForcedCell &forced) const;
  static _DI_ ThreeBoardC4Near<N, W> load_from(const board_array_t<W> &on,
                                               const board_array_t<W> &off);
  _DI_ bool operator==(const ThreeBoardC4Near<N, W> &other) const;

  _DI_ ThreeBoardC4Near<N, W> force_orthogonal() const;
  _DI_ BitBoard<W> vulnerable() const;
  _DI_ BitBoard<W> preferred_branch_cells() const;
  template <unsigned UnknownTarget>
  _DI_ BitBoard<W> semivulnerable_like() const;
  _DI_ void apply_bounds();

  static _DI_ cuda::std::pair<int, int> local_to_full(cuda::std::pair<unsigned, unsigned> p);
  static _DI_ cuda::std::pair<int, int> rotate90(cuda::std::pair<int, int> p);
  static _DI_ bool is_active_diagonal(int fx, int fy);

  _DI_ BitBoard<W> eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                  cuda::std::pair<unsigned, unsigned> q) const;
  _DI_ BitBoard<W> eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                       cuda::std::pair<unsigned, unsigned> q) const;
  _DI_ BitBoard<W> eliminate_pair(cuda::std::pair<int, int> pi,
                                  cuda::std::pair<int, int> qj) const;
  _DI_ BitBoard<W> eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                        cuda::std::pair<int, int> qj,
                                        int step_x,
                                        int step_y) const;

  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<W> seed);
  _DI_ void eliminate_all_lines_slow(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines_slow(BitBoard<W> seed);

  _DI_ void propagate();
  _DI_ void propagate_slow();
};

template <unsigned N>
__global__ void init_c4near_line_table_kernel_32(uint32_t *__restrict__ table) {
  constexpr unsigned cell_count = N * (N - 1);
  constexpr unsigned line_rows = ThreeBoardC4Near<N, 32>::LINE_ROWS;
  const unsigned pair_idx = blockIdx.x;
  if (pair_idx >= cell_count * cell_count) {
    return;
  }

  const unsigned p_idx = pair_idx / cell_count;
  const unsigned q_idx = pair_idx - p_idx * cell_count;
  const unsigned px = p_idx % N;
  const unsigned py = p_idx / N;
  const unsigned qx = q_idx % N;
  const unsigned qy = q_idx / N;

  ThreeBoardC4Near<N, 32> line_only;
  BitBoard<32> line_mask = line_only.eliminate_line_slow({px, py}, {qx, qy});

  ThreeBoardC4Near<N, 32> board;
  board.known_on.set({px, py});
  board.known_on.set({qx, qy});
  board.eliminate_all_lines_slow({px, py});
  board.eliminate_all_lines_slow({qx, qy});
  board.propagate_slow();
  BitBoard<32> mask = line_mask | board.known_off;

  const unsigned lane = threadIdx.x & 31;
  if (lane < line_rows) {
    table[static_cast<size_t>(pair_idx) * line_rows + lane] = mask.state;
  }
}

template <unsigned N>
__global__ void init_c4near_line_table_kernel_64(ulonglong2 *__restrict__ table) {
  constexpr unsigned cell_count = N * (N - 1);
  constexpr unsigned line_rows = ThreeBoardC4Near<N, 64>::LINE_ROWS;
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

  ThreeBoardC4Near<N, 64> line_only;
  BitBoard<64> line_mask = line_only.eliminate_line_slow({px, py}, {qx, qy});

  ThreeBoardC4Near<N, 64> board;
  board.known_on.set({px, py});
  board.known_on.set({qx, qy});
  board.eliminate_all_lines_slow({px, py});
  board.eliminate_all_lines_slow({qx, qy});
  board.propagate_slow();
  BitBoard<64> mask = line_mask | board.known_off;

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
inline void ThreeBoardC4Near<N, W>::init_line_table_host() {
  static uint32_t *d_table_32 = nullptr;
  static ulonglong2 *d_table_64 = nullptr;

  constexpr unsigned cell_count = N * (N - 1);
  constexpr size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
  constexpr size_t total_rows = total_entries * LINE_ROWS;

  if constexpr (W == 32) {
    if (d_table_32 != nullptr) {
      cudaFree(d_table_32);
      d_table_32 = nullptr;
    }
    cudaMalloc((void **)&d_table_32, total_rows * sizeof(uint32_t));
    init_c4near_line_table_kernel_32<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_32);
    cudaGetLastError();
    cudaDeviceSynchronize();
  } else {
    if (d_table_64 != nullptr) {
      cudaFree(d_table_64);
      d_table_64 = nullptr;
    }
    cudaMalloc((void **)&d_table_64, total_rows * sizeof(ulonglong2));
    init_c4near_line_table_kernel_64<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_64);
    cudaGetLastError();
    cudaDeviceSynchronize();
  }

  if constexpr (W == 32) {
    cudaMemcpyToSymbol(g_c4near_line_table_32, &d_table_32, sizeof(d_table_32));
  } else {
    cudaMemcpyToSymbol(g_c4near_line_table_64, &d_table_64, sizeof(d_table_64));
  }
}

template <unsigned N, unsigned W>
inline void ThreeBoardC4Near<N, W>::init_tables_host() {
  init_lookup_tables_host();
  init_line_table_host();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::bounds() {
  return BitBoard<W>::rect(STORE_W, STORE_H);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::active_diagonal() {
  const unsigned lane = threadIdx.x & 31;
  if constexpr (W == 32) {
    board_row_t<32> row_mask = 0u;
    if (lane < STORE_H) {
      row_mask = board_row_t<32>(1) << (lane + 1);
    }
    return BitBoard<32>(row_mask);
  } else {
    BitBoard<64> result;
    const unsigned row_even = 2 * lane;
    const unsigned row_odd = row_even + 1;

    if (row_even < STORE_H) {
      const unsigned x = row_even + 1;
      const uint32_t bit = 1u << (x & 31u);
      if (x < 32) {
        result.state.x = bit;
      } else {
        result.state.y = bit;
      }
    }
    if (row_odd < STORE_H) {
      const unsigned x = row_odd + 1;
      const uint32_t bit = 1u << (x & 31u);
      if (x < 32) {
        result.state.z = bit;
      } else {
        result.state.w = bit;
      }
    }
    return result;
  }
}

template <unsigned N, unsigned W>
template <typename CounterT>
_DI_ CounterT ThreeBoardC4Near<N, W>::orthogonal_counts_excluding_active(
    BitBoard<W> board,
    BitBoard<W> active_points) {
  if constexpr (W == 32) {
    const board_row_t<32> no_active = (board & ~active_points).state;
    return CounterT::horizontal(no_active).lshift(1) + CounterT::vertical(no_active);
  } else {
    constexpr board_row_t<64> row_mask =
        (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
    const BitBoard<64> no_active = board & ~active_points;
    const board_row_t<64> no_active_even =
        ((static_cast<board_row_t<64>>(no_active.state.y) << 32) | no_active.state.x) &
        row_mask;
    const board_row_t<64> no_active_odd =
        ((static_cast<board_row_t<64>>(no_active.state.w) << 32) | no_active.state.z) &
        row_mask;
    return CounterT::horizontal_interleave(no_active_even, no_active_odd).lshift(1) +
           CounterT::vertical(no_active_even) + CounterT::vertical(no_active_odd);
  }
}

template <unsigned N, unsigned W>
_DI_ BinaryCountSaturating<W> ThreeBoardC4Near<N, W>::orthogonal_counts(
    BitBoard<W> board,
    BitBoard<W> active_points,
    board_row_t<W> diagonal_mask) {
  BinaryCountSaturating<W> total =
      orthogonal_counts_excluding_active<BinaryCountSaturating<W>>(board, active_points);
  total += BinaryCountSaturating<W>{diagonal_mask, 0};
  total += BinaryCountSaturating<W>{total.bit0 & board_row_t<W>(1),
                                    total.bit1 & board_row_t<W>(1)};
  return total;
}

template <unsigned N, unsigned W>
_DI_ BinaryCountSaturating3<W> ThreeBoardC4Near<N, W>::orthogonal_counts3(
    BitBoard<W> board,
    BitBoard<W> active_points,
    board_row_t<W> diagonal_mask) {
  BinaryCountSaturating3<W> total =
      orthogonal_counts_excluding_active<BinaryCountSaturating3<W>>(board, active_points);
  total += BinaryCountSaturating3<W>{diagonal_mask, 0, 0};
  total += BinaryCountSaturating3<W>{total.bit0 & board_row_t<W>(1),
                                     total.bit1 & board_row_t<W>(1),
                                     total.bit2 & board_row_t<W>(1)};
  return total;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::relevant_endpoint(cuda::std::pair<unsigned, unsigned>) {
  return bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4Near<N, W>::apply_bounds() {
  const BitBoard<W> b = bounds();
  known_on &= b;
  known_off &= b;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardC4Near<N, W>::consistent() const {
  return (known_on & known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ unsigned ThreeBoardC4Near<N, W>::unknown_pop() const {
  return STORE_W * STORE_H - (known_on | known_off).pop();
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardC4Near<N, W>::complete() const {
  BitBoard<W> unknown = (~known_on & ~known_off) & bounds();
  return unknown.empty();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::canonical_reflect(BitBoard<W> board) {
  BitBoard<W> refl = board.flip_diagonal().rotate_torus(1, -1);

  if constexpr (W == 32) {
    const unsigned lane = threadIdx.x & 31;
    const bool active = lane < STORE_H;

    // After transpose+shift, source x=0 lands on wrapped row -1 (lane 31).
    // Move those bits into local column 0 and clear the wrapped row.
    const board_row_t<32> carry = refl.row(31);
    refl.erase_row(31);
    if (active) {
      const board_row_t<32> bit = board_row_t<32>(1) << (lane + 1);
      refl.state |= ((carry & bit) != 0u) ? 1u : 0u;
    } else {
      refl.state = 0u;
    }
  } else {
    const unsigned lane = threadIdx.x & 31;
    const unsigned row_even = 2 * lane;
    const unsigned row_odd = row_even + 1;
    const board_row_t<64> carry = refl.row(63);
    refl.erase_row(63);

    if (row_even < STORE_H) {
      const board_row_t<64> bit = board_row_t<64>(1) << (row_even + 1);
      if ((carry & bit) != 0) {
        refl.state.x |= 1u;
      }
    }
    if (row_odd < STORE_H) {
      const board_row_t<64> bit = board_row_t<64>(1) << (row_odd + 1);
      if ((carry & bit) != 0) {
        refl.state.z |= 1u;
      }
    }
  }

  refl &= bounds();
  return refl;
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoardC4Near<N, W>::canonical_with_forced(ForcedCell &forced) const {
  BitBoard<W> refl_on = canonical_reflect(known_on);
  BitBoard<W> refl_off = canonical_reflect(known_off);

  const BitBoard<W> bds = bounds();
  ForceCandidate local_force{};
  LexStatus order = compare_with_unknowns_forced<W>(known_on,
                                                     known_off,
                                                     refl_on,
                                                     refl_off,
                                                     bds,
                                                     local_force);

  forced = ForcedCell{};
  if (order == LexStatus::Unknown && local_force.has_force) {
    forced.has_force = true;
    forced.force_on = local_force.force_on;
    auto cell = local_force.cell;
    if (local_force.on_b) {
      if (cell.first == 0) {
        cell = {0, cell.second};
      } else {
        cell = {cell.second + 1, cell.first - 1};
      }
    }
    forced.cell = cell;
  }
  return order;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoardC4Near<N, W> ThreeBoardC4Near<N, W>::load_from(const board_array_t<W> &on,
                                                              const board_array_t<W> &off) {
  ThreeBoardC4Near<N, W> board;
  board.known_on = BitBoard<W>::load(on.data());
  board.known_off = BitBoard<W>::load(off.data());
  board.apply_bounds();
  return board;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardC4Near<N, W>::operator==(const ThreeBoardC4Near<N, W> &other) const {
  return known_on == other.known_on && known_off == other.known_off;
}

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<int, int> ThreeBoardC4Near<N, W>::local_to_full(cuda::std::pair<unsigned, unsigned> p) {
  return {static_cast<int>(p.first), static_cast<int>(p.second) + 1};
}

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<int, int> ThreeBoardC4Near<N, W>::rotate90(cuda::std::pair<int, int> p) {
  return {-p.second, p.first};
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardC4Near<N, W>::is_active_diagonal(int fx, int fy) {
  return fx == fy && fx != 0;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoardC4Near<N, W> ThreeBoardC4Near<N, W>::force_orthogonal() const {
  ThreeBoardC4Near<N, W> result = *this;

  {
    BitBoard<W> active_point = known_on & active_diagonal();
    if constexpr (W == 32) {
      board_row_t<32> diagonal_on_mask = active_point.occupied_columns();
      if (diagonal_on_mask != 0) {
        result.known_off |= active_diagonal() & ~known_on;
      }
      if ((diagonal_on_mask & (diagonal_on_mask - 1)) != 0) {
        result.known_off |= known_on;
      }

      const BinaryCountSaturating<32> total_on_counter =
          orthogonal_counts(known_on, active_point, diagonal_on_mask);

      const unsigned lane = threadIdx.x & 31;
      const board_row_t<32> col_bit =
          (lane < STORE_H) ? (board_row_t<32>(1) << (lane + 1)) : 0u;

      const board_row_t<32> total_on_eq_2 = total_on_counter.bit1 & ~total_on_counter.bit0;
      const board_row_t<32> total_on_gt_2 = total_on_counter.bit1 & total_on_counter.bit0;

      result.known_off.state |= (~known_on.state) & total_on_eq_2;
      result.known_on.state |= total_on_gt_2;
      result.known_off.state |= total_on_gt_2;

      if (total_on_eq_2 & col_bit) {
        result.known_off.state |= ~known_on.state;
      }
      if (total_on_gt_2 & col_bit) {
        result.known_on.state = ~0u;
        result.known_off.state = ~0u;
      }
    } else {
      constexpr board_row_t<64> row_mask =
          (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
      const uint32_t row_mask_lo = static_cast<uint32_t>(row_mask);
      const uint32_t row_mask_hi = static_cast<uint32_t>(row_mask >> 32);
      const unsigned lane = threadIdx.x & 31;
      const unsigned row_even = 2 * lane;
      const unsigned row_odd = row_even + 1;
      const bool has_even = row_even < STORE_H;
      const bool has_odd = row_odd < STORE_H;
      const board_row_t<64> lane_even_bit =
          has_even ? (board_row_t<64>(1) << (row_even + 1)) : board_row_t<64>(0);
      const board_row_t<64> lane_odd_bit =
          has_odd ? (board_row_t<64>(1) << (row_odd + 1)) : board_row_t<64>(0);

      const board_row_t<64> diagonal_on_mask = active_point.occupied_columns();
      if (diagonal_on_mask != 0) {
        result.known_off |= active_diagonal() & ~known_on;
      }
      if ((diagonal_on_mask & (diagonal_on_mask - 1)) != 0) {
        result.known_off |= known_on;
      }

      const BinaryCountSaturating<64> total_on_counter =
          orthogonal_counts(known_on, active_point, diagonal_on_mask);

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

      if ((total_on_eq_2 & lane_even_bit) != 0) {
        result.known_off.state.x |= (~known_on.state.x) & row_mask_lo;
        result.known_off.state.y |= (~known_on.state.y) & row_mask_hi;
      }
      if ((total_on_eq_2 & lane_odd_bit) != 0) {
        result.known_off.state.z |= (~known_on.state.z) & row_mask_lo;
        result.known_off.state.w |= (~known_on.state.w) & row_mask_hi;
      }
      if ((total_on_gt_2 & lane_even_bit) != 0) {
        result.known_on.state.x |= row_mask_lo;
        result.known_on.state.y |= row_mask_hi;
        result.known_off.state.x |= row_mask_lo;
        result.known_off.state.y |= row_mask_hi;
      }
      if ((total_on_gt_2 & lane_odd_bit) != 0) {
        result.known_on.state.z |= row_mask_lo;
        result.known_on.state.w |= row_mask_hi;
        result.known_off.state.z |= row_mask_lo;
        result.known_off.state.w |= row_mask_hi;
      }
    }
  }

  {
    BitBoard<W> not_known_off = (~known_off) & bounds();
    BitBoard<W> active_not_known_off = not_known_off & active_diagonal();
    if constexpr (W == 32) {
      board_row_t<32> diagonal_not_known_off_mask =
          active_not_known_off.occupied_columns();
      if (diagonal_not_known_off_mask == 0) {
        result.known_on.state |= ~0u;
      }
      if ((diagonal_not_known_off_mask & (diagonal_not_known_off_mask - 1)) == 0) {
        result.known_on |= active_not_known_off;
      }

      const BinaryCountSaturating<32> total_not_off_counter =
          orthogonal_counts(not_known_off, active_not_known_off, diagonal_not_known_off_mask);

      const board_row_t<32> total_not_off_eq_2 =
          total_not_off_counter.bit1 & ~total_not_off_counter.bit0;
      const board_row_t<32> total_not_off_lt_2 = ~total_not_off_counter.bit1;

      result.known_on.state |= (~known_off.state) & total_not_off_eq_2;
      result.known_on.state |= total_not_off_lt_2;
      result.known_off.state |= total_not_off_lt_2;

      const unsigned lane = threadIdx.x & 31;
      const board_row_t<32> col_bit =
          (lane < STORE_H) ? (board_row_t<32>(1) << (lane + 1)) : 0u;
      if (total_not_off_eq_2 & col_bit) {
        result.known_on.state |= ~known_off.state;
      }
      if (total_not_off_lt_2 & col_bit) {
        result.known_on.state = ~0u;
        result.known_off.state = ~0u;
      }
    } else {
      constexpr board_row_t<64> row_mask =
          (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
      const uint32_t row_mask_lo = static_cast<uint32_t>(row_mask);
      const uint32_t row_mask_hi = static_cast<uint32_t>(row_mask >> 32);
      const unsigned lane = threadIdx.x & 31;
      const unsigned row_even = 2 * lane;
      const unsigned row_odd = row_even + 1;
      const bool has_even = row_even < STORE_H;
      const bool has_odd = row_odd < STORE_H;
      const board_row_t<64> lane_even_bit =
          has_even ? (board_row_t<64>(1) << (row_even + 1)) : board_row_t<64>(0);
      const board_row_t<64> lane_odd_bit =
          has_odd ? (board_row_t<64>(1) << (row_odd + 1)) : board_row_t<64>(0);

      const board_row_t<64> diagonal_not_known_off_mask =
          active_not_known_off.occupied_columns();
      if (diagonal_not_known_off_mask == 0) {
        result.known_on.state = {~0u, ~0u, ~0u, ~0u};
      }
      if ((diagonal_not_known_off_mask & (diagonal_not_known_off_mask - 1)) == 0) {
        result.known_on |= active_not_known_off;
      }

      const BinaryCountSaturating<64> total_not_off_counter =
          orthogonal_counts(not_known_off, active_not_known_off, diagonal_not_known_off_mask);

      const board_row_t<64> total_not_off_eq_2 = total_not_off_counter.template eq_target<2>();
      const board_row_t<64> total_not_off_lt_2 = ~total_not_off_counter.bit1;
      const uint32_t eq2_lo = static_cast<uint32_t>(total_not_off_eq_2);
      const uint32_t eq2_hi = static_cast<uint32_t>(total_not_off_eq_2 >> 32);
      const uint32_t lt2_lo = static_cast<uint32_t>(total_not_off_lt_2);
      const uint32_t lt2_hi = static_cast<uint32_t>(total_not_off_lt_2 >> 32);

      result.known_on.state.x |= not_known_off.state.x & eq2_lo;
      result.known_on.state.y |= not_known_off.state.y & eq2_hi;
      result.known_on.state.z |= not_known_off.state.z & eq2_lo;
      result.known_on.state.w |= not_known_off.state.w & eq2_hi;

      result.known_on.state.x |= lt2_lo;
      result.known_on.state.y |= lt2_hi;
      result.known_on.state.z |= lt2_lo;
      result.known_on.state.w |= lt2_hi;
      result.known_off.state.x |= lt2_lo;
      result.known_off.state.y |= lt2_hi;
      result.known_off.state.z |= lt2_lo;
      result.known_off.state.w |= lt2_hi;

      if ((total_not_off_eq_2 & lane_even_bit) != 0) {
        result.known_on.state.x |= not_known_off.state.x & row_mask_lo;
        result.known_on.state.y |= not_known_off.state.y & row_mask_hi;
      }
      if ((total_not_off_eq_2 & lane_odd_bit) != 0) {
        result.known_on.state.z |= not_known_off.state.z & row_mask_lo;
        result.known_on.state.w |= not_known_off.state.w & row_mask_hi;
      }
      if ((total_not_off_lt_2 & lane_even_bit) != 0) {
        result.known_on.state.x |= row_mask_lo;
        result.known_on.state.y |= row_mask_hi;
        result.known_off.state.x |= row_mask_lo;
        result.known_off.state.y |= row_mask_hi;
      }
      if ((total_not_off_lt_2 & lane_odd_bit) != 0) {
        result.known_on.state.z |= row_mask_lo;
        result.known_on.state.w |= row_mask_hi;
        result.known_off.state.z |= row_mask_lo;
        result.known_off.state.w |= row_mask_hi;
      }
    }
  }

  result.apply_bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::vulnerable() const {
  if constexpr (W == 32) {
    constexpr board_row_t<32> row_mask =
        (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
    constexpr unsigned pos_count = STORE_H;

    const unsigned lane = threadIdx.x & 31;
    const bool active = lane < pos_count;

    const BitBoard<W> not_known_off = (~known_off) & bounds();
    const BitBoard<W> active_on = known_on & active_diagonal();
    const BitBoard<W> active_not_off = not_known_off & active_diagonal();
    const board_row_t<32> active_on_mask = active_on.occupied_columns();
    const board_row_t<32> active_not_off_mask = active_not_off.occupied_columns();

    const BinaryCountSaturating3<32> total_on_counter =
        orthogonal_counts3(known_on, active_on, active_on_mask);

    const BinaryCountSaturating3<32> total_not_off_counter =
        orthogonal_counts3(not_known_off, active_not_off, active_not_off_mask);

    const board_row_t<32> line_match =
        total_not_off_counter.template eq_target<3>() &
        ~(total_on_counter.bit1 | total_on_counter.bit2);
    const board_row_t<32> col_bit =
        active ? (board_row_t<32>(1) << (lane + 1)) : board_row_t<32>(0);
    const bool row_match = (line_match & col_bit) != 0;
    const board_row_t<32> unknown_row =
        active ? ((~known_on.state & ~known_off.state) & row_mask) : 0u;

    BitBoard<W> result{};
    result.state =
        unknown_row & (line_match | (row_match ? row_mask : board_row_t<32>(0)));
    return result;
  } else {
    constexpr board_row_t<64> row_mask =
        (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
    const unsigned lane = threadIdx.x & 31;
    const unsigned row_even = 2 * lane;
    const unsigned row_odd = row_even + 1;

    const BitBoard<64> not_known_off = (~known_off) & bounds();
    const BitBoard<64> active_on = known_on & active_diagonal();
    const BitBoard<64> active_not_off = not_known_off & active_diagonal();

    const board_row_t<64> active_on_mask = active_on.occupied_columns();
    const board_row_t<64> active_not_off_mask = active_not_off.occupied_columns();

    const BinaryCountSaturating3<64> total_on_counter =
        orthogonal_counts3(known_on, active_on, active_on_mask);

    const BinaryCountSaturating3<64> total_not_off_counter =
        orthogonal_counts3(not_known_off, active_not_off, active_not_off_mask);

    const board_row_t<64> line_match =
        total_not_off_counter.template eq_target<3>() &
        ~(total_on_counter.bit1 | total_on_counter.bit2);

    const BitBoard<64> unknown = (~known_on & ~known_off) & bounds();
    const board_row_t<64> unknown_even =
        (static_cast<board_row_t<64>>(unknown.state.y) << 32) | unknown.state.x;
    const board_row_t<64> unknown_odd =
        (static_cast<board_row_t<64>>(unknown.state.w) << 32) | unknown.state.z;

    board_row_t<64> mask_even = 0;
    if (row_even < STORE_H) {
      mask_even = line_match & row_mask;
      if (((line_match >> (row_even + 1)) & 1u) != 0u) {
        mask_even |= row_mask;
      }
    }

    board_row_t<64> mask_odd = 0;
    if (row_odd < STORE_H) {
      mask_odd = line_match & row_mask;
      if (((line_match >> (row_odd + 1)) & 1u) != 0u) {
        mask_odd |= row_mask;
      }
    }

    const board_row_t<64> out_even = unknown_even & mask_even;
    const board_row_t<64> out_odd = unknown_odd & mask_odd;

    BitBoard<64> result;
    result.state.x = static_cast<uint32_t>(out_even);
    result.state.y = static_cast<uint32_t>(out_even >> 32);
    result.state.z = static_cast<uint32_t>(out_odd);
    result.state.w = static_cast<uint32_t>(out_odd >> 32);
    return result;
  }
}

template <unsigned N, unsigned W>
template <unsigned UnknownTarget>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::semivulnerable_like() const {
  static_assert(UnknownTarget == 4 || UnknownTarget == 5,
                "UnknownTarget must be 4 or 5");
  if constexpr (W == 32) {
    constexpr board_row_t<32> row_mask =
        (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
    constexpr unsigned pos_count = STORE_H;

    const unsigned lane = threadIdx.x & 31;
    const bool active = lane < pos_count;

    const BitBoard<W> not_known_off = (~known_off) & bounds();
    const BitBoard<W> active_on = known_on & active_diagonal();
    const BitBoard<W> active_not_off = not_known_off & active_diagonal();
    const board_row_t<32> active_on_mask = active_on.occupied_columns();
    const board_row_t<32> active_not_off_mask = active_not_off.occupied_columns();

    const BinaryCountSaturating3<32> total_on_counter =
        orthogonal_counts3(known_on, active_on, active_on_mask);

    const BinaryCountSaturating3<32> total_not_off_counter =
        orthogonal_counts3(not_known_off, active_not_off, active_not_off_mask);

    const board_row_t<32> line_match = total_on_counter.template eq_target<0>() &
                                       total_not_off_counter.template eq_target<UnknownTarget>();
    const board_row_t<32> col_bit =
        active ? (board_row_t<32>(1) << (lane + 1)) : board_row_t<32>(0);
    const bool row_match = (line_match & col_bit) != 0;
    const board_row_t<32> unknown_row =
        active ? ((~known_on.state & ~known_off.state) & row_mask) : 0u;

    BitBoard<W> result{};
    result.state =
        unknown_row & (line_match | (row_match ? row_mask : board_row_t<32>(0)));
    return result;
  } else {
    constexpr board_row_t<64> row_mask =
        (N == 64) ? ~board_row_t<64>(0) : ((board_row_t<64>(1) << N) - 1u);
    const unsigned lane = threadIdx.x & 31;
    const unsigned row_even = 2 * lane;
    const unsigned row_odd = row_even + 1;

    const BitBoard<64> not_known_off = (~known_off) & bounds();
    const BitBoard<64> active_on = known_on & active_diagonal();
    const BitBoard<64> active_not_off = not_known_off & active_diagonal();

    const board_row_t<64> active_on_mask = active_on.occupied_columns();
    const board_row_t<64> active_not_off_mask = active_not_off.occupied_columns();

    const BinaryCountSaturating3<64> total_on_counter =
        orthogonal_counts3(known_on, active_on, active_on_mask);

    const BinaryCountSaturating3<64> total_not_off_counter =
        orthogonal_counts3(not_known_off, active_not_off, active_not_off_mask);

    const board_row_t<64> line_match = total_on_counter.template eq_target<0>() &
                                       total_not_off_counter.template eq_target<UnknownTarget>();

    const BitBoard<64> unknown = (~known_on & ~known_off) & bounds();
    const board_row_t<64> unknown_even =
        (static_cast<board_row_t<64>>(unknown.state.y) << 32) | unknown.state.x;
    const board_row_t<64> unknown_odd =
        (static_cast<board_row_t<64>>(unknown.state.w) << 32) | unknown.state.z;

    board_row_t<64> mask_even = 0;
    if (row_even < STORE_H) {
      mask_even = line_match & row_mask;
      if (((line_match >> (row_even + 1)) & 1u) != 0u) {
        mask_even |= row_mask;
      }
    }

    board_row_t<64> mask_odd = 0;
    if (row_odd < STORE_H) {
      mask_odd = line_match & row_mask;
      if (((line_match >> (row_odd + 1)) & 1u) != 0u) {
        mask_odd |= row_mask;
      }
    }

    const board_row_t<64> out_even = unknown_even & mask_even;
    const board_row_t<64> out_odd = unknown_odd & mask_odd;

    BitBoard<64> result;
    result.state.x = static_cast<uint32_t>(out_even);
    result.state.y = static_cast<uint32_t>(out_even >> 32);
    result.state.z = static_cast<uint32_t>(out_odd);
    result.state.w = static_cast<uint32_t>(out_odd >> 32);
    return result;
  }
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::preferred_branch_cells() const {
  BitBoard<W> cells = vulnerable();
  if (!cells.empty()) {
    return cells;
  }

  cells = semivulnerable_like<4>();
  if (!cells.empty()) {
    return cells;
  }

  cells = semivulnerable_like<5>();
  if (!cells.empty()) {
    return cells;
  }
  return BitBoard<W>{};
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                                              cuda::std::pair<int, int> qj,
                                                              int step_x,
                                                              int step_y) const {
  BitBoard<W> result;

  auto process_row = [&](int ly, auto &&set_col) {
    if (ly < 0 || ly >= static_cast<int>(STORE_H)) {
      return;
    }

    const int fy = ly + 1;
    if (pi.second == fy || qj.second == fy) {
      return;
    }

    const int diff = fy - pi.second;
    if (diff % step_y != 0) {
      return;
    }

    const int k = diff / step_y;
    const int fx = pi.first + step_x * k;
    if (fx < 0 || fx >= static_cast<int>(STORE_W)) {
      return;
    }

    set_col(static_cast<unsigned>(fx));
  };

  if constexpr (W == 32) {
    const int ly = static_cast<int>(threadIdx.x & 31);
    process_row(ly, [&](unsigned fx) {
      result.state |= (board_row_t<32>(1) << fx);
    });
  } else {
    const int lane = static_cast<int>(threadIdx.x & 31);
    process_row(2 * lane, [&](unsigned fx) {
      if (fx < 32) {
        result.state.x |= (1u << fx);
      } else {
        result.state.y |= (1u << (fx - 32));
      }
    });
    process_row(2 * lane + 1, [&](unsigned fx) {
      if (fx < 32) {
        result.state.z |= (1u << fx);
      } else {
        result.state.w |= (1u << (fx - 32));
      }
    });
  }
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::eliminate_pair(cuda::std::pair<int, int> pi,
                                                        cuda::std::pair<int, int> qj) const {
  BitBoard<W> result;
  if (pi == qj) {
    return result;
  }

  const int dx = qj.first - pi.first;
  const int dy = qj.second - pi.second;
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

  return eliminate_pair_steps(pi, qj, step_x, step_y);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                                             cuda::std::pair<unsigned, unsigned> q) const {
  BitBoard<W> result;
  const cuda::std::pair<int, int> pf = local_to_full(p);
  const cuda::std::pair<int, int> qf = local_to_full(q);

  cuda::std::array<cuda::std::pair<int, int>, 4> po{};
  cuda::std::array<cuda::std::pair<int, int>, 4> qo{};
  int pn = 0;
  int qn = 0;

  if (is_active_diagonal(pf.first, pf.second)) {
    po[pn++] = pf;
    po[pn++] = {-pf.first, -pf.second};
  } else {
    auto t = pf;
    for (int r = 0; r < 4; ++r) {
      po[pn++] = t;
      t = rotate90(t);
    }
  }

  if (is_active_diagonal(qf.first, qf.second)) {
    qo[qn++] = qf;
    qo[qn++] = {-qf.first, -qf.second};
  } else {
    auto t = qf;
    for (int r = 0; r < 4; ++r) {
      qo[qn++] = t;
      t = rotate90(t);
    }
  }

  for (int i = 0; i < pn; ++i) {
    for (int j = 0; j < qn; ++j) {
      result |= eliminate_pair(po[i], qo[j]);
    }
  }

  result &= bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                                        cuda::std::pair<unsigned, unsigned> q) const {
  constexpr unsigned cell_count = STORE_W * STORE_H;
  const unsigned p_idx = p.second * STORE_W + p.first;
  const unsigned q_idx = q.second * STORE_W + q.first;
  const size_t base = (static_cast<size_t>(p_idx) * cell_count + q_idx) * LINE_ROWS;
  const unsigned lane = threadIdx.x & 31;
  if constexpr (W == 32) {
    const uint32_t *__restrict__ table = g_c4near_line_table_32;
    const uint32_t row = (lane < LINE_ROWS) ? __ldg(table + base + lane) : 0u;
    return BitBoard<32>(row);
  } else {
    const ulonglong2 *__restrict__ table = g_c4near_line_table_64;
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
_DI_ void ThreeBoardC4Near<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
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
_DI_ void ThreeBoardC4Near<N, W>::eliminate_all_lines(BitBoard<W> seed) {
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
_DI_ void ThreeBoardC4Near<N, W>::eliminate_all_lines_slow(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = known_on;
  cuda::std::pair<int, int> q;
  while (qs.pop_on_if_any(q)) {
    known_off |= eliminate_line_slow(p, q);
    if (!consistent()) {
      return;
    }
  }
  apply_bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4Near<N, W>::eliminate_all_lines_slow(BitBoard<W> seed) {
  cuda::std::pair<int, int> p;
  while (seed.pop_on_if_any(p)) {
    BitBoard<W> qs = known_on & ~seed;
    cuda::std::pair<int, int> q;
    while (qs.pop_on_if_any(q)) {
      known_off |= eliminate_line_slow(p, q);
      if (!consistent()) {
        return;
      }
    }
  }
  apply_bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4Near<N, W>::propagate() {
  ThreeBoardC4Near<N, W> prev;
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

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4Near<N, W>::propagate_slow() {
  ThreeBoardC4Near<N, W> prev;
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
    eliminate_all_lines_slow(known_on & ~done_ons);
    if (!consistent()) {
      return;
    }
    done_ons = known_on;
  } while (!(*this == prev));
}
