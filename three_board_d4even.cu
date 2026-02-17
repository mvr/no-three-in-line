#pragma once

#include "board.cu"
#include "binary_count.cuh"
#include "common.hpp"
#include "compare_with_unknowns.cuh"
#include "lookup_tables.cuh"
#include "three_board.cu"

#include <cuda/std/utility>

__device__ const uint32_t *__restrict__ g_d4even_line_table_32 = nullptr;

// D4Even storage contract (W=32 first pass)
// =========================================
//
// Parameterization:
//   - Storage is STORE_H x STORE_W = (N-1) x N, with 2 <= N <= 32.
//   - Full board side length is FULL_N = 2*(N-1).
//
// Full coordinates and symmetry:
//   - Let H = N-1.
//   - Full coordinates are (fx, fy) in [-H, H-1].
//   - D4 generators in this frame:
//       diagonal reflection: (fx, fy) -> (fy, fx)
//       180 rotation:        (fx, fy) -> (-fx-1, -fy-1)
//     (anti-diagonal reflection is their composition).
//
// Stored representatives (includes both diagonals):
//   - Local rows ly in [0, H-1], cols lx in [0, H].
//
//   For each ly:
//     1) Main-side triangle (includes main diagonal):
//          if lx <= ly:
//            (fx, fy) = (lx, ly)
//     2) Anti-side triangle (includes anti diagonal), row-compacted:
//          if lx > ly:
//            rx = lx - (ly + 1)         in [0, H-ly-1]
//            (fx, fy) = (-H + rx, ly)
//
// This gives exactly N*(N-1) orbit representatives for FULL_N x FULL_N.
//
// Notes for orthogonal counting:
//   - The anti-side triangle has a row-dependent offset; alignment is
//     by (ly+1), not ly.
template <unsigned N, unsigned W = 32>
struct ThreeBoardD4Even {
  static_assert(W == 32, "ThreeBoardD4Even currently supports only W=32");
  static_assert(N >= 2 && N <= 32, "ThreeBoardD4Even requires 2 <= N <= 32");

  static constexpr unsigned H = N - 1;
  static constexpr unsigned FULL_N = 2 * H;
  static constexpr unsigned FULL_W = (FULL_N <= 32) ? 32 : 64;
  static constexpr unsigned STORE_H = H;
  static constexpr unsigned STORE_W = N;
  static constexpr unsigned CELL_COUNT = STORE_H * STORE_W;
  static constexpr unsigned LINE_ROWS =
      LINE_TABLE_FULL_WARP_LOAD ? 32 : ((STORE_H + 7u) & ~7u);
  static_assert(LINE_ROWS <= 32, "D4Even line table rows must fit one warp");

  BitBoard<W> known_on;
  BitBoard<W> known_off;

  _DI_ ThreeBoardD4Even() : known_on{}, known_off{} {}
  _DI_ ThreeBoardD4Even(BitBoard<W> on, BitBoard<W> off) : known_on{on}, known_off{off} {}

  static _DI_ constexpr board_row_t<32> row_mask() {
    if constexpr (N == 32) {
      return 0xffffffffu;
    } else {
      return (board_row_t<32>(1u) << N) - 1u;
    }
  }

  static _DI_ constexpr board_row_t<32> family_mask() {
    if constexpr (H == 32) {
      return 0xffffffffu;
    } else {
      return (board_row_t<32>(1u) << H) - 1u;
    }
  }

  static _DI_ constexpr board_row_t<32> main_triangle_mask(unsigned ly) {
    return (ly < H) ? ((board_row_t<32>(1u) << (ly + 1u)) - 1u) : 0u;
  }

  static _DI_ constexpr board_row_t<32> anti_triangle_mask(unsigned ly) {
    return row_mask() & ~main_triangle_mask(ly);
  }

  static _DI_ constexpr unsigned anti_width(unsigned ly) {
    return (ly < H) ? (H - ly) : 0u;
  }

  // Convert between compact anti-side storage bits and a matrix-form row that
  // lives in columns [ly, H-1] and can be transposed with flip_diagonal().
  static _DI_ board_row_t<32> unpack_anti_compact(board_row_t<32> compact, unsigned ly) {
    const unsigned width = anti_width(ly);
    if (width == 0u) {
      return 0u;
    }
    const board_row_t<32> compact_mask = (board_row_t<32>(1u) << width) - 1u;
    const board_row_t<32> compact_bits = compact & compact_mask;
    const board_row_t<32> rev = __brev(compact_bits) >> (32u - width);
    return rev << ly;
  }

  static _DI_ board_row_t<32> pack_anti_compact(board_row_t<32> matrix_row, unsigned ly) {
    const unsigned width = anti_width(ly);
    if (width == 0u) {
      return 0u;
    }
    const board_row_t<32> compact_mask = (board_row_t<32>(1u) << width) - 1u;
    const board_row_t<32> seg = (matrix_row >> ly) & compact_mask;
    return __brev(seg) >> (32u - width);
  }

  static _DI_ board_row_t<32> reverse_low_bits(board_row_t<32> value) {
    return (H == 0) ? 0u : (__brev(value) >> (32u - H));
  }

  static _DI_ BitBoard<W> bounds();
  static void init_line_table_host();
  static void init_tables_host();

  _DI_ bool consistent() const;
  _DI_ bool complete() const;
  _DI_ void apply_bounds();
  _DI_ LexStatus canonical_with_forced(ForcedCell &forced) const;
  static _DI_ ThreeBoardD4Even<N, W> load_from(const board_array_t<W> &on,
                                               const board_array_t<W> &off);
  _DI_ bool operator==(const ThreeBoardD4Even<N, W> &other) const;

  static _HD_ cuda::std::pair<int, int> local_to_full(cuda::std::pair<unsigned, unsigned> p);
  static _HD_ bool in_domain(int fx, int fy);
  static _HD_ cuda::std::pair<unsigned, unsigned> full_to_local_in_domain(int fx, int fy);
  static _HD_ cuda::std::pair<unsigned, unsigned> full_to_local_rep(int fx, int fy);
  static _HD_ unsigned storage_col_family(unsigned lx, unsigned ly);
  static _DI_ board_row_t<32> logical_col_mask_on_storage_row(unsigned family, unsigned ly);
  static _DI_ BitBoard<W> logical_family_mask(unsigned family);

  static _DI_ BitBoard<W> canonical_reflect(BitBoard<W> board);

  static _DI_ board_row_t<32> column_family_mask(board_row_t<32> families, unsigned ly);
  static _DI_ BinaryCountSaturating<32> family_on_counts(BitBoard<32> board);
  static _DI_ BinaryCountSaturating3<32> family_on_counts3(BitBoard<32> board);
  _DI_ ThreeBoardD4Even<N, W> force_orthogonal() const;
  _DI_ BitBoard<W> vulnerable() const;
  template <unsigned UnknownTarget>
  _DI_ BitBoard<W> semivulnerable_like() const;
  _DI_ BitBoard<W> semivulnerable() const;
  _DI_ BitBoard<W> quasivulnerable() const;

  _DI_ BitBoard<W> preferred_branch_cells() const;

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
__global__ void init_d4even_line_table_kernel_32(uint32_t *__restrict__ table) {
  constexpr unsigned cell_count = ThreeBoardD4Even<N, 32>::CELL_COUNT;
  constexpr unsigned line_rows = ThreeBoardD4Even<N, 32>::LINE_ROWS;
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

  ThreeBoardD4Even<N, 32> board;
  board.known_on.set(px, py);
  board.known_on.set(qx, qy);
  board.eliminate_all_lines_slow({px, py});
  board.eliminate_all_lines_slow({qx, qy});
  board.propagate_slow();

  if (lane < line_rows) {
    table[static_cast<size_t>(pair_idx) * line_rows + lane] = board.known_off.state;
  }
}

template <unsigned N, unsigned W>
inline void ThreeBoardD4Even<N, W>::init_line_table_host() {
  static_assert(W == 32, "ThreeBoardD4Even init_line_table_host expects W=32");
  static uint32_t *d_table_32 = nullptr;

  constexpr unsigned cell_count = CELL_COUNT;
  constexpr size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
  constexpr size_t total_rows = total_entries * LINE_ROWS;

  if (d_table_32 != nullptr) {
    cudaFree(d_table_32);
    d_table_32 = nullptr;
  }
  cudaMalloc((void **)&d_table_32, total_rows * sizeof(uint32_t));
  init_d4even_line_table_kernel_32<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_32);
  cudaGetLastError();
  cudaDeviceSynchronize();

  cudaMemcpyToSymbol(g_d4even_line_table_32, &d_table_32, sizeof(d_table_32));
}

template <unsigned N, unsigned W>
inline void ThreeBoardD4Even<N, W>::init_tables_host() {
  init_lookup_tables_host();
  init_line_table_host();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::bounds() {
  return BitBoard<W>::rect(STORE_W, STORE_H);
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardD4Even<N, W>::apply_bounds() {
  const BitBoard<W> b = bounds();
  known_on &= b;
  known_off &= b;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardD4Even<N, W>::consistent() const {
  return (known_on & known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardD4Even<N, W>::complete() const {
  BitBoard<W> unknown = (~known_on & ~known_off) & bounds();
  return unknown.empty();
}

template <unsigned N, unsigned W>
_HD_ cuda::std::pair<int, int> ThreeBoardD4Even<N, W>::local_to_full(cuda::std::pair<unsigned, unsigned> p) {
  const unsigned lx = p.first;
  const unsigned ly = p.second;
  if (lx <= ly) {
    return {static_cast<int>(lx), static_cast<int>(ly)};
  }
  const int rx = static_cast<int>(lx - (ly + 1));
  return {-static_cast<int>(H) + rx, static_cast<int>(ly)};
}

template <unsigned N, unsigned W>
_HD_ bool ThreeBoardD4Even<N, W>::in_domain(int fx, int fy) {
  if (fy < 0 || fy >= static_cast<int>(H)) {
    return false;
  }
  if (fx >= 0) {
    return fx <= fy;
  }
  return fx >= -static_cast<int>(H) && fx <= (-fy - 1);
}

template <unsigned N, unsigned W>
_HD_ cuda::std::pair<unsigned, unsigned> ThreeBoardD4Even<N, W>::full_to_local_in_domain(int fx, int fy) {
  const unsigned ly = static_cast<unsigned>(fy);
  unsigned lx;
  if (fx >= 0) {
    lx = static_cast<unsigned>(fx);
  } else {
    const int rx = fx + static_cast<int>(H);
    lx = ly + 1 + static_cast<unsigned>(rx);
  }
  return {lx, ly};
}

template <unsigned N, unsigned W>
_HD_ cuda::std::pair<unsigned, unsigned> ThreeBoardD4Even<N, W>::full_to_local_rep(int fx, int fy) {
  if (in_domain(fx, fy)) {
    return full_to_local_in_domain(fx, fy);
  }
  const int dfx = fy;
  const int dfy = fx;
  if (in_domain(dfx, dfy)) {
    return full_to_local_in_domain(dfx, dfy);
  }
  const int rfx = -fx - 1;
  const int rfy = -fy - 1;
  if (in_domain(rfx, rfy)) {
    return full_to_local_in_domain(rfx, rfy);
  }
  const int afx = -fy - 1;
  const int afy = -fx - 1;
  return full_to_local_in_domain(afx, afy);
}

template <unsigned N, unsigned W>
_HD_ unsigned ThreeBoardD4Even<N, W>::storage_col_family(unsigned lx, unsigned ly) {
  if (lx <= ly) {
    return lx;
  }
  return H + ly - lx;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::canonical_reflect(BitBoard<W> board) {
  const unsigned lane = threadIdx.x & 31;
  const bool active = lane < STORE_H;
  const board_row_t<32> mask_row = row_mask();
  const board_row_t<32> src_row = active ? (board.state & mask_row) : 0u;

  // Split storage row into:
  //   A: lower triangle (x >= 0 side), columns [0..ly]
  //   B: upper-triangle tile encoded in compact form (x < 0 side), columns [ly+1..H]
  const board_row_t<32> a_mask = active ? main_triangle_mask(lane) : 0u;
  const board_row_t<32> a_row = src_row & a_mask;
  const board_row_t<32> b_compact = active ? ((src_row & anti_triangle_mask(lane)) >> (lane + 1)) : 0u;

  // Convert compact B tile into an HxH upper-triangular matrix row:
  // compact index k in [0..H-ly-1] maps to matrix column j = H-k-1.
  const unsigned b_width = active ? anti_width(lane) : 0u;
  const board_row_t<32> b_row_mat = active ? unpack_anti_compact(b_compact, lane) : 0u;

  // Reflection x -> -x-1 under this storage is:
  //   A_out = transpose(B_mat)
  //   B_mat_out = transpose(A)
  const BitBoard<32> a_t = BitBoard<32>(a_row).flip_diagonal();
  const BitBoard<32> b_t = BitBoard<32>(b_row_mat).flip_diagonal();

  const board_row_t<32> out_a = active ? (b_t.state & a_mask) : 0u;
  board_row_t<32> out_b = 0u;
  if (active && b_width != 0u) {
    const board_row_t<32> b_compact_out = pack_anti_compact(a_t.state, lane);
    out_b = b_compact_out << (lane + 1u);
  }

  BitBoard<W> out(out_a | out_b);
  out &= bounds();
  return out;
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoardD4Even<N, W>::canonical_with_forced(ForcedCell &forced) const {
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
      const auto full_b = local_to_full(cell);
      const int ax = -full_b.first - 1;
      const int ay = full_b.second;
      cell = full_to_local_rep(ax, ay);
    }
    forced.cell = cell;
  }
  return order;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoardD4Even<N, W> ThreeBoardD4Even<N, W>::load_from(const board_array_t<W> &on,
                                                              const board_array_t<W> &off) {
  ThreeBoardD4Even<N, W> board;
  board.known_on = BitBoard<W>::load(on.data());
  board.known_off = BitBoard<W>::load(off.data());
  board.apply_bounds();
  return board;
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoardD4Even<N, W>::operator==(const ThreeBoardD4Even<N, W> &other) const {
  return known_on == other.known_on && known_off == other.known_off;
}

template <unsigned N, unsigned W>
_DI_ board_row_t<32> ThreeBoardD4Even<N, W>::column_family_mask(board_row_t<32> families, unsigned ly) {
  const board_row_t<32> lower = main_triangle_mask(ly);
  const board_row_t<32> main = families & lower;
  const board_row_t<32> tail = families >> ly;  // j >= ly
  const board_row_t<32> anti = __brev(tail) >> (31u - H);
  return (main | anti) & row_mask();
}

template <unsigned N, unsigned W>
_DI_ board_row_t<32> ThreeBoardD4Even<N, W>::logical_col_mask_on_storage_row(unsigned family,
                                                                              unsigned ly) {
  if (family >= H || ly >= H) {
    return 0u;
  }
  return column_family_mask(board_row_t<32>(1u) << family, ly);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::logical_family_mask(unsigned family) {
  const unsigned lane = threadIdx.x & 31;
  const board_row_t<32> row_bits =
      (family < H && lane == family) ? row_mask() : board_row_t<32>(0u);
  const board_row_t<32> col_bits = logical_col_mask_on_storage_row(family, lane);
  BitBoard<W> out(row_bits | col_bits);
  out &= bounds();
  return out;
}

template <unsigned N, unsigned W>
_DI_ BinaryCountSaturating<32> ThreeBoardD4Even<N, W>::family_on_counts(BitBoard<32> board) {
  const unsigned lane = threadIdx.x & 31;
  const bool active = lane < STORE_H;

  const board_row_t<32> row = active ? (board.state & row_mask()) : 0u;
  const board_row_t<32> tri_main_mask = active ? main_triangle_mask(lane) : 0u;
  const board_row_t<32> tri_anti_mask = active ? anti_triangle_mask(lane) : 0u;

  const board_row_t<32> main = row & tri_main_mask;
  const board_row_t<32> anti = row & tri_anti_mask;

  const board_row_t<32> main_no_diag = main & ~(board_row_t<32>(1u) << lane);
  const board_row_t<32> anti_no_diag = anti & ~(board_row_t<32>(1u) << H);
  const board_row_t<32> anti_aligned = active ? (anti_no_diag >> (lane + 1)) : 0u;

  const BinaryCountSaturating<32> row_counter = BinaryCountSaturating<32>::horizontal(main) +
                                                BinaryCountSaturating<32>::horizontal(anti);
  const BinaryCountSaturating<32> main_col_counter = BinaryCountSaturating<32>::vertical(main_no_diag);
  const BinaryCountSaturating<32> anti_col_k_counter = BinaryCountSaturating<32>::vertical(anti_aligned);
  const BinaryCountSaturating<32> anti_col_i_counter = {
      reverse_low_bits(anti_col_k_counter.bit0),
      reverse_low_bits(anti_col_k_counter.bit1),
  };
  return row_counter + main_col_counter + anti_col_i_counter;
}

template <unsigned N, unsigned W>
_DI_ BinaryCountSaturating3<32> ThreeBoardD4Even<N, W>::family_on_counts3(BitBoard<32> board) {
  const unsigned lane = threadIdx.x & 31;
  const bool active = lane < STORE_H;

  const board_row_t<32> row = active ? (board.state & row_mask()) : 0u;
  const board_row_t<32> tri_main_mask = active ? main_triangle_mask(lane) : 0u;
  const board_row_t<32> tri_anti_mask = active ? anti_triangle_mask(lane) : 0u;

  const board_row_t<32> main = row & tri_main_mask;
  const board_row_t<32> anti = row & tri_anti_mask;

  const board_row_t<32> main_no_diag = main & ~(board_row_t<32>(1u) << lane);
  const board_row_t<32> anti_no_diag = anti & ~(board_row_t<32>(1u) << H);
  const board_row_t<32> anti_aligned = active ? (anti_no_diag >> (lane + 1)) : 0u;

  const BinaryCountSaturating3<32> row_counter = BinaryCountSaturating3<32>::horizontal(main) +
                                                 BinaryCountSaturating3<32>::horizontal(anti);
  const BinaryCountSaturating3<32> main_col_counter = BinaryCountSaturating3<32>::vertical(main_no_diag);
  const BinaryCountSaturating3<32> anti_col_k_counter = BinaryCountSaturating3<32>::vertical(anti_aligned);
  const BinaryCountSaturating3<32> anti_col_i_counter = {
      reverse_low_bits(anti_col_k_counter.bit0),
      reverse_low_bits(anti_col_k_counter.bit1),
      reverse_low_bits(anti_col_k_counter.bit2),
  };
  return row_counter + main_col_counter + anti_col_i_counter;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoardD4Even<N, W> ThreeBoardD4Even<N, W>::force_orthogonal() const {
  ThreeBoardD4Even<N, W> result = *this;
  const unsigned lane = threadIdx.x & 31;

  const BinaryCountSaturating<32> on_counter = family_on_counts(known_on);
  const board_row_t<32> on_eq_2 = on_counter.template eq_target<2>() & family_mask();
  const board_row_t<32> on_gt_2 = (on_counter.bit1 & on_counter.bit0) & family_mask();

  const BitBoard<32> not_known_off = (~known_off) & bounds();
  const BinaryCountSaturating<32> not_off_counter = family_on_counts(not_known_off);
  const board_row_t<32> not_off_eq_2 = not_off_counter.template eq_target<2>() & family_mask();
  const board_row_t<32> not_off_lt_2 = (~not_off_counter.bit1) & family_mask();

  const bool contradiction = (on_gt_2 != 0u) || (not_off_lt_2 != 0u);
  if (__any_sync(0xffffffffu, contradiction)) {
    const BitBoard<32> bds = bounds();
    result.known_on |= bds;
    result.known_off |= bds;
    result.apply_bounds();
    return result;
  }

  const board_row_t<32> row_full = row_mask();
  const board_row_t<32> row_on_eq = ((on_eq_2 >> lane) & 1u) ? row_full : 0u;
  const board_row_t<32> col_on_eq = column_family_mask(on_eq_2, lane);
  const board_row_t<32> force_off_mask = row_on_eq | col_on_eq;
  result.known_off.state |= (~known_on.state) & force_off_mask;

  const board_row_t<32> row_not_off_eq = ((not_off_eq_2 >> lane) & 1u) ? row_full : 0u;
  const board_row_t<32> col_not_off_eq = column_family_mask(not_off_eq_2, lane);
  const board_row_t<32> force_on_mask = row_not_off_eq | col_not_off_eq;
  result.known_on.state |= (~known_off.state) & force_on_mask;

  result.apply_bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::vulnerable() const {
  const unsigned lane = threadIdx.x & 31;

  const BitBoard<W> unknown = (~known_on & ~known_off) & bounds();
  const BinaryCountSaturating3<32> on_counter = family_on_counts3(known_on);
  const BinaryCountSaturating3<32> unknown_counter = family_on_counts3(unknown);

  const board_row_t<32> on_eq_0 = on_counter.template eq_target<0>() & family_mask();
  const board_row_t<32> on_eq_1 = on_counter.template eq_target<1>() & family_mask();
  const board_row_t<32> unknown_eq_2 = unknown_counter.template eq_target<2>() & family_mask();
  const board_row_t<32> unknown_eq_3 = unknown_counter.template eq_target<3>() & family_mask();
  const board_row_t<32> line_match = (on_eq_1 & unknown_eq_2) | (on_eq_0 & unknown_eq_3);

  const board_row_t<32> row_match = ((line_match >> lane) & 1u) ? row_mask() : board_row_t<32>(0u);
  const board_row_t<32> col_match = column_family_mask(line_match, lane);

  BitBoard<W> result{};
  result.state = unknown.state & (row_match | col_match);
  result &= bounds();
  return result;
}

template <unsigned N, unsigned W>
template <unsigned UnknownTarget>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::semivulnerable_like() const {
  static_assert(UnknownTarget < 8, "semivulnerable_like expects target in [0, 7]");
  const unsigned lane = threadIdx.x & 31;

  const BitBoard<W> unknown = (~known_on & ~known_off) & bounds();
  const BinaryCountSaturating3<32> on_counter = family_on_counts3(known_on);
  const BinaryCountSaturating3<32> unknown_counter = family_on_counts3(unknown);

  const board_row_t<32> on_eq_0 = on_counter.template eq_target<0>() & family_mask();
  const board_row_t<32> unknown_eq = unknown_counter.template eq_target<UnknownTarget>() & family_mask();
  const board_row_t<32> line_match = on_eq_0 & unknown_eq;

  const board_row_t<32> row_match = ((line_match >> lane) & 1u) ? row_mask() : board_row_t<32>(0u);
  const board_row_t<32> col_match = column_family_mask(line_match, lane);

  BitBoard<W> result{};
  result.state = unknown.state & (row_match | col_match);
  result &= bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::semivulnerable() const {
  return semivulnerable_like<4>();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::quasivulnerable() const {
  return semivulnerable_like<5>();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::preferred_branch_cells() const {
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

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                                              cuda::std::pair<int, int> qj,
                                                              int step_x,
                                                              int step_y) const {
  BitBoard<W> result;
  const int row = static_cast<int>(threadIdx.x & 31);
  if (row >= static_cast<int>(STORE_H)) {
    return result;
  }
  const int fy = row;
  if (fy == pi.second || fy == qj.second) {
    return result;
  }

  const int diff = fy - pi.second;
  if (diff % step_y != 0) {
    return result;
  }

  const int k = diff / step_y;
  const int fx = pi.first + step_x * k;
  if (!in_domain(fx, fy)) {
    return result;
  }

  const auto local = full_to_local_in_domain(fx, fy);
  result.state |= (board_row_t<32>(1u) << local.first);
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::eliminate_pair(cuda::std::pair<int, int> pi,
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

  const int abs_dx = dx < 0 ? -dx : dx;
  const int abs_dy = dy < 0 ? -dy : dy;
  int step_x = (dx < 0 ? -1 : 1) * static_cast<int>(div_gcd_table[abs_dx][abs_dy]);
  int step_y = (dy < 0 ? -1 : 1) * static_cast<int>(div_gcd_table[abs_dy][abs_dx]);
  if (step_y < 0) {
    step_y = -step_y;
    step_x = -step_x;
  }
  return eliminate_pair_steps(pi, qj, step_x, step_y);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                                             cuda::std::pair<unsigned, unsigned> q) const {
  BitBoard<W> result;
  if (p == q) {
    return result;
  }

  const auto pf = local_to_full(p);
  const auto qf = local_to_full(q);

  cuda::std::pair<int, int> po[4];
  cuda::std::pair<int, int> qo[4];
  int pn = 0;
  int qn = 0;

  auto add_unique = [](cuda::std::pair<int, int> out[4], int &count, cuda::std::pair<int, int> pt) {
    for (int i = 0; i < count; ++i) {
      if (out[i] == pt) {
        return;
      }
    }
    out[count++] = pt;
  };

  add_unique(po, pn, pf);
  add_unique(po, pn, {pf.second, pf.first});
  add_unique(po, pn, {-pf.first - 1, -pf.second - 1});
  add_unique(po, pn, {-pf.second - 1, -pf.first - 1});

  add_unique(qo, qn, qf);
  add_unique(qo, qn, {qf.second, qf.first});
  add_unique(qo, qn, {-qf.first - 1, -qf.second - 1});
  add_unique(qo, qn, {-qf.second - 1, -qf.first - 1});

  for (int i = 0; i < pn; ++i) {
    for (int j = 0; j < qn; ++j) {
      result |= eliminate_pair(po[i], qo[j]);
    }
  }

  result &= bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardD4Even<N, W>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                                        cuda::std::pair<unsigned, unsigned> q) const {
  constexpr unsigned cell_count = CELL_COUNT;
  const unsigned p_idx = p.second * STORE_W + p.first;
  const unsigned q_idx = q.second * STORE_W + q.first;
  const size_t base = (static_cast<size_t>(p_idx) * cell_count + q_idx) * LINE_ROWS;
  const unsigned lane = threadIdx.x & 31;
  const uint32_t *__restrict__ table = g_d4even_line_table_32;
  const uint32_t row = (lane < LINE_ROWS) ? __ldg(table + base + lane) : 0u;
  return BitBoard<W>(row);
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardD4Even<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
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
_DI_ void ThreeBoardD4Even<N, W>::eliminate_all_lines(BitBoard<W> seed) {
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
_DI_ void ThreeBoardD4Even<N, W>::eliminate_all_lines_slow(cuda::std::pair<unsigned, unsigned> p) {
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
_DI_ void ThreeBoardD4Even<N, W>::eliminate_all_lines_slow(BitBoard<W> seed) {
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
_DI_ void ThreeBoardD4Even<N, W>::propagate() {
  ThreeBoardD4Even<N, W> prev;
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
_DI_ void ThreeBoardD4Even<N, W>::propagate_slow() {
  ThreeBoardD4Even<N, W> prev;
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
