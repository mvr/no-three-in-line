#pragma once

#include "board.cu"
#include "common.hpp"
#include "three_board.cu"

#include <cuda/std/array>
#include <cuda/std/utility>

__device__ const uint32_t *__restrict__ g_c4near_line_table_32 = nullptr;

// C4-near board for odd full size (2N-1)x(2N-1), W=32 only.
// Stored domain is x in [0, N), y in [1, N), represented as local:
//   lx = x in [0, N), ly = y-1 in [0, N-1)
// Active long diagonal (x=y, y>0) may contain exactly one ON pair.
// Opposite long diagonal (x=-y) is forced OFF.
template <unsigned N, unsigned W = 32>
struct ThreeBoardC4Near {
  static_assert(W == 32, "ThreeBoardC4Near currently supports only W=32");
  static_assert(N >= 2 && N <= 32, "ThreeBoardC4Near requires 2 <= N <= 32");

  static constexpr unsigned FULL_N = 2 * N - 1;
  static constexpr unsigned FULL_W = (FULL_N <= 32) ? 32 : 64;
  static constexpr unsigned STORE_H = N - 1;
  static constexpr unsigned STORE_W = N;
  static constexpr unsigned LINE_ROWS =
      LINE_TABLE_FULL_WARP_LOAD ? 32 : ((STORE_H + 7u) & ~7u);

  BitBoard<W> known_on;
  BitBoard<W> known_off;

  _DI_ ThreeBoardC4Near() : known_on{}, known_off{} {}
  _DI_ ThreeBoardC4Near(BitBoard<W> on, BitBoard<W> off) : known_on{on}, known_off{off} {}

  static _DI_ BitBoard<W> bounds();
  static _DI_ BitBoard<W> active_diagonal();
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

template <unsigned N, unsigned W>
inline void ThreeBoardC4Near<N, W>::init_line_table_host() {
  static_assert(W == 32, "ThreeBoardC4Near init_line_table_host expects W=32");
  static uint32_t *d_table_32 = nullptr;

  constexpr unsigned cell_count = N * (N - 1);
  constexpr size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
  constexpr size_t total_rows = total_entries * LINE_ROWS;

  if (d_table_32 != nullptr) {
    cudaFree(d_table_32);
    d_table_32 = nullptr;
  }
  cudaMalloc((void **)&d_table_32, total_rows * sizeof(uint32_t));
  init_c4near_line_table_kernel_32<N><<<static_cast<unsigned>(total_entries), 32>>>(d_table_32);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpyToSymbol(g_c4near_line_table_32, &d_table_32, sizeof(d_table_32));
}

template <unsigned N, unsigned W>
inline void ThreeBoardC4Near<N, W>::init_tables_host() {
  static_assert(W == 32, "ThreeBoardC4Near init_tables_host expects W=32");
  init_lookup_tables_host();
  init_line_table_host();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::bounds() {
  return BitBoard<W>::rect(STORE_W, STORE_H);
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::active_diagonal() {
  // TODO W=64
  const unsigned lane = threadIdx.x & 31;

  board_row_t<32> row_mask = 0u;
  if (lane < N - 1) {
    row_mask = board_row_t<32>(1) << (lane + 1);  // Offset by one
  }

  return BitBoard<W>(row_mask);
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
  static_assert(W == 32, "ThreeBoardC4Near canonical_reflect expects W=32");
  const unsigned lane = threadIdx.x & 31;
  const bool active = lane < STORE_H;

  BitBoard<W> refl = board.flip_diagonal().rotate_torus(1, -1);

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

  refl &= bounds();
  return refl;
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoardC4Near<N, W>::canonical_with_forced(ForcedCell &forced) const {
  static_assert(W == 32, "ThreeBoardC4Near canonical_with_forced expects W=32");
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

    board_row_t<32> diagonal_on_mask = active_point.occupied_columns();
    // Already occupied, remainder must be off
    if (diagonal_on_mask != 0) {
      result.known_off |= active_diagonal() & ~known_on;
    }
    // Overpopulated, we have a contradiction
    if ((diagonal_on_mask & (diagonal_on_mask - 1)) != 0) {
      result.known_off |= known_on;
    }

    const BinaryCountSaturating<32> row_on_counter =
        BinaryCountSaturating<32>::horizontal((known_on & ~active_point).state);
    const BinaryCountSaturating<32> col_on_counter =
        BinaryCountSaturating<32>::vertical((known_on & ~active_point).state);
    BinaryCountSaturating<32> total_on_counter =
        row_on_counter.lshift(1) + col_on_counter;

    // Re-add the active point (once)
    total_on_counter +=
        BinaryCountSaturating<32>{active_point.occupied_columns(), 0};

    // Handle the overhanging column by adding it to itself
    total_on_counter += BinaryCountSaturating<32>{total_on_counter.bit0 & 1,
                                                  total_on_counter.bit1 & 1};

    // Proceed as C4
    const unsigned lane = threadIdx.x & 31;
    const board_row_t<32> col_bit =
        (lane < STORE_H) ? (board_row_t<32>(1) << (lane + 1)) : 0u;

    const board_row_t<32> total_on_eq_2 =
        total_on_counter.bit1 & ~total_on_counter.bit0;
    const board_row_t<32> total_on_gt_2 =
        total_on_counter.bit1 & total_on_counter.bit0;

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
  }

  {
    BitBoard<W> not_known_off = (~known_off) & bounds();

    BitBoard<W> active_not_known_off = not_known_off & active_diagonal();

    board_row_t<32> diagonal_not_known_off_mask = active_not_known_off.occupied_columns();
    // All known off, contradiction
    if (diagonal_not_known_off_mask == 0) {
      result.known_on.state |= ~0u;
    }
    // One not known off, must be on
    if ((diagonal_not_known_off_mask & (diagonal_not_known_off_mask - 1)) == 0) {
      result.known_on |= active_not_known_off;
    }

    const BinaryCountSaturating<32> row_not_off_counter =
        BinaryCountSaturating<32>::horizontal(
            (not_known_off & ~active_not_known_off).state);
    const BinaryCountSaturating<32> col_not_off_counter =
        BinaryCountSaturating<32>::vertical(
            (not_known_off & ~active_not_known_off).state);
    BinaryCountSaturating<32> total_not_off_counter =
        row_not_off_counter.lshift(1) + col_not_off_counter;

    // Add active unknowns once
    total_not_off_counter +=
        BinaryCountSaturating<32>{active_not_known_off.occupied_columns(), 0};

    // Handle the overhanging column by adding it to itself
    total_not_off_counter += BinaryCountSaturating<32>{
        total_not_off_counter.bit0 & 1, total_not_off_counter.bit1 & 1};

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
  }

  result.apply_bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::vulnerable() const {
  static_assert(W == 32, "ThreeBoardC4Near vulnerable expects W=32");
  constexpr board_row_t<32> row_mask =
      (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
  constexpr unsigned pos_count = STORE_H;

  const unsigned lane = threadIdx.x & 31;
  const bool active = lane < pos_count;

  const BitBoard<W> not_known_off = (~known_off) & bounds();
  const BitBoard<W> active_on = known_on & active_diagonal();
  const BitBoard<W> active_not_off = not_known_off & active_diagonal();

  BinaryCountSaturating3<32> total_on_counter =
      BinaryCountSaturating3<32>::horizontal((known_on & ~active_on).state).lshift(1) +
      BinaryCountSaturating3<32>::vertical((known_on & ~active_on).state);
  total_on_counter += BinaryCountSaturating3<32>{active_on.occupied_columns(), 0, 0};
  total_on_counter +=
      BinaryCountSaturating3<32>{total_on_counter.bit0 & 1, total_on_counter.bit1 & 1,
                                 total_on_counter.bit2 & 1};

  BinaryCountSaturating3<32> total_not_off_counter =
      BinaryCountSaturating3<32>::horizontal((not_known_off & ~active_not_off).state).lshift(1) +
      BinaryCountSaturating3<32>::vertical((not_known_off & ~active_not_off).state);
  total_not_off_counter +=
      BinaryCountSaturating3<32>{active_not_off.occupied_columns(), 0, 0};
  total_not_off_counter += BinaryCountSaturating3<32>{
      total_not_off_counter.bit0 & 1, total_not_off_counter.bit1 & 1,
      total_not_off_counter.bit2 & 1};

  const board_row_t<32> line_match =
      total_not_off_counter.template eq_target<3>() &
      ~(total_on_counter.bit1 | total_on_counter.bit2);  // on in {0,1}
  const board_row_t<32> col_bit =
      active ? (board_row_t<32>(1) << (lane + 1)) : board_row_t<32>(0);
  const bool row_match = (line_match & col_bit) != 0;
  const board_row_t<32> unknown_row =
      active ? ((~known_on.state & ~known_off.state) & row_mask) : 0u;

  BitBoard<W> result{};
  result.state =
      unknown_row & (line_match | (row_match ? row_mask : board_row_t<32>(0)));
  return result;
}

template <unsigned N, unsigned W>
template <unsigned UnknownTarget>
_DI_ BitBoard<W> ThreeBoardC4Near<N, W>::semivulnerable_like() const {
  static_assert(W == 32, "ThreeBoardC4Near semivulnerable_like expects W=32");
  static_assert(UnknownTarget == 4 || UnknownTarget == 5,
                "UnknownTarget must be 4 or 5");
  constexpr board_row_t<32> row_mask =
      (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
  constexpr unsigned pos_count = STORE_H;

  const unsigned lane = threadIdx.x & 31;
  const bool active = lane < pos_count;

  const BitBoard<W> not_known_off = (~known_off) & bounds();
  const BitBoard<W> active_on = known_on & active_diagonal();
  const BitBoard<W> active_not_off = not_known_off & active_diagonal();

  BinaryCountSaturating3<32> total_on_counter =
      BinaryCountSaturating3<32>::horizontal((known_on & ~active_on).state).lshift(1) +
      BinaryCountSaturating3<32>::vertical((known_on & ~active_on).state);
  total_on_counter += BinaryCountSaturating3<32>{active_on.occupied_columns(), 0, 0};
  total_on_counter +=
      BinaryCountSaturating3<32>{total_on_counter.bit0 & 1, total_on_counter.bit1 & 1,
                                 total_on_counter.bit2 & 1};

  BinaryCountSaturating3<32> total_not_off_counter =
      BinaryCountSaturating3<32>::horizontal((not_known_off & ~active_not_off).state).lshift(1) +
      BinaryCountSaturating3<32>::vertical((not_known_off & ~active_not_off).state);
  total_not_off_counter +=
      BinaryCountSaturating3<32>{active_not_off.occupied_columns(), 0, 0};
  total_not_off_counter += BinaryCountSaturating3<32>{
      total_not_off_counter.bit0 & 1, total_not_off_counter.bit1 & 1,
      total_not_off_counter.bit2 & 1};

  const board_row_t<32> line_match =
      total_on_counter.template eq_target<0>() &
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
  const int ly = static_cast<int>(threadIdx.x & 31);
  if (ly >= static_cast<int>(STORE_H)) {
    return result;
  }

  const int fy = ly + 1;
  if (pi.second == fy || qj.second == fy) {
    return result;
  }

  const int diff = fy - pi.second;
  if (diff % step_y != 0) {
    return result;
  }

  const int k = diff / step_y;
  const int fx = pi.first + step_x * k;
  if (fx < 0 || fx >= static_cast<int>(STORE_W)) {
    return result;
  }

  result.state |= (board_row_t<W>(1) << static_cast<unsigned>(fx));
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
  const uint32_t *__restrict__ table = g_c4near_line_table_32;
  const uint32_t row = (lane < LINE_ROWS) ? __ldg(table + base + lane) : 0u;
  return BitBoard<W>(row);
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoardC4Near<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = known_on;
  cuda::std::pair<int, int> q;
  while (qs.pop_on_if_any(q)) {
    known_off |= eliminate_line(p, {static_cast<unsigned>(q.first), static_cast<unsigned>(q.second)});
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
      known_off |= eliminate_line({static_cast<unsigned>(p.first), static_cast<unsigned>(p.second)},
                                  {static_cast<unsigned>(q.first), static_cast<unsigned>(q.second)});
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
    known_off |= eliminate_line_slow(p, {static_cast<unsigned>(q.first), static_cast<unsigned>(q.second)});
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
      known_off |= eliminate_line_slow({static_cast<unsigned>(p.first), static_cast<unsigned>(p.second)},
                                       {static_cast<unsigned>(q.first), static_cast<unsigned>(q.second)});
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
