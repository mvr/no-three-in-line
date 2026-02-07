#pragma once

#include "common.hpp"
#include "board.cu"
#include "three_board.cu"

#include <cuda/std/array>
#include <cuda/std/utility>

__device__ uint32_t *g_c4_line_table_32 = nullptr;

// C4-symmetric board 2N × 2N for N up to 32. Stores only the
// fundamental domain [0, N) × [0, N); the remaining three quadrants are
// implied by fourfold rotational symmetry around (-0.5, -0.5).
template <unsigned N>
struct ThreeBoardC4 {
  static_assert(N <= 32, "Initial C4 implementation limited to N <= 32");

  BitBoard<32> known_on;
  BitBoard<32> known_off;

  _DI_ ThreeBoardC4() : known_on{}, known_off{} {}
  _DI_ ThreeBoardC4(BitBoard<32> on, BitBoard<32> off) : known_on{on}, known_off{off} {}

  static _DI_ BitBoard<32> bounds();
  static _DI_ BitBoard<32> relevant_endpoint(cuda::std::pair<unsigned, unsigned> p);

  _DI_ bool consistent() const;
  _DI_ unsigned unknown_pop() const;
  _DI_ bool complete() const;
  _DI_ LexStatus canonical_with_forced(ForcedCell &forced) const;
  static _DI_ ThreeBoardC4<N> load_from(const board_array_t<32> &on,
                                        const board_array_t<32> &off);
  _DI_ bool operator==(const ThreeBoardC4<N> &other) const;

  _DI_ ThreeBoardC4<N> force_orthogonal() const;
  _DI_ BitBoard<32> vulnerable() const;
  _DI_ BitBoard<32> semivulnerable() const;
  _DI_ BitBoard<32> quasivulnerable() const;
  template <unsigned UnknownTarget>
  _DI_ BitBoard<32> semivulnerable_like() const;
  _DI_ void apply_bounds();

  static constexpr unsigned FULL_N = 2 * N;
  static constexpr unsigned FULL_W = (FULL_N <= 32) ? 32 : 64;

  // Helpers for reasoning about orbits.
  static _DI_ cuda::std::pair<int, int> rotate90(cuda::std::pair<int, int> p);
  static _DI_ cuda::std::array<cuda::std::pair<int, int>, 4> orbit(cuda::std::pair<int, int> p);
  template <typename Visitor>
  static _DI_ void for_each_orbit_point(cuda::std::pair<int, int> p, Visitor &&visitor);

  _DI_ BitBoard<32> eliminate_line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
  _DI_ BitBoard<32> eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                        cuda::std::pair<unsigned, unsigned> q) const;
  _DI_ BitBoard<32> eliminate_pair(cuda::std::pair<int, int> pi, cuda::std::pair<int, int> qj) const;
  _DI_ BitBoard<32> eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                         cuda::std::pair<int, int> qj,
                                         int step_x, int step_y) const;
  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<32> seed);
  _DI_ void eliminate_all_lines_slow(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines_slow(BitBoard<32> seed);

  _DI_ void propagate();
  _DI_ void propagate_slow();
};

// --- Inline implementation -------------------------------------------------

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::bounds() {
  BitBoard<32> result;
  const unsigned lane = threadIdx.x & 31;
  if (lane < N) {
    const board_row_t<32> mask = (N == 32) ? 0xffffffffu : ((board_row_t<32>(1) << N) - 1u);
    result.state = mask;
  } else {
    result.state = 0;
  }
  return result;
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::relevant_endpoint(cuda::std::pair<unsigned, unsigned> p) {
  uint64_t fullrow = relevant_endpoint_table[32 - p.second + (threadIdx.x & 31)];
  uint32_t moved_row = fullrow >> (32 - p.first);
  return BitBoard<32>(moved_row);
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::apply_bounds() {
  const BitBoard<32> b = bounds();
  known_on &= b;
  known_off &= b;
}

template <unsigned N>
_DI_ bool ThreeBoardC4<N>::consistent() const {
  return (known_on & known_off).empty();
}

template <unsigned N>
_DI_ unsigned ThreeBoardC4<N>::unknown_pop() const {
  return N * N - (known_on | known_off).pop();
}

template <unsigned N>
_DI_ bool ThreeBoardC4<N>::complete() const {
  BitBoard<32> unknown = ~(known_on | known_off) & bounds();
  return unknown.empty();
}

template <unsigned N>
_DI_ LexStatus ThreeBoardC4<N>::canonical_with_forced(ForcedCell &forced) const {
  BitBoard<32> diag_on = known_on.flip_diagonal();
  BitBoard<32> diag_off = known_off.flip_diagonal();
  BitBoard<32> bds = bounds();
  diag_on &= bds;
  diag_off &= bds;
  ForceCandidate local_force{};
  LexStatus order = compare_with_unknowns_forced<32>(known_on,
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

template <unsigned N>
_DI_ ThreeBoardC4<N> ThreeBoardC4<N>::load_from(const board_array_t<32> &on,
                                                const board_array_t<32> &off) {
  ThreeBoardC4<N> board;
  board.known_on = BitBoard<32>::load(on.data());
  board.known_off = BitBoard<32>::load(off.data());
  board.apply_bounds();
  return board;
}

template <unsigned N>
_DI_ bool ThreeBoardC4<N>::operator==(const ThreeBoardC4<N> &other) const {
  return known_on == other.known_on && known_off == other.known_off;
}

template <unsigned N>
_DI_ ThreeBoardC4<N> ThreeBoardC4<N>::force_orthogonal() const {
  ThreeBoardC4<N> result = *this;

  const board_row_t<32> lane_bit = 1u << (threadIdx.x & 31);

  const BinaryCountSaturating<32> row_on_counter = count_horizontally_saturating<32>(known_on.state);
  const BinaryCountSaturating<32> col_on_counter = count_vertically_saturating<32>(known_on.state);
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

  BitBoard<32> not_known_off = (~known_off) & bounds();

  const BinaryCountSaturating<32> row_not_off_counter = count_horizontally_saturating<32>(not_known_off.state);
  const BinaryCountSaturating<32> col_not_off_counter = count_vertically_saturating<32>(not_known_off.state);
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

  result.apply_bounds();

  return result;
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::vulnerable() const {
  BitBoard<32> result;

  BitBoard<32> unknown = (~known_on & ~known_off) & bounds();

  const BinaryCountSaturating<32> row_on_counter = count_horizontally_saturating<32>(known_on.state);
  const BinaryCountSaturating<32> col_on_counter = count_vertically_saturating<32>(known_on.state);
  const BinaryCountSaturating<32> total_on_counter = row_on_counter + col_on_counter;

  const BinaryCountSaturating3<32> row_unknown_counter = count_horizontally_saturating3<32>(unknown.state);
  const BinaryCountSaturating3<32> col_unknown_counter = count_vertically_saturating3<32>(unknown.state);
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

  result &= unknown & bounds();

  return result;
}

template <unsigned N>
template <unsigned UnknownTarget>
_DI_ BitBoard<32> ThreeBoardC4<N>::semivulnerable_like() const {
  static_assert(UnknownTarget < 8, "semivulnerable_like expects a target < 8");
  BitBoard<32> result;

  BitBoard<32> unknown = (~known_on & ~known_off) & bounds();

  const BinaryCountSaturating<32> row_on_counter = count_horizontally_saturating<32>(known_on.state);
  const BinaryCountSaturating<32> col_on_counter = count_vertically_saturating<32>(known_on.state);
  const BinaryCountSaturating<32> total_on_counter = row_on_counter + col_on_counter;

  const BinaryCountSaturating3<32> row_unknown_counter = count_horizontally_saturating3<32>(unknown.state);
  const BinaryCountSaturating3<32> col_unknown_counter = count_vertically_saturating3<32>(unknown.state);
  const BinaryCountSaturating3<32> total_unknown_counter = row_unknown_counter + col_unknown_counter;

  const board_row_t<32> total_on_eq_0 = total_on_counter.template eq_target<0>();
  const board_row_t<32> total_unknown_eq = total_unknown_counter.template eq_target<UnknownTarget>();

  const board_row_t<32> semivuln_rows = total_on_eq_0 & total_unknown_eq;

  const board_row_t<32> lane_bit = 1u << (threadIdx.x & 31);
  if (semivuln_rows & lane_bit) {
    result.state = ~0u;
  }
  result.state |= semivuln_rows;

  result &= unknown & bounds();
  return result;
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::semivulnerable() const {
  return semivulnerable_like<4>();
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::quasivulnerable() const {
  return semivulnerable_like<5>();
}

// --- Orbit helpers ---------------------------------------------------------

template <unsigned N>
_DI_ cuda::std::pair<int, int> ThreeBoardC4<N>::rotate90(cuda::std::pair<int, int> p) {
  return {-p.second - 1, p.first};
}

template <unsigned N>
_DI_ cuda::std::array<cuda::std::pair<int, int>, 4> ThreeBoardC4<N>::orbit(cuda::std::pair<int, int> p) {
  cuda::std::array<cuda::std::pair<int, int>, 4> result{};
  result[0] = p;
  for (int r = 1; r < 4; ++r) {
    result[r] = rotate90(result[r - 1]);
  }
  return result;
}

template <unsigned N>
template <typename Visitor>
_DI_ void ThreeBoardC4<N>::for_each_orbit_point(cuda::std::pair<int, int> p, Visitor &&visitor) {
  const auto pts = orbit(p);
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    visitor(pts[r]);
  }
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::eliminate_pair(cuda::std::pair<int, int> pi,
                                                  cuda::std::pair<int, int> qj) const {
  BitBoard<32> result;

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
  return eliminate_pair(pi, qj, dx, dy, step_x_mag, step_y_mag);
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                                        cuda::std::pair<int, int> qj,
                                                        int step_x, int step_y) const {
  BitBoard<32> result;

  int row = static_cast<int>(threadIdx.x & 31);

  if (pi.second == row || qj.second == row)
    return result;

  int diff = row - pi.second;
  if (diff % step_y != 0)
    return result;

  int k = diff / step_y;
  int col = pi.first + step_x * k;
  if (col < 0 || col >= static_cast<int>(N))
    return result;

  result.state |= 1u << col;
  return result;
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                                  cuda::std::pair<unsigned, unsigned> q) {
  BitBoard<32> result;
  constexpr unsigned cell_count = N * N;
  const unsigned p_idx = p.second * N + p.first;
  const unsigned q_idx = q.second * N + q.first;
  const uint32_t *entry =
      g_c4_line_table_32 + (static_cast<size_t>(p_idx) * cell_count + q_idx) * LINE_TABLE_ROWS;
  const unsigned lane = threadIdx.x & 31;
  const uint32_t row = __ldg(entry + lane);
  return BitBoard<32>(row);
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::eliminate_line_slow(cuda::std::pair<unsigned, unsigned> p,
                                                       cuda::std::pair<unsigned, unsigned> q) const {
  BitBoard<32> result;

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

template <unsigned N>
_DI_ void ThreeBoardC4<N>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<32> qs = known_on;
  const unsigned lane = threadIdx.x & 31;
  const unsigned p_idx = p.second * N + p.first;
  const uint32_t *base =
      g_c4_line_table_32 + (static_cast<size_t>(p_idx) * N * N) * LINE_TABLE_ROWS;

  cuda::std::pair<int, int> q;
  while (qs.some_on_if_any(q)) {
    qs.erase(q);
    const unsigned q_idx = static_cast<unsigned>(q.second) * N + static_cast<unsigned>(q.first);
    const uint32_t row = __ldg(base + q_idx * LINE_TABLE_ROWS + lane);
    known_off |= BitBoard<32>(row);
    if (__any_sync(0xffffffff, row & known_on.state)) {
      return;
    }
  }
  apply_bounds();
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::eliminate_all_lines_slow(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<32> qs = known_on;
  cuda::std::pair<int, int> q;
  while (qs.some_on_if_any(q)) {
    qs.erase(q);
    known_off |= eliminate_line_slow(p, {static_cast<unsigned>(q.first), static_cast<unsigned>(q.second)});
    if (!consistent())
      return;
  }
  apply_bounds();
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::eliminate_all_lines(BitBoard<32> ps) {
  cuda::std::pair<int, int> p;
  while (ps.some_on_if_any(p)) {
    ps.erase(p);

    BitBoard<32> qs = known_on & ~ps;
    const unsigned lane = threadIdx.x & 31;
    const unsigned p_idx = static_cast<unsigned>(p.second) * N + static_cast<unsigned>(p.first);
    const uint32_t *base =
        g_c4_line_table_32 + (static_cast<size_t>(p_idx) * N * N) * LINE_TABLE_ROWS;

    cuda::std::pair<int, int> q;
    while (qs.some_on_if_any(q)) {
      qs.erase(q);
      const unsigned q_idx = static_cast<unsigned>(q.second) * N + static_cast<unsigned>(q.first);
      const uint32_t row = __ldg(base + q_idx * LINE_TABLE_ROWS + lane);
      known_off |= BitBoard<32>(row);
      if (__any_sync(0xffffffff, row & known_on.state)) {
        return;
      }
    }
  }
  apply_bounds();
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::eliminate_all_lines_slow(BitBoard<32> ps) {
  cuda::std::pair<int, int> p;
  while (ps.some_on_if_any(p)) {
    ps.erase(p);

    BitBoard<32> qs = known_on & ~ps;
    cuda::std::pair<int, int> q;
    while (qs.some_on_if_any(q)) {
      qs.erase(q);
      known_off |= eliminate_line_slow(p, q);
      if (!consistent())
        return;
    }
  }
  apply_bounds();
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::propagate() {
  ThreeBoardC4<N> prev;
  BitBoard<32> done_ons = known_on;

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

template <unsigned N>
_DI_ void ThreeBoardC4<N>::propagate_slow() {
  ThreeBoardC4<N> prev;
  BitBoard<32> done_ons = known_on;

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
