#pragma once

#include "common.hpp"
#include "board.cu"
#include "three_board.cu"

#include <cuda/std/array>
#include <cuda/std/utility>

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

  _DI_ bool consistent() const;
  _DI_ unsigned unknown_pop() const;
  _DI_ uint32_t priority() const;
  _DI_ bool operator==(const ThreeBoardC4<N> &other) const;

  _DI_ ThreeBoardC4<N> force_orthogonal() const;
  _DI_ BitBoard<32> vulnerable() const;
  _DI_ void apply_bounds();

  static constexpr unsigned FULL_N = 2 * N;
  static constexpr unsigned FULL_W = (FULL_N <= 32) ? 32 : 64;
  using FullBoard = ThreeBoard<FULL_N, FULL_W>;
  using FullBitBoard = BitBoard<FULL_W>;

  static _DI_ FullBitBoard expand_mask(BitBoard<32> mask);
  static _DI_ BitBoard<32> project_mask(const FullBitBoard &mask);
  _DI_ FullBoard expand_to_full() const;
  _DI_ void project_from_full(const FullBoard &full);

  // Helpers for reasoning about orbits.
  static _DI_ cuda::std::pair<int, int> rotate90(cuda::std::pair<int, int> p);
  static _DI_ cuda::std::pair<int, int> apply_rotation(cuda::std::pair<int, int> p, int quarter_turns);
  static _DI_ bool in_domain(cuda::std::pair<int, int> p);
  static _DI_ cuda::std::pair<int, int> fold_to_domain(cuda::std::pair<int, int> p, int &rotation);
  static _DI_ cuda::std::array<cuda::std::pair<int, int>, 4> orbit(cuda::std::pair<int, int> p);
  template <typename Visitor>
  static _DI_ void for_each_orbit_point(cuda::std::pair<int, int> p, Visitor &&visitor);

  _DI_ BitBoard<32> eliminate_line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
  _DI_ BitBoard<32> eliminate_pair(cuda::std::pair<int, int> pi, cuda::std::pair<int, int> qj) const;
  _DI_ BitBoard<32> eliminate_pair_steps(cuda::std::pair<int, int> pi,
                                         cuda::std::pair<int, int> qj,
                                         int step_x, int step_y) const;
  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<32> seed);

  _DI_ void propagate();

  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_row() const;
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
_DI_ uint32_t ThreeBoardC4<N>::priority() const {
  const unsigned on = known_on.pop();
  const unsigned off = known_off.pop();
  // const unsigned vul = vulnerable().pop();

  // const unsigned unknown = N*N - on - off;

  // return on + off;
  // return on;
  return N*N*on - off;
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

  const BinaryCount<32> row_on_counter = count_horizontally<32>(known_on.state);
  const BinaryCount<32> col_on_counter = count_vertically<32>(known_on.state);
  const BinaryCount<32> total_on_counter = row_on_counter + col_on_counter;

  const BinaryCount<32> row_unknown_counter = count_horizontally<32>(unknown.state);
  const BinaryCount<32> col_unknown_counter = count_vertically<32>(unknown.state);
  const BinaryCount<32> total_unknown_counter = row_unknown_counter + col_unknown_counter;

  auto eq0 = [](const BinaryCount<32> &cnt) {
    return ~cnt.bit0 & ~cnt.bit1 & ~cnt.overflow;
  };
  auto eq1 = [](const BinaryCount<32> &cnt) {
    return cnt.bit0 & ~cnt.bit1 & ~cnt.overflow;
  };
  auto eq2 = [](const BinaryCount<32> &cnt) {
    return ~cnt.bit0 & cnt.bit1 & ~cnt.overflow;
  };
  auto eq3 = [](const BinaryCount<32> &cnt) {
    return cnt.bit0 & cnt.bit1 & ~cnt.overflow;
  };

  const board_row_t<32> total_on_eq_0 = eq0(total_on_counter);
  const board_row_t<32> total_on_eq_1 = eq1(total_on_counter);
  const board_row_t<32> total_unknown_eq_2 = eq2(total_unknown_counter);
  const board_row_t<32> total_unknown_eq_3 = eq3(total_unknown_counter);

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

// TODO just use symmetry helpers
template <unsigned N>
_DI_ typename ThreeBoardC4<N>::FullBoard ThreeBoardC4<N>::expand_to_full() const {
  FullBoard full;

  for (int y = 0; y < static_cast<int>(N); ++y) {
    const board_row_t<32> row_on = known_on.row(y);
    const board_row_t<32> row_off = known_off.row(y);

    for (int x = 0; x < static_cast<int>(N); ++x) {
      const board_row_t<32> bit = board_row_t<32>(1u) << x;
      const bool is_on = (row_on & bit) != 0;
      const bool is_off = (row_off & bit) != 0;

      if (!is_on && !is_off)
        continue;

      for_each_orbit_point({x, y}, [&](cuda::std::pair<int, int> pt) {
        const int board_x = pt.first + static_cast<int>(N);
        const int board_y = pt.second + static_cast<int>(N);
        if (is_on) {
          full.known_on.set(board_x, board_y);
        }
        if (is_off) {
          full.known_off.set(board_x, board_y);
        }
      });
    }
  }

  return full;
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::project_from_full(const FullBoard &full) {
  BitBoard<32> proj_on;
  BitBoard<32> proj_off;

  for (int y = 0; y < static_cast<int>(N); ++y) {
    for (int x = 0; x < static_cast<int>(N); ++x) {
      const unsigned fx = static_cast<unsigned>(x + N);
      const unsigned fy = static_cast<unsigned>(y + N);
      if (full.known_on.get(fx, fy)) {
        proj_on.set(x, y);
      }
      if (full.known_off.get(fx, fy)) {
        proj_off.set(x, y);
      }
    }
  }

  known_on = proj_on;
  known_off = proj_off;
  apply_bounds();
}

template <unsigned N>
_DI_ typename ThreeBoardC4<N>::FullBitBoard ThreeBoardC4<N>::expand_mask(BitBoard<32> mask) {
  FullBitBoard result;

  while (!mask.empty()) {
    auto cell = mask.some_on();
    mask.erase(cell);

    for_each_orbit_point(cell, [&](cuda::std::pair<int, int> pt) {
      const int fx = pt.first + static_cast<int>(N);
      const int fy = pt.second + static_cast<int>(N);
      result.set(fx, fy);
    });
  }

  return result;
}

template <unsigned N>
_DI_ BitBoard<32> ThreeBoardC4<N>::project_mask(const FullBitBoard &mask) {
  BitBoard<32> result;

  for (int y = 0; y < static_cast<int>(N); ++y) {
    for (int x = 0; x < static_cast<int>(N); ++x) {
      bool set = false;
      for_each_orbit_point({x, y}, [&](cuda::std::pair<int, int> pt) {
        if (set) return;
        const int fx = pt.first + static_cast<int>(N);
        const int fy = pt.second + static_cast<int>(N);
        if (mask.get(fx, fy)) {
          set = true;
        }
      });

      if (set) {
        result.set(x, y);
      }
    }
  }

  result &= bounds();
  return result;
}

// --- Orbit helpers ---------------------------------------------------------

template <unsigned N>
_DI_ cuda::std::pair<int, int> ThreeBoardC4<N>::rotate90(cuda::std::pair<int, int> p) {
  return {-p.second - 1, p.first};
}

template <unsigned N>
_DI_ cuda::std::pair<int, int> ThreeBoardC4<N>::apply_rotation(cuda::std::pair<int, int> p, int quarter_turns) {
  quarter_turns &= 3;
  for (int i = 0; i < quarter_turns; ++i) {
    p = rotate90(p);
  }
  return p;
}

template <unsigned N>
_DI_ bool ThreeBoardC4<N>::in_domain(cuda::std::pair<int, int> p) {
  return p.first >= 0 && p.first < static_cast<int>(N) && p.second >= 0 && p.second < static_cast<int>(N);
}

template <unsigned N>
_DI_ cuda::std::pair<int, int> ThreeBoardC4<N>::fold_to_domain(cuda::std::pair<int, int> p, int &rotation) {
  for (int r = 0; r < 4; ++r) {
    if (in_domain(p)) {
      rotation = r;
      return p;
    }
    p = rotate90(p);
  }
  rotation = 0;
  return {-1, -1};
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
  while (!qs.empty()) {
    auto q = qs.some_on();
    qs.erase(q);
    known_off |= eliminate_line(p, q);
    if (!consistent())
      return;
  }
  apply_bounds();
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::eliminate_all_lines(BitBoard<32> ps) {
  while (!ps.empty()) {
    auto p = ps.some_on();
    ps.erase(p);

    BitBoard<32> qs = known_on & ~ps;

    while (!qs.empty()) {
      auto q = qs.some_on();
      qs.erase(q);
      known_off |= eliminate_line(p, q);
      if (!consistent())
        return;
    }
  }
  apply_bounds();
}

template <unsigned N>
_DI_ void ThreeBoardC4<N>::propagate() {
  ThreeBoardC4<N> prev_state;
  BitBoard<32> processed = known_on;

  do {
    prev_state = *this;

    ThreeBoardC4<N> prev_force;
    do {
      prev_force = *this;
      ThreeBoardC4<N> forced = force_orthogonal();
      known_on = forced.known_on;
      known_off = forced.known_off;
      apply_bounds();
      if (!consistent())
        return;
    } while (!(*this == prev_force));

    BitBoard<32> newly_on = known_on & ~processed;
    if (!newly_on.empty()) {
      eliminate_all_lines(newly_on);
      if (!consistent())
        return;
      processed = known_on;
    }
  } while (!(*this == prev_state));
}

template <unsigned N>
_DI_ cuda::std::pair<unsigned, unsigned>
ThreeBoardC4<N>::most_constrained_row() const {
  unsigned row;

  BitBoard<32> unknown = ~(known_on | known_off) & bounds();

  BitBoard<32> known_on_rot = known_on.flip_diagonal();
  BitBoard<32> unknown_rot = unknown.flip_diagonal();

  unsigned ons = popcount<32>(known_on.state) + popcount<32>(known_on_rot.state);
  unsigned unknowns = popcount<32>(unknown.state) + popcount<32>(unknown_rot.state);

  if(ons == 1)
    unknowns = unknowns * (unknowns + 1) / 2;

  if ((threadIdx.x & 31) >= N || unknowns == 0)
    unknowns = std::numeric_limits<unsigned>::max();

  row = (threadIdx.x & 31);

  for (int offset = 16; offset > 0; offset /= 2) {
    unsigned other_row = __shfl_down_sync(0xffffffff, row, offset);
    unsigned other_unknowns = __shfl_down_sync(0xffffffff, unknowns, offset);

    if (other_unknowns < unknowns) {
      row = other_row;
      unknowns = other_unknowns;
    }
  }

  row = __shfl_sync(0xffffffff, row, 0);
  unknowns = __shfl_sync(0xffffffff, unknowns, 0);

  return {row, unknowns};
}
