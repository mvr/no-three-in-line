#pragma once

#include "common.hpp"

#include "board.cu"

template <unsigned N, unsigned W>
struct ThreeBoard {
  BitBoard<W> knownOn;
  BitBoard<W> knownOff;

  _DI_ ThreeBoard() : knownOn{}, knownOff{} {}
  _DI_ explicit ThreeBoard(BitBoard<W> knownOn, BitBoard<W> knownOff) : knownOn{knownOn}, knownOff{knownOff} {}

  _DI_ bool operator==(ThreeBoard<N, W> other) const { return (knownOn == other.knownOn) && (knownOff == other.knownOff); }
  _DI_ bool operator!=(ThreeBoard<N, W> other) const { return !(*this == other); }

  static _DI_ BitBoard<W> bounds();
  static _DI_ BitBoard<W> line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);

  _DI_ bool consistent() const;
  _DI_ unsigned unknown_pop() const;

  _DI_ ThreeBoard<N, W> force_orthogonal_horiz() const;
  _DI_ ThreeBoard<N, W> force_orthogonal_vert() const;
  _DI_ ThreeBoard<N, W> force_orthogonal() const { return force_orthogonal_horiz().force_orthogonal_vert(); }

  _DI_ BitBoard<W> eliminate_line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<W> seed);
  _DI_ void eliminate_all_lines() { eliminate_all_lines(knownOn); }
  _DI_ void propagate();

  template<Axis d>
  _DI_ void soft_branch(unsigned row);
  _DI_ void soft_branch_all();

  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_row() const;
  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_col() const;
  _DI_ cuda::std::pair<Axis, unsigned> most_constrained() const;
};

// bounds implementation
template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::bounds() {
  if constexpr (W == 64) {
    uint32_t row_bound_x = N >= 32 ? (~0) : (1 << N) - 1;
    uint32_t row_bound_y = N >= 32 ? (1 << (N - 32)) - 1 : 0;
    bool has_half = (threadIdx.x & 31) < ((N + 1) >> 1);
    bool has_full = (threadIdx.x & 31) < (N >> 1);
    BitBoard<W> result;
    result.state.x = has_half ? row_bound_x : 0;
    result.state.y = has_half ? row_bound_y : 0;
    result.state.z = has_full ? row_bound_x : 0;
    result.state.w = has_full ? row_bound_y : 0;
    return result;
  } else {
    uint32_t row_bound = N >= 32 ? (~0) : (1 << N) - 1;
    bool has_row = (threadIdx.x & 31) < N;
    BitBoard<W> result;
    result.state = has_row ? row_bound : 0;
    return result;
  }
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoard<N, W>::consistent() const {
  return (knownOn & knownOff).empty();
}

template <unsigned N, unsigned W>
_DI_ unsigned ThreeBoard<N, W>::unknown_pop() const {
  return N*N - (knownOn | knownOff).pop();
}

template <unsigned N, unsigned W>
_DI_ ThreeBoard<N, W> ThreeBoard<N, W>::force_orthogonal_horiz() const {
  ThreeBoard<N, W> result = *this;

  if constexpr (W == 64) {
    int on_pop_x = __popc(knownOn.state.x) + __popc(knownOn.state.y);
    if(on_pop_x == 2) {
      result.knownOff.state.x = ~knownOn.state.x;
      result.knownOff.state.y = ~knownOn.state.y;
    }
    if(on_pop_x > 2) {
      result.knownOn = BitBoard<W>::solid();
      result.knownOff = BitBoard<W>::solid();
    }

    int on_pop_z = __popc(knownOn.state.z) + __popc(knownOn.state.w);
    if(on_pop_z == 2) {
      result.knownOff.state.z = ~knownOn.state.z;
      result.knownOff.state.w = ~knownOn.state.w;
    }
    if(on_pop_z > 2) {
      result.knownOn = BitBoard<W>::solid();
      result.knownOff = BitBoard<W>::solid();
    }

    int off_pop_x = __popc(knownOff.state.x) + __popc(knownOff.state.y);
    if(off_pop_x == N - 2) {
      result.knownOn.state.x = ~knownOff.state.x;
      result.knownOn.state.y = ~knownOff.state.y;
    }
    if(off_pop_x > N - 2) {
      result.knownOn = BitBoard<W>::solid();
      result.knownOff = BitBoard<W>::solid();
    }

    int off_pop_z = __popc(knownOff.state.z) + __popc(knownOff.state.w);
    if(off_pop_z == N - 2) {
      result.knownOn.state.z = ~knownOff.state.z;
      result.knownOn.state.w = ~knownOff.state.w;
    }
    if(off_pop_z > N - 2) {
      result.knownOn = BitBoard<W>::solid();
      result.knownOff = BitBoard<W>::solid();
    }
  } else {
    int on_pop = __popc(knownOn.state);
    if(on_pop == 2) {
      result.knownOff.state = ~knownOn.state;
    }
    if(on_pop > 2) {
      result.knownOn = BitBoard<W>::solid();
      result.knownOff = BitBoard<W>::solid();
    }

    int off_pop = __popc(knownOff.state);
    if(off_pop == N - 2) {
      result.knownOn.state = ~knownOff.state;
    }
    if(off_pop > N - 2) {
      result.knownOn = BitBoard<W>::solid();
      result.knownOff = BitBoard<W>::solid();
    }
  }

  const BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  result.knownOn &= bds;
  result.knownOff &= bds;

  return result;
}

struct BinaryCount {
  uint32_t bit0;
  uint32_t bit1;
  uint32_t overflow;

  _DI_ BinaryCount operator+(const BinaryCount other) const {
    const uint32_t out0 = bit0 ^ other.bit0;
    const uint32_t carry0 = bit0 & other.bit0;

    const uint32_t out1 = bit1 ^ other.bit1 ^ carry0;
    const uint32_t carry1 = (bit1 & other.bit1) | (carry0 & (bit1 | other.bit1));
    const uint32_t out_overflow = carry1 | overflow | other.overflow;

    return {out0, out1, out_overflow};
  }
  _DI_ void operator+=(const BinaryCount other) { *this = *this + other; };
};

_DI_ BinaryCount count_vertically(const uint32_t value) {
  BinaryCount result = {value, 0, 0};

  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    BinaryCount other;
    other.bit0 = __shfl_down_sync(0xffffffff, result.bit0, offset);
    other.bit1 = __shfl_down_sync(0xffffffff, result.bit1, offset);
    other.overflow = __shfl_down_sync(0xffffffff, result.overflow, offset);

    result += other;
  }

  result.bit0 = __shfl_sync(0xffffffff, result.bit0, 0);
  result.bit1 = __shfl_sync(0xffffffff, result.bit1, 0);
  result.overflow = __shfl_sync(0xffffffff, result.overflow, 0);

  return result;
}

// force_orthogonal_vert implementation
template <unsigned N, unsigned W>
_DI_ ThreeBoard<N, W> ThreeBoard<N, W>::force_orthogonal_vert() const {
  ThreeBoard<N, W> result = *this;

  if constexpr (W == 64) {
    const BinaryCount on_count_xz = count_vertically(knownOn.state.x) + count_vertically(knownOn.state.z);
    const uint32_t on_count_xz_eq_2 = ~on_count_xz.overflow & on_count_xz.bit1 & ~on_count_xz.bit0;
    result.knownOff.state.x |= ~knownOn.state.x & on_count_xz_eq_2;
    result.knownOff.state.z |= ~knownOn.state.z & on_count_xz_eq_2;

    const uint32_t on_count_xz_gt_2 = on_count_xz.overflow | (on_count_xz.bit1 & on_count_xz.bit0);
    result.knownOn.state.x |= on_count_xz_gt_2;
    result.knownOff.state.x |= on_count_xz_gt_2;

    const BinaryCount on_count_yw = count_vertically(knownOn.state.y) + count_vertically(knownOn.state.w);
    const uint32_t on_count_yw_eq_2 = ~on_count_yw.overflow & on_count_yw.bit1 & ~on_count_yw.bit0;
    result.knownOff.state.y |= ~knownOn.state.y & on_count_yw_eq_2;
    result.knownOff.state.w |= ~knownOn.state.w & on_count_yw_eq_2;

    const uint32_t on_count_yw_gt_2 = on_count_yw.overflow | (on_count_yw.bit1 & on_count_yw.bit0);
    result.knownOn.state.y |= on_count_yw_gt_2;
    result.knownOff.state.y |= on_count_yw_gt_2;

    BitBoard<W> notKnownOff = ~knownOff & ThreeBoard<N, W>::bounds();

    const BinaryCount not_off_count_xz = count_vertically(notKnownOff.state.x) + count_vertically(notKnownOff.state.z);
    const uint32_t not_off_count_xz_eq_2 = ~not_off_count_xz.overflow & not_off_count_xz.bit1 & ~not_off_count_xz.bit0;
    result.knownOn.state.x |= ~knownOff.state.x & not_off_count_xz_eq_2;
    result.knownOn.state.z |= ~knownOff.state.z & not_off_count_xz_eq_2;

    const uint32_t not_off_count_xz_lt_2 = ~not_off_count_xz.overflow & ~not_off_count_xz.bit1;
    result.knownOn.state.x |= not_off_count_xz_lt_2;
    result.knownOff.state.x |= not_off_count_xz_lt_2;

    const BinaryCount not_off_count_yw = count_vertically(notKnownOff.state.y) + count_vertically(notKnownOff.state.w);
    const uint32_t not_off_count_yw_eq_2 = ~not_off_count_yw.overflow & not_off_count_yw.bit1 & ~not_off_count_yw.bit0;
    result.knownOn.state.y |= ~knownOff.state.y & not_off_count_yw_eq_2;
    result.knownOn.state.w |= ~knownOff.state.w & not_off_count_yw_eq_2;

    const uint32_t not_off_count_yw_lt_2 = ~not_off_count_yw.overflow & ~not_off_count_yw.bit1;
    result.knownOn.state.y |= not_off_count_yw_lt_2;
    result.knownOff.state.y |= not_off_count_yw_lt_2;
  } else {
    const BinaryCount on_count = count_vertically(knownOn.state);
    const uint32_t on_count_eq_2 = ~on_count.overflow & on_count.bit1 & ~on_count.bit0;
    result.knownOff.state |= ~knownOn.state & on_count_eq_2;

    const uint32_t on_count_gt_2 = on_count.overflow | (on_count.bit1 & on_count.bit0);
    result.knownOn.state |= on_count_gt_2;
    result.knownOff.state |= on_count_gt_2;

    BitBoard<W> notKnownOff = ~knownOff & ThreeBoard<N, W>::bounds();

    const BinaryCount not_off_count = count_vertically(notKnownOff.state);
    const uint32_t not_off_count_eq_2 = ~not_off_count.overflow & not_off_count.bit1 & ~not_off_count.bit0;
    result.knownOn.state |= ~knownOff.state & not_off_count_eq_2;

    const uint32_t not_off_count_lt_2 = ~not_off_count.overflow & ~not_off_count.bit1;
    result.knownOn.state |= not_off_count_lt_2;
    result.knownOff.state |= not_off_count_lt_2;
  }

  const BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  result.knownOn &= bds;
  result.knownOff &= bds;

  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W>
ThreeBoard<N, W>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                              cuda::std::pair<unsigned, unsigned> q) {
  if (p.first == q.first || p.second == q.second)
    return BitBoard<W>();

  BitBoard<W> line = BitBoard<W>::line(p, q);

  if constexpr (W == 64) {
    {
      unsigned row = 2*threadIdx.x;
      if (p.second == row || q.second == row) {
        line.state.x = 0;
        line.state.y = 0;
      }
    }

    {
      unsigned row = 2*threadIdx.x+1;
      if (p.second == row || q.second == row) {
        line.state.z = 0;
        line.state.w = 0;
      }
    }
  } else {
    unsigned row = threadIdx.x;
    if (p.second == row || q.second == row) {
      line.state = 0;
    }
  }
  
  return line;
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = knownOn;
  for (auto q = qs.first_on(); !qs.empty();
       qs.erase(q), q = qs.first_on()) {
    knownOff |= eliminate_line(p, q);
  }
  knownOff &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines(BitBoard<W> seed) {
  for (auto p = seed.first_on(); !seed.empty();
       seed.erase(p), p = seed.first_on()) {
    eliminate_all_lines(p);
  }
  knownOff &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoard<N, W>::propagate() {
  ThreeBoard<N, W> prev;

  do {
    prev = *this;

    auto forced_orthogonal = force_orthogonal();
    if(!consistent()) break;

    BitBoard<W> newOns = forced_orthogonal.knownOn & ~prev.knownOn;
    *this = forced_orthogonal;

    eliminate_all_lines(newOns);
  } while (*this != prev);
}

template <unsigned N, unsigned W>
template <Axis d>
_DI_ void ThreeBoard<N, W>::soft_branch<d>(unsigned r) {
  typename BitBoard<W>::row_t row_knownOn;
  if constexpr(d == Axis::Horizontal) {
    row_knownOn = knownOn.row(r);
  } else {
    row_knownOn = knownOn.column(r);
  }

  unsigned on_count;
  if constexpr (W == 64) {
    on_count = __popcll(row_knownOn);
  } else {
    on_count = __popc(row_knownOn);
  }
  if(on_count >= 2) return;

  typename BitBoard<W>::row_t row_knownOff;
  if constexpr(d == Axis::Horizontal) {
    row_knownOff = knownOff.row(r);
  } else {
    row_knownOff = knownOff.column(r);
  }

  unsigned off_count;
  if constexpr (W == 64) {
    off_count = __popcll(row_knownOff);
  } else {
    off_count = __popc(row_knownOff);
  }
  unsigned unknown_count = N - on_count - off_count;
  if (on_count == 1 && unknown_count > SOFT_BRANCH_1_THRESHOLD)
    return;
  if (on_count == 0 && unknown_count > SOFT_BRANCH_2_THRESHOLD)
    return;

  ThreeBoard<N, W> common(BitBoard<W>::solid(), BitBoard<W>::solid());

  typename BitBoard<W>::row_t remaining;
  remaining = ~row_knownOn & ~row_knownOff & (((typename BitBoard<W>::row_t)1 << N) - 1);

  if(on_count == 1) {
    for (; remaining; remaining &= remaining - 1) {
      unsigned c;
      if constexpr (W == 64) {
        c = __ffsll(remaining) - 1;
      } else {
        c = __ffs(remaining) - 1;
      }

      cuda::std::pair<unsigned, unsigned> cell;
      if constexpr (d == Axis::Horizontal)
        cell = {c, r};
      else
        cell = {r, c};

      ThreeBoard<N, W> subBoard = *this;
      subBoard.knownOn.set(cell);
      subBoard.eliminate_all_lines(cell);
      subBoard.propagate();
      if (!subBoard.consistent()) {
        knownOff.set(cell);
      } else {
        common.knownOn &= subBoard.knownOn;
        common.knownOff &= subBoard.knownOff;
      }
    }
  } else {
    for (; remaining; remaining &= remaining - 1) {
      unsigned c;
      if constexpr (W == 64) {
        c = __ffsll(remaining) - 1;
      } else {
        c = __ffs(remaining) - 1;
      }

      cuda::std::pair<unsigned, unsigned> cell;
      if constexpr (d == Axis::Horizontal)
        cell = {c, r};
      else
        cell = {r, c};

      ThreeBoard<N, W> subBoard = *this;
      subBoard.knownOn.set(cell);
      subBoard.eliminate_all_lines(cell);
      subBoard.propagate();

      if (!subBoard.consistent()) {
        knownOff.set(cell);
      } else {
        typename BitBoard<W>::row_t row_knownOff2;
        if constexpr(d == Axis::Horizontal) {
          row_knownOff2 = subBoard.knownOff.row(r);
        } else {
          row_knownOff2 = subBoard.knownOff.column(r);
        }

        typename BitBoard<W>::row_t remaining2;
        remaining2 = ~row_knownOff2 & (((typename BitBoard<W>::row_t)1 << N) - 1);

        for (; remaining2; remaining2 &= remaining2 - 1) {
          unsigned c2;
          if constexpr (W == 64) {
            c2 = __ffsll(remaining2) - 1;
          } else {
            c2 = __ffs(remaining2) - 1;
          }

          cuda::std::pair<unsigned, unsigned> cell2;
          if constexpr (d == Axis::Horizontal)
            cell2 = {c2, r};
          else
            cell2 = {r, c2};

          ThreeBoard<N, W> subBoard2 = *this;
          subBoard2.knownOn.set(cell2);
          subBoard2.propagate();

          if (!subBoard2.consistent()) {
            subBoard.knownOff.set(cell2);
          } else {
            common.knownOn &= subBoard2.knownOn;
            common.knownOff &= subBoard2.knownOff;
          }
        }
      }
    }
  }

  knownOn |= common.knownOn;
  knownOff |= common.knownOff;
}


// soft_branch_all implementation (combined)
template <unsigned N, unsigned W>
_DI_ void ThreeBoard<N, W>::soft_branch_all() {
  for (int r = 0; r < N; r++) {
    soft_branch<Axis::Horizontal>(r);
  }
  for (int r = 0; r < N; r++) {
    soft_branch<Axis::Vertical>(r);
  }
}

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<unsigned, unsigned>
ThreeBoard<N, W>::most_constrained_row() const {
  if constexpr (W == 64) {
    BitBoard<W> known = knownOn | knownOff;
    unsigned unknown_xy = N - __popc(known.state.x) + __popc(known.state.y);
    unsigned unknown_zw = N - __popc(known.state.z) + __popc(known.state.w);

    if(knownOn.state.x == 0 && knownOn.state.y == 0)
      unknown_xy = unknown_xy * (unknown_xy - 1);

    if(knownOn.state.z == 0 && knownOn.state.w == 0)
      unknown_zw = unknown_zw * (unknown_zw - 1);

    if (threadIdx.x * 2 >= N || unknown_xy == 0)
      unknown_xy = std::numeric_limits<unsigned>::max();
    if (threadIdx.x * 2 + 1 >= N || unknown_zw == 0)
      unknown_zw = std::numeric_limits<unsigned>::max();

    unsigned row;
    unsigned unknown;

    if (unknown_xy < unknown_zw) {
      row = threadIdx.x * 2;
      unknown = unknown_xy;
    } else {
      row = threadIdx.x * 2 + 1;
      unknown = unknown_zw;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      unsigned other_row = __shfl_down_sync(0xffffffff, row, offset);
      unsigned other_unknown = __shfl_down_sync(0xffffffff, unknown, offset);
      if (other_unknown < unknown) {
        row = other_row;
        unknown = other_unknown;
      }
    }

    row = __shfl_sync(0xffffffff, row, 0);
    unknown = __shfl_sync(0xffffffff, unknown, 0);

    return {row, unknown};
  } else {
    BitBoard<W> known = knownOn | knownOff;
    unsigned unknown = N - __popc(known.state);

    if(knownOn.state == 0)
      unknown = unknown * (unknown - 1);

    if (threadIdx.x >= N || unknown == 0)
      unknown = std::numeric_limits<unsigned>::max();

    unsigned row = threadIdx.x;

    for (int offset = 16; offset > 0; offset /= 2) {
      unsigned other_row = __shfl_down_sync(0xffffffff, row, offset);
      unsigned other_unknown = __shfl_down_sync(0xffffffff, unknown, offset);
      if (other_unknown < unknown) {
        row = other_row;
        unknown = other_unknown;
      }
    }

    row = __shfl_sync(0xffffffff, row, 0);
    unknown = __shfl_sync(0xffffffff, unknown, 0);

    return {row, unknown};
  }
}

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<unsigned, unsigned>
ThreeBoard<N, W>::most_constrained_col() const {
  unsigned best_col = 0;
  unsigned min_unknown = std::numeric_limits<unsigned>::max();

  for (unsigned c = 0; c < N; c++) {
    typename BitBoard<W>::row_t col_knownOn = knownOn.column(c);
    typename BitBoard<W>::row_t col_knownOff = knownOff.column(c);
    typename BitBoard<W>::row_t col_known = col_knownOn | col_knownOff;

    unsigned unknown;
    if constexpr (W == 64) {
      unknown = N - __popcll(col_known);
    } else {
      unknown = N - __popc(col_known);
    }

    if (col_knownOn == 0) {
      unknown = unknown * (unknown - 1);
    }

    if (unknown > 0 && unknown < min_unknown) {
      best_col = c;
      min_unknown = unknown;
    }
  }

  return {best_col, min_unknown};
}

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<Axis, unsigned>
ThreeBoard<N, W>::most_constrained() const {
  auto [row, row_unknown] = most_constrained_row();
  auto [col, col_unknown] = most_constrained_col();
  if (row_unknown < col_unknown)
    return {Axis::Horizontal, row};
  else
    return {Axis::Vertical, col};
}
