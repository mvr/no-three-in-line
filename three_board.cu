#pragma once

#include "common.hpp"
#include "params.hpp"

#include "board.cu"

__device__ uint32_t *g_line_table_32 = nullptr;

template <unsigned N, unsigned W>
struct ThreeBoard {
  BitBoard<W> known_on;
  BitBoard<W> known_off;

  _DI_ ThreeBoard() : known_on{}, known_off{} {}
  _DI_ explicit ThreeBoard(BitBoard<W> known_on, BitBoard<W> known_off) : known_on{known_on}, known_off{known_off} {}

  _DI_ bool operator==(ThreeBoard<N, W> other) const { return (known_on == other.known_on) && (known_off == other.known_off); }
  _DI_ bool operator!=(ThreeBoard<N, W> other) const { return !(*this == other); }

  static _DI_ BitBoard<W> bounds();
  static _DI_ BitBoard<W> relevant_endpoint(cuda::std::pair<unsigned, unsigned> p);

  _DI_ bool consistent() const;
  _DI_ bool complete() const;
  _DI_ unsigned unknown_pop() const;
  _DI_ LexStatus is_canonical_orientation() const;

  _DI_ ThreeBoard<N, W> force_orthogonal_horiz() const;
  _DI_ ThreeBoard<N, W> force_orthogonal_vert() const;
  _DI_ ThreeBoard<N, W> force_orthogonal() const { return force_orthogonal_horiz().force_orthogonal_vert(); }

  _DI_ BitBoard<W> vulnerable() const;
  _DI_ BitBoard<W> semivulnerable() const;

  _DI_ BitBoard<W> eliminate_line_inner(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q, cuda::std::pair<unsigned, unsigned> delta);
  _DI_ BitBoard<W> eliminate_line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<W> seed);
  _DI_ void eliminate_all_lines() { eliminate_all_lines(known_on); }
  _DI_ void eliminate_all_lines_unfiltered(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines_unfiltered(BitBoard<W> seed);
  _DI_ void eliminate_all_lines_unfiltered() { eliminate_all_lines_unfiltered(known_on); }

  _DI_ void eliminate_one_hop(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_one_hop(BitBoard<W> seed);

  _DI_ void propagate();

  _DI_ void soft_branch_cell(cuda::std::pair<unsigned, unsigned> cell);
  _DI_ void soft_branch_cells(BitBoard<W> cells);
  template<Axis d>
  _DI_ void soft_branch(unsigned row);
  _DI_ void soft_branch_all();

  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_row() const;
  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_col() const;
  _DI_ cuda::std::pair<Axis, unsigned> most_constrained() const;
};

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::bounds() {
  if constexpr (W == 32) {
    uint32_t row_bound = N >= 32 ? (~0u) : (1u << N) - 1;
    bool has_row = (threadIdx.x & 31) < N;
    BitBoard<W> result;
    result.state = has_row ? row_bound : 0;
    return result;
  } else {
    uint32_t row_bound_x = N >= 32 ? (~0u) : (1u << N) - 1;
    uint32_t row_bound_y;
    if constexpr (N <= 32) {
      row_bound_y = 0;
    } else if constexpr (N >= 64) {
      row_bound_y = 0xffffffff;
    } else {
      row_bound_y = (1u << (N - 32)) - 1;
    }
    bool has_half = (threadIdx.x & 31) < ((N + 1) >> 1);
    bool has_full = (threadIdx.x & 31) < (N >> 1);
    BitBoard<W> result;
    result.state.x = has_half ? row_bound_x : 0;
    result.state.y = has_half ? row_bound_y : 0;
    result.state.z = has_full ? row_bound_x : 0;
    result.state.w = has_full ? row_bound_y : 0;
    return result;
  }
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::relevant_endpoint(cuda::std::pair<unsigned, unsigned> p) {
  if constexpr (W == 32) {
    uint64_t fullrow = relevant_endpoint_table[32-p.second+(threadIdx.x & 31)];
    uint32_t moved_row = fullrow >> (32-p.first); // And truncated
    return BitBoard<W>(moved_row);
  } else {
    BitBoard<W> result;

    // For row threadIdx.x * 2
    {
      unsigned row_idx = (64 - p.second + ((threadIdx.x & 31) * 2));
      uint64_t full_low_bits = relevant_endpoint_table_64[row_idx * 2];
      uint64_t full_high_bits = relevant_endpoint_table_64[row_idx * 2 + 1];
      if(p.first < 32) {
        // Origin ends up in state.x
        result.state.x = (full_low_bits >> (64 - p.first)) | (full_high_bits << p.first);
        result.state.y = full_high_bits >> (32 - p.first);
      } else {
        // Origin ends up in state.y
        result.state.x = full_low_bits >> (64 - p.first);
        result.state.y = (full_low_bits >> (64 - (p.first - 32))) | (full_high_bits << (p.first - 32));
      }
    }

    // For row threadIdx.x * 2 + 1
    {
      unsigned row_idx = 64 - p.second + ((threadIdx.x & 31) * 2 + 1);
      uint64_t full_low_bits = relevant_endpoint_table_64[row_idx * 2];
      uint64_t full_high_bits = relevant_endpoint_table_64[row_idx * 2 + 1];
      if(p.first < 32) {
        // Origin ends up in state.z
        result.state.z = (full_low_bits >> (64 - p.first)) | (full_high_bits << p.first);
        result.state.w = full_high_bits >> (32 - p.first);
      } else {
        // Origin ends up in state.w
        result.state.z = full_low_bits >> (64 - p.first);
        result.state.w = (full_low_bits >> (64 - (p.first - 32))) | (full_high_bits << (p.first - 32));
      }
    }

    return result;
  }
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoard<N, W>::consistent() const {
  return (known_on & known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoard<N, W>::complete() const {
  return (bounds() & ~known_on & ~known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ unsigned ThreeBoard<N, W>::unknown_pop() const {
  return N*N - (known_on | known_off).pop();
}

template<unsigned N, unsigned W>
_DI_ LexStatus compare_with_unknowns(const BitBoard<W> a_on, const BitBoard<W> a_off,
                                     const BitBoard<W> b_on, const BitBoard<W> b_off) {
  // We need to find the first differing bit position where:
  // - Both are known (not unknown)
  // - They differ in value

  BitBoard<W> a_unknown = ~(a_on | a_off);
  BitBoard<W> b_unknown = ~(b_on | b_off);

  BitBoard<W> both_known = ~(a_unknown | b_unknown);
  BitBoard<W> diff = (a_on ^ b_on) & both_known;

  if (diff.empty()) {
    // Check if any unknowns exist that could change the comparison
    BitBoard<W> critical_unknowns = (a_unknown | b_unknown) & ThreeBoard<N, W>::bounds();
    return critical_unknowns.empty() ? LexStatus::Equal : LexStatus::Unknown;
  }

  auto cell = diff.first_on();

  BitBoard<W> before_mask = BitBoard<W>::positions_before(cell) & ThreeBoard<N, W>::bounds();

  if (a_on.get(cell)) {
    // a = 1, b = 0 at first difference
    // But we need to check if there's an earlier unknown that could flip this

    BitBoard<W> critical_before = before_mask & (
        (a_unknown & b_on) |   // a unknown, b = 1 (could make a < b)
        (a_off & b_unknown) |  // a = 0, b unknown (could make a < b)
        (a_unknown & b_unknown) // both unknown (could become different)
    );

    return critical_before.empty() ? LexStatus::Greater : LexStatus::Unknown;
  } else {
    // a = 0, b = 1 at first difference

    BitBoard<W> critical_before = before_mask & (
        (a_unknown & b_off) |   // a unknown, b = 0 (could make a > b)
        (a_on & b_unknown) |    // a = 1, b unknown (could make a > b)
        (a_unknown & b_unknown) // both unknown (could become different)
    );

    return critical_before.empty() ? LexStatus::Less : LexStatus::Unknown;
  }
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoard<N, W>::is_canonical_orientation() const {
  bool any_unknown = false;
  LexStatus order;

  BitBoard<W> flip_v_on = known_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> flip_v_off = known_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<N, W>(known_on, known_off, flip_v_on, flip_v_off);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> flip_h_on = known_on.flip_horizontal().rotate_torus(N, 0);
  BitBoard<W> flip_h_off = known_off.flip_horizontal().rotate_torus(N, 0);
  order = compare_with_unknowns<N, W>(known_on, known_off, flip_h_on, flip_h_off);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> rot180_on = flip_h_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> rot180_off = flip_h_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<N, W>(known_on, known_off, rot180_on, rot180_off);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> diag_on = known_on.flip_diagonal();
  BitBoard<W> diag_off = known_off.flip_diagonal();
  order = compare_with_unknowns<N, W>(known_on, known_off, diag_on, diag_off);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> rot90_on = diag_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> rot90_off = diag_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<N, W>(known_on, known_off, rot90_on, rot90_off);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> rot270_on = diag_on.flip_horizontal().rotate_torus(N, 0);
  BitBoard<W> rot270_off = diag_off.flip_horizontal().rotate_torus(N, 0);
  order = compare_with_unknowns<N, W>(known_on, known_off, rot270_on, rot270_off);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> anti_diag_on = rot270_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> anti_diag_off = rot270_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<N, W>(known_on, known_off, anti_diag_on, anti_diag_off);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  if (any_unknown)
    return LexStatus::Unknown;

  return LexStatus::Less;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoard<N, W> ThreeBoard<N, W>::force_orthogonal_horiz() const {
  ThreeBoard<N, W> result = *this;

  if constexpr (W == 32) {
    int on_pop = popcount<32>(known_on.state);
    if(on_pop == 2) {
      result.known_off.state = ~known_on.state;
    }
    if(on_pop > 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int off_pop = popcount<32>(known_off.state);
    if(off_pop == N - 2) {
      result.known_on.state = ~known_off.state;
    }
    if(off_pop > N - 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }
  } else {
    int on_pop_x = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
    if(on_pop_x == 2) {
      result.known_off.state.x = ~known_on.state.x;
      result.known_off.state.y = ~known_on.state.y;
    }
    if(on_pop_x > 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int on_pop_z = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
    if(on_pop_z == 2) {
      result.known_off.state.z = ~known_on.state.z;
      result.known_off.state.w = ~known_on.state.w;
    }
    if(on_pop_z > 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int off_pop_x = popcount<32>(known_off.state.x) + popcount<32>(known_off.state.y);
    if(off_pop_x == N - 2) {
      result.known_on.state.x = ~known_off.state.x;
      result.known_on.state.y = ~known_off.state.y;
    }
    if(off_pop_x > N - 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int off_pop_z = popcount<32>(known_off.state.z) + popcount<32>(known_off.state.w);
    if(off_pop_z == N - 2) {
      result.known_on.state.z = ~known_off.state.z;
      result.known_on.state.w = ~known_off.state.w;
    }
    if(off_pop_z > N - 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }
  }

  const BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  result.known_on &= bds;
  result.known_off &= bds;

  return result;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoard<N, W> ThreeBoard<N, W>::force_orthogonal_vert() const {
  ThreeBoard<N, W> result = *this;

  if constexpr (W == 32) {
    const BinaryCountSaturating on_count = count_vertically_saturating<32>(known_on.state);
    const uint32_t on_count_eq_2 = on_count.bit1 & ~on_count.bit0;
    result.known_off.state |= ~known_on.state & on_count_eq_2;

    const uint32_t on_count_gt_2 = on_count.bit1 & on_count.bit0;
    result.known_on.state |= on_count_gt_2;
    result.known_off.state |= on_count_gt_2;

    BitBoard<W> notKnownOff = ~known_off & ThreeBoard<N, W>::bounds();

    const BinaryCountSaturating not_off_count = count_vertically_saturating<32>(notKnownOff.state);
    const uint32_t not_off_count_eq_2 = not_off_count.bit1 & ~not_off_count.bit0;
    result.known_on.state |= ~known_off.state & not_off_count_eq_2;

    const uint32_t not_off_count_lt_2 = ~not_off_count.bit1;
    result.known_on.state |= not_off_count_lt_2;
    result.known_off.state |= not_off_count_lt_2;
  } else {
    const BinaryCountSaturating on_count_xz = count_vertically_saturating<32>(known_on.state.x) + count_vertically_saturating<32>(known_on.state.z);
    const uint32_t on_count_xz_eq_2 = on_count_xz.bit1 & ~on_count_xz.bit0;
    result.known_off.state.x |= ~known_on.state.x & on_count_xz_eq_2;
    result.known_off.state.z |= ~known_on.state.z & on_count_xz_eq_2;

    const uint32_t on_count_xz_gt_2 = on_count_xz.bit1 & on_count_xz.bit0;
    result.known_on.state.x |= on_count_xz_gt_2;
    result.known_off.state.x |= on_count_xz_gt_2;

    const BinaryCountSaturating on_count_yw = count_vertically_saturating<32>(known_on.state.y) + count_vertically_saturating<32>(known_on.state.w);
    const uint32_t on_count_yw_eq_2 = on_count_yw.bit1 & ~on_count_yw.bit0;
    result.known_off.state.y |= ~known_on.state.y & on_count_yw_eq_2;
    result.known_off.state.w |= ~known_on.state.w & on_count_yw_eq_2;

    const uint32_t on_count_yw_gt_2 = on_count_yw.bit1 & on_count_yw.bit0;
    result.known_on.state.y |= on_count_yw_gt_2;
    result.known_off.state.y |= on_count_yw_gt_2;

    BitBoard<W> notKnownOff = ~known_off & ThreeBoard<N, W>::bounds();

    const BinaryCountSaturating not_off_count_xz = count_vertically_saturating<32>(notKnownOff.state.x) + count_vertically_saturating<32>(notKnownOff.state.z);
    const uint32_t not_off_count_xz_eq_2 = not_off_count_xz.bit1 & ~not_off_count_xz.bit0;
    result.known_on.state.x |= ~known_off.state.x & not_off_count_xz_eq_2;
    result.known_on.state.z |= ~known_off.state.z & not_off_count_xz_eq_2;

    const uint32_t not_off_count_xz_lt_2 = ~not_off_count_xz.bit1;
    result.known_on.state.x |= not_off_count_xz_lt_2;
    result.known_off.state.x |= not_off_count_xz_lt_2;

    const BinaryCountSaturating not_off_count_yw = count_vertically_saturating<32>(notKnownOff.state.y) + count_vertically_saturating<32>(notKnownOff.state.w);
    const uint32_t not_off_count_yw_eq_2 = not_off_count_yw.bit1 & ~not_off_count_yw.bit0;
    result.known_on.state.y |= ~known_off.state.y & not_off_count_yw_eq_2;
    result.known_on.state.w |= ~known_off.state.w & not_off_count_yw_eq_2;

    const uint32_t not_off_count_yw_lt_2 = ~not_off_count_yw.bit1;
    result.known_on.state.y |= not_off_count_yw_lt_2;
    result.known_off.state.y |= not_off_count_yw_lt_2;
  }

  const BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  result.known_on &= bds;
  result.known_off &= bds;

  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::vulnerable() const {
  BitBoard<W> result;

  if constexpr (W == 32) {
    // Vulnerable horizontally
    {
      unsigned on_pop = popcount<32>(known_on.state);
      unsigned off_pop = popcount<32>(known_off.state);
      unsigned unknown_pop = N - on_pop - off_pop;
      bool vulnerable_row =
        (on_pop == 1 && unknown_pop == 2) || (on_pop == 0 && unknown_pop == 3);

      if (vulnerable_row)
        result.state = ~(board_row_t<W>)0;
    }

    // Vulnerable vertically
    {
      const BinaryCount<32> on_count = count_vertically<32>(known_on.state);
      BitBoard<32> unknown = ~known_on & ~known_off & ThreeBoard<N, W>::bounds();
      const BinaryCount<32> unknown_count = count_vertically<32>(unknown.state);

      uint32_t vulnerable_column =
          (on_count.bit0 & ~on_count.bit1 & ~on_count.overflow &
           ~unknown_count.bit0 & unknown_count.bit1 & ~unknown_count.overflow)
        | (~on_count.bit0 & ~on_count.bit1 & ~on_count.overflow &
           unknown_count.bit0 & unknown_count.bit1 & ~unknown_count.overflow);

      result.state |= vulnerable_column;
    }
  } else {
    // Vulnerable horizontally
    {
      unsigned on_pop_xy = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
      unsigned off_pop_xy = popcount<32>(known_off.state.x) + popcount<32>(known_off.state.y);
      unsigned unknown_pop_xy = N - on_pop_xy - off_pop_xy;
      bool vulnerable_row_xy =
        (on_pop_xy == 1 && unknown_pop_xy == 2) || (on_pop_xy == 0 && unknown_pop_xy == 3);

      if (vulnerable_row_xy) {
        result.state.x = ~(board_row_t<W>)0;
        result.state.y = ~(board_row_t<W>)0;
      }

      unsigned on_pop_zw = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
      unsigned off_pop_zw = popcount<32>(known_off.state.z) + popcount<32>(known_off.state.w);
      unsigned unknown_pop_zw = N - on_pop_zw - off_pop_zw;
      bool vulnerable_row_zw =
        (on_pop_zw == 1 && unknown_pop_zw == 2) || (on_pop_zw == 0 && unknown_pop_zw == 3);

      if (vulnerable_row_zw) {
        result.state.z = ~(board_row_t<W>)0;
        result.state.w = ~(board_row_t<W>)0;
      }
    }

    // Vulnerable vertically
    {
      BitBoard<64> unknown = ~known_on & ~known_off & ThreeBoard<N, W>::bounds();

      const BinaryCount<32> on_count_xz = count_vertically<32>(known_on.state.x) + count_vertically<32>(known_on.state.z);
      const BinaryCount<32> unknown_count_xz = count_vertically<32>(unknown.state.x) + count_vertically<32>(unknown.state.z);

      uint32_t vulnerable_column_xz =
          (on_count_xz.bit0 & ~on_count_xz.bit1 & ~on_count_xz.overflow &
           ~unknown_count_xz.bit0 & unknown_count_xz.bit1 & ~unknown_count_xz.overflow)
        | (~on_count_xz.bit0 & ~on_count_xz.bit1 & ~on_count_xz.overflow &
           unknown_count_xz.bit0 & unknown_count_xz.bit1 & ~unknown_count_xz.overflow);

      result.state.x |= vulnerable_column_xz;
      result.state.z |= vulnerable_column_xz;

      const BinaryCount<32> on_count_yw = count_vertically<32>(known_on.state.y) + count_vertically<32>(known_on.state.w);
      const BinaryCount<32> unknown_count_yw = count_vertically<32>(unknown.state.y) + count_vertically<32>(unknown.state.w);

      uint32_t vulnerable_column_yw =
          (on_count_yw.bit0 & ~on_count_yw.bit1 & ~on_count_yw.overflow &
           ~unknown_count_yw.bit0 & unknown_count_yw.bit1 & ~unknown_count_yw.overflow)
        | (~on_count_yw.bit0 & ~on_count_yw.bit1 & ~on_count_yw.overflow &
           unknown_count_yw.bit0 & unknown_count_yw.bit1 & ~unknown_count_yw.overflow);

      result.state.y |= vulnerable_column_yw;
      result.state.w |= vulnerable_column_yw;
    }
  }

  result &= ~known_on & ~known_off & ThreeBoard<N, W>::bounds();

  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::semivulnerable() const {
  BitBoard<W> result;

  if constexpr (W == 32) {
    // Semivulnerable horizontally: 0 on, 4 unknown.
    {
      unsigned on_pop = popcount<32>(known_on.state);
      unsigned off_pop = popcount<32>(known_off.state);
      unsigned unknown_pop = N - on_pop - off_pop;
      bool semivuln_row = (on_pop == 0 && unknown_pop == 4);

      if (semivuln_row)
        result.state = ~(board_row_t<W>)0;
    }

    // Semivulnerable vertically: 0 on, 4 unknown.
    {
      BitBoard<32> unknown = ~known_on & ~known_off & ThreeBoard<N, W>::bounds();
      const BinaryCountSaturating3<32> on_count = count_vertically_saturating3<32>(known_on.state);
      const BinaryCountSaturating3<32> unknown_count = count_vertically_saturating3<32>(unknown.state);

      const uint32_t on_zero = ~(on_count.bit0 | on_count.bit1 | on_count.bit2);
      const uint32_t unknown_eq_4 = unknown_count.bit2 & ~unknown_count.bit1 & ~unknown_count.bit0;
      const uint32_t semivuln_column = on_zero & unknown_eq_4;

      result.state |= semivuln_column;
    }
  } else {
    // Semivulnerable horizontally: 0 on, 4 unknown.
    {
      unsigned on_pop_xy = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
      unsigned off_pop_xy = popcount<32>(known_off.state.x) + popcount<32>(known_off.state.y);
      unsigned unknown_pop_xy = N - on_pop_xy - off_pop_xy;
      bool semivuln_row_xy = (on_pop_xy == 0 && unknown_pop_xy == 4);

      if (semivuln_row_xy) {
        result.state.x = ~(board_row_t<W>)0;
        result.state.y = ~(board_row_t<W>)0;
      }

      unsigned on_pop_zw = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
      unsigned off_pop_zw = popcount<32>(known_off.state.z) + popcount<32>(known_off.state.w);
      unsigned unknown_pop_zw = N - on_pop_zw - off_pop_zw;
      bool semivuln_row_zw = (on_pop_zw == 0 && unknown_pop_zw == 4);

      if (semivuln_row_zw) {
        result.state.z = ~(board_row_t<W>)0;
        result.state.w = ~(board_row_t<W>)0;
      }
    }

    // Semivulnerable vertically: 0 on, 4 unknown.
    {
      BitBoard<64> unknown = ~known_on & ~known_off & ThreeBoard<N, W>::bounds();

      const BinaryCountSaturating3<32> on_count_xz =
          count_vertically_saturating3<32>(known_on.state.x) + count_vertically_saturating3<32>(known_on.state.z);
      const BinaryCountSaturating3<32> unknown_count_xz =
          count_vertically_saturating3<32>(unknown.state.x) + count_vertically_saturating3<32>(unknown.state.z);

      const uint32_t on_zero_xz = ~(on_count_xz.bit0 | on_count_xz.bit1 | on_count_xz.bit2);
      const uint32_t unknown_eq_4_xz = unknown_count_xz.bit2 & ~unknown_count_xz.bit1 & ~unknown_count_xz.bit0;
      const uint32_t semivuln_column_xz = on_zero_xz & unknown_eq_4_xz;

      result.state.x |= semivuln_column_xz;
      result.state.z |= semivuln_column_xz;

      const BinaryCountSaturating3<32> on_count_yw =
          count_vertically_saturating3<32>(known_on.state.y) + count_vertically_saturating3<32>(known_on.state.w);
      const BinaryCountSaturating3<32> unknown_count_yw =
          count_vertically_saturating3<32>(unknown.state.y) + count_vertically_saturating3<32>(unknown.state.w);

      const uint32_t on_zero_yw = ~(on_count_yw.bit0 | on_count_yw.bit1 | on_count_yw.bit2);
      const uint32_t unknown_eq_4_yw = unknown_count_yw.bit2 & ~unknown_count_yw.bit1 & ~unknown_count_yw.bit0;
      const uint32_t semivuln_column_yw = on_zero_yw & unknown_eq_4_yw;

      result.state.y |= semivuln_column_yw;
      result.state.w |= semivuln_column_yw;
    }
  }

  result &= ~known_on & ~known_off & ThreeBoard<N, W>::bounds();
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W>
ThreeBoard<N, W>::eliminate_line_inner(cuda::std::pair<unsigned, unsigned> p,
                                       cuda::std::pair<unsigned, unsigned> q,
                                       cuda::std::pair<unsigned, unsigned> delta) {
  BitBoard<W> result;

  unsigned p_quo = p.second / delta.second;
  unsigned p_rem = p.second % delta.second;

  if constexpr (W == 32) {
    unsigned row = threadIdx.x & 31;
    if (row % delta.second == p_rem) {
      int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
      if(col >= 0 && col < 32) result.state |= 1 << col;
    }
    if (p.second == row || q.second == row) {
      result.state = 0;
    }
  } else {
    {
      unsigned row = 2 * (threadIdx.x & 31);
      if (row % delta.second == p_rem) {
        int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
        if(col >= 0 && col < 32) result.state.x |= 1 << col;
        else if(col >= 32 && col < 64) result.state.y |= 1 << (col-32);
      }
      if (p.second == row || q.second == row) {
        result.state.x = 0;
        result.state.y = 0;
      }
    }

    {
      unsigned row = 2 * (threadIdx.x & 31) + 1;
      if (row % delta.second == p_rem) {
        int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
        if(col >= 0 && col < 32) result.state.z |= 1 << col;
        else if(col >= 32 && col < 64) result.state.w |= 1 << (col-32);
      }
      if (p.second == row || q.second == row) {
        result.state.z = 0;
        result.state.w = 0;
      }
    }
  }

  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W>
ThreeBoard<N, W>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                 cuda::std::pair<unsigned, unsigned> q) {
  if constexpr (W == 32) {
    constexpr unsigned cell_count = N * N;
    unsigned p_idx = p.second * N + p.first;
    unsigned q_idx = q.second * N + q.first;
    const uint32_t *entry = g_line_table_32 + (static_cast<size_t>(p_idx) * cell_count + q_idx) * LINE_TABLE_ROWS;
    const unsigned lane = threadIdx.x & 31;
    const uint32_t row = __ldg(entry + lane);
    return BitBoard<32>(row);
  } else {
    if (p.second > q.second)
      cuda::std::swap(p, q);

    cuda::std::pair<int, unsigned> delta = {(int)q.first - p.first, q.second - p.second};

    // Recall div_gcd_table[x][y] = x / gcd(x, y)
    const unsigned first_div = div_gcd_table[std::abs(delta.first)][delta.second];
    const unsigned second_div = div_gcd_table[delta.second][std::abs(delta.first)];
    delta.first = (delta.first < 0 ? -1 : 1) * first_div;
    delta.second = second_div;

    switch(delta.second) {
    case 1: return eliminate_line_inner(p, q, {delta.first, 1});
    case 2: return eliminate_line_inner(p, q, {delta.first, 2});
    case 4: return eliminate_line_inner(p, q, {delta.first, 4});
    default: return eliminate_line_inner(p, q, delta);
    }
  }
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = known_on & ThreeBoard<N, W>::relevant_endpoint(p);
  cuda::std::pair<int, int> q;
  while (qs.some_on_if_any(q)) {
    qs.erase(q);
    known_off |= eliminate_line(p, q);
    if (!consistent())
      return;
  }
  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines(BitBoard<W> ps) {
  cuda::std::pair<int, int> p;
  while (ps.some_on_if_any(p)) {
    ps.erase(p);

    BitBoard<W> qs = known_on & ~ps & ThreeBoard<N, W>::relevant_endpoint(p);

    cuda::std::pair<int, int> q;
    while (qs.some_on_if_any(q)) {
      qs.erase(q);
      known_off |= eliminate_line(p, q);
      if (!consistent())
        return;
    }
  }
  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines_unfiltered(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = known_on;
  qs.erase(p.first, p.second);

  cuda::std::pair<int, int> q;
  while (qs.some_on_if_any(q)) {
    qs.erase(q);

    if (p.first == q.first || p.second == q.second)
      continue;

    known_off |= eliminate_line(p, q);
    if (!consistent())
      return;
  }

  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines_unfiltered(BitBoard<W> ps) {
  cuda::std::pair<int, int> p;
  while (ps.some_on_if_any(p)) {
    ps.erase(p);

    BitBoard<W> qs = known_on;
    qs.erase(p);

    cuda::std::pair<int, int> q;
    while (qs.some_on_if_any(q)) {
      qs.erase(q);

      if (p.first == q.first || p.second == q.second)
        continue;

      known_off |= eliminate_line(p, q);
      if (!consistent())
        return;
    }
  }

  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_one_hop(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> to_eliminate = known_on.mirror_around(p);
  to_eliminate.erase_row(p.second);
  known_off |= to_eliminate;
  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_one_hop(BitBoard<W> ps) {
  cuda::std::pair<int, int> p;
  while (ps.some_on_if_any(p)) {
    BitBoard<W> to_eliminate = known_on.mirror_around(p);
    to_eliminate.erase_row(p.second);
    known_off |= to_eliminate;
    ps.erase(p);
  }

  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoard<N, W>::propagate() {
  ThreeBoard<N, W> prev;

  BitBoard<W> done_ons = known_on;

  do {
    do {
      prev = *this;
      *this = force_orthogonal();

      if(!consistent())
        return;

    } while(*this != prev);

    prev = *this;

    eliminate_all_lines(known_on & ~done_ons);

    if(!consistent())
        return;

    done_ons = known_on;
  } while (*this != prev);
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::soft_branch_cell(cuda::std::pair<unsigned, unsigned> cell) {
  ThreeBoard<N, W> common(BitBoard<W>::solid(), BitBoard<W>::solid());

  {
    ThreeBoard<N, W> subBoard = *this;
    subBoard.known_on.set(cell);
    subBoard.eliminate_all_lines(cell);
    subBoard.propagate();
    if (subBoard.consistent()) {
      common.known_on &= subBoard.known_on;
      common.known_off &= subBoard.known_off;
    }
  }
  {
    ThreeBoard<N, W> subBoard = *this;
    subBoard.known_off.set(cell);
    subBoard.propagate();
    if (subBoard.consistent()) {
      common.known_on &= subBoard.known_on;
      common.known_off &= subBoard.known_off;
    }
  }

  known_on |= common.known_on;
  known_off |= common.known_off;
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::soft_branch_cells(BitBoard<W> ps) {
  for (auto p = ps.first_on(); !ps.empty();
       ps.erase(p), p = ps.first_on()) {
    soft_branch_cell(p);

    if (!consistent())
      break;

    ps &= ~known_on & ~known_off;
  }
}

template <unsigned N, unsigned W>
template <Axis d>
_DI_ void ThreeBoard<N, W>::soft_branch<d>(unsigned r) {
  auto row_known_on = (d == Axis::Horizontal) ? known_on.row(r) : known_on.column(r);
  auto row_known_off = (d == Axis::Horizontal) ? known_off.row(r) : known_off.column(r);
  
  unsigned on_count = popcount<W>(row_known_on);
  if(on_count >= 2) return;

  unsigned off_count = popcount<W>(row_known_off);
  unsigned unknown_count = N - on_count - off_count;
  
  if (on_count == 1 && unknown_count > SOFT_BRANCH_1_THRESHOLD) return;
  if (on_count == 0 && unknown_count > SOFT_BRANCH_2_THRESHOLD) return;

  ThreeBoard<N, W> common(BitBoard<W>::solid(), BitBoard<W>::solid());
  board_row_t<W> remaining = ~row_known_on & ~row_known_off & (((board_row_t<W>)1 << N) - 1);

  auto make_cell = [&](unsigned c) {
    return (d == Axis::Horizontal) ? cuda::std::pair<unsigned, unsigned>{c, r} : cuda::std::pair<unsigned, unsigned>{r, c};
  };

  for (; remaining; remaining &= remaining - 1) {
    auto cell = make_cell(find_first_set<W>(remaining));

    ThreeBoard<N, W> subBoard = *this;
    subBoard.known_on.set(cell);
    subBoard.eliminate_all_lines(cell);
    subBoard.propagate();

    if (!subBoard.consistent()) {
      known_off.set(cell);
      continue;
    }

    if(on_count == 1) {
      common.known_on &= subBoard.known_on;
      common.known_off &= subBoard.known_off;
      continue;
    }

    auto row_known_on2 = (d == Axis::Horizontal) ? subBoard.known_on.row(r) : subBoard.known_on.column(r);
    auto row_known_off2 = (d == Axis::Horizontal) ? subBoard.known_off.row(r) : subBoard.known_off.column(r);
    board_row_t<W> remaining2 =
      remaining &
      ~row_known_on2 & ~row_known_off2 &
      ~((board_row_t<W>)1 << find_first_set<W>(remaining)) &
      (((board_row_t<W>)1 << N) - 1);

    if (remaining2 == 0) {
      common.known_on &= subBoard.known_on;
      common.known_off &= subBoard.known_off;
    }

    unsigned remaining2_count = popcount<W>(remaining2);

    for (; remaining2_count>1; remaining2_count--, remaining2 &= remaining2 - 1) {
      auto cell2 = make_cell(find_first_set<W>(remaining2));

      ThreeBoard<N, W> subBoard2 = subBoard;
      subBoard2.known_on.set(cell2);
      subBoard2.eliminate_all_lines(cell2);
      subBoard2.propagate();

      if (!subBoard2.consistent()) {
        subBoard.known_off.set(cell2);
      } else {
        common.known_on &= subBoard2.known_on;
        common.known_off &= subBoard2.known_off;
      }
    }
  }

  known_on |= common.known_on;
  known_off |= common.known_off;
}


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
  unsigned row;
  unsigned unknown;

  if constexpr (W == 32) {
    BitBoard<W> known = known_on | known_off;
    unknown = N - popcount<32>(known.state);

    if(known_on.state == 0) {
      unknown = unknown * (unknown - 1) / 2;
    } else {
      unknown *= ROW_SINGLE_ON_PENALTY;
    }

    if ((threadIdx.x & 31) >= N || unknown == 0)
      unknown = std::numeric_limits<unsigned>::max();

    row = (threadIdx.x & 31);
  } else {
    BitBoard<W> known = known_on | known_off;
    unsigned unknown_xy = N - popcount<32>(known.state.x) - popcount<32>(known.state.y);
    unsigned unknown_zw = N - popcount<32>(known.state.z) - popcount<32>(known.state.w);

    bool on_xy_empty = (known_on.state.x == 0 && known_on.state.y == 0);
    bool on_zw_empty = (known_on.state.z == 0 && known_on.state.w == 0);

    if(on_xy_empty) {
      unknown_xy = unknown_xy * (unknown_xy - 1) / 2;
    } else {
      unsigned on_pop_xy = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
      if (on_pop_xy == 1) {
        unknown_xy *= ROW_SINGLE_ON_PENALTY;
      }
    }

    if(on_zw_empty) {
      unknown_zw = unknown_zw * (unknown_zw - 1) / 2;
    } else {
      unsigned on_pop_zw = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
      if (on_pop_zw == 1) {
        unknown_zw *= ROW_SINGLE_ON_PENALTY;
      }
    }

    if ((threadIdx.x & 31) * 2 >= N || unknown_xy == 0)
      unknown_xy = std::numeric_limits<unsigned>::max();
    if ((threadIdx.x & 31) * 2 + 1 >= N || unknown_zw == 0)
      unknown_zw = std::numeric_limits<unsigned>::max();

    if (unknown_xy < unknown_zw) {
      row = (threadIdx.x & 31) * 2;
      unknown = unknown_xy;
    } else {
      row = (threadIdx.x & 31) * 2 + 1;
      unknown = unknown_zw;
    }
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
}

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<unsigned, unsigned>
ThreeBoard<N, W>::most_constrained_col() const {
  unsigned best_col = 0;
  unsigned min_unknown = std::numeric_limits<unsigned>::max();

  for (unsigned c = 0; c < N; c++) {
    board_row_t<W> col_known_on = known_on.column(c);
    board_row_t<W> col_known_off = known_off.column(c);
    board_row_t<W> col_known = col_known_on | col_known_off;

    unsigned unknown = N - popcount<W>(col_known);

    if (col_known_on == 0) {
      unknown = unknown * (unknown - 1) / 2;
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
