#pragma once

#include <stdint.h>
#include <cuda/std/utility>

#include "common.hpp"

template<unsigned W>
struct BitBoard {
  using state_t = std::conditional_t<W == 64, uint4, uint32_t>;
  using row_t = std::conditional_t<W == 64, uint64_t, uint32_t>;
  
  state_t state;

  _DI_ BitBoard() {
    if constexpr (W == 32) {
      state = 0;
    } else {
      state = {0, 0, 0, 0};
    }
  }
  
  _DI_ explicit BitBoard(state_t initial_state) : state(initial_state) {}
  
  _DI_ static BitBoard solid() {
    if constexpr (W == 32) {
      return BitBoard(~0U);
    } else {
      return BitBoard({~0U, ~0U, ~0U, ~0U});
    }
  }

  [[nodiscard]] _DI_ static BitBoard load(const row_t *data);
  _DI_ void save(row_t *data) const;

  _DI_ bool operator==(BitBoard other) const { return (*this ^ other).empty(); }
  _DI_ bool operator<(BitBoard other) const;

  _DI_ BitBoard operator~() const {
    if constexpr (W == 32) {
      return BitBoard(~state);
    } else {
      return BitBoard({~state.x, ~state.y, ~state.z, ~state.w});
    }
  }
  
  _DI_ BitBoard operator|(const BitBoard other) const {
    if constexpr (W == 32) {
      return BitBoard(state | other.state);
    } else {
      return BitBoard({state.x | other.state.x, state.y | other.state.y, state.z | other.state.z, state.w | other.state.w});
    }
  }
  
  _DI_ BitBoard operator&(const BitBoard other) const {
    if constexpr (W == 32) {
      return BitBoard(state & other.state);
    } else {
      return BitBoard({state.x & other.state.x, state.y & other.state.y, state.z & other.state.z, state.w & other.state.w});
    }
  }
  
  _DI_ BitBoard operator^(const BitBoard other) const {
    if constexpr (W == 32) {
      return BitBoard(state ^ other.state);
    } else {
      return BitBoard({state.x ^ other.state.x, state.y ^ other.state.y, state.z ^ other.state.z, state.w ^ other.state.w});
    }
  }
  
  _DI_ void operator|=(const BitBoard other) {
    if constexpr (W == 32) {
      state |= other.state;
    } else {
      state.x |= other.state.x; state.y |= other.state.y; state.z |= other.state.z; state.w |= other.state.w;
    }
  }
  
  _DI_ void operator&=(const BitBoard other) {
    if constexpr (W == 32) {
      state &= other.state;
    } else {
      state.x &= other.state.x; state.y &= other.state.y; state.z &= other.state.z; state.w &= other.state.w;
    }
  }
  
  _DI_ void operator^=(const BitBoard other) {
    if constexpr (W == 32) {
      state ^= other.state;
    } else {
      state.x ^= other.state.x; state.y ^= other.state.y; state.z ^= other.state.z; state.w ^= other.state.w;
    }
  }

  _DI_ row_t row(int y) const;
  _DI_ row_t column(int x) const;
  _DI_ bool get(int x, int y) const;
  _DI_ bool get(cuda::std::pair<int, int> cell) const { return get(cell.first, cell.second); }
  _DI_ void set(int x, int y);
  _DI_ void set(cuda::std::pair<int, int> cell) { set(cell.first, cell.second); }
  _DI_ void erase(int x, int y);
  _DI_ void erase(cuda::std::pair<int, int> cell) { erase(cell.first, cell.second); }

  _DI_ cuda::std::pair<int, int> first_on() const;

  _DI_ static BitBoard<W> positions_before(int x, int y);
  _DI_ static BitBoard<W> positions_before(cuda::std::pair<int, int> cell) { return positions_before(cell.first, cell.second); }

  _DI_ bool empty() const;
  _DI_ int pop() const;

  _DI_ BitBoard<W> rotate_torus(int x, int y) const;
  _DI_ BitBoard<W> rotate_torus(cuda::std::pair<int, int> cell) const { return rotate_torus(cell.first, cell.second); }
  _DI_ BitBoard<W> zoi() const;

  // These do *not* preserve the origin
  _DI_ BitBoard<W> rotate_90() const;
  _DI_ BitBoard<W> rotate_180() const;
  _DI_ BitBoard<W> rotate_270() const;
  _DI_ BitBoard<W> flip_horizontal() const;
  _DI_ BitBoard<W> flip_vertical() const;
  _DI_ BitBoard<W> flip_diagonal() const;
  _DI_ BitBoard<W> flip_anti_diagonal() const;

  _DI_ bool is_canonical() const;
  
  template<unsigned N>
  _DI_ bool is_canonical_subsquare() const;
};

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::load(const row_t *in) {
  if constexpr (W == 32) {
    return BitBoard(in[threadIdx.x & 31]);
  } else {
    const uint4 *u4ptr = (const uint4 *)in;
    uint4 result = u4ptr[threadIdx.x & 31];
    return BitBoard(result);
  }
}

template<unsigned W>
_DI_ void BitBoard<W>::save(row_t *out) const {
  if constexpr (W == 32) {
    out[threadIdx.x & 31] = state;
  } else {
    uint4 *u4ptr = (uint4 *)out;
    u4ptr[threadIdx.x & 31] = state;
  }
}

template<unsigned W>
_DI_ typename BitBoard<W>::row_t BitBoard<W>::row(int y) const {
  if constexpr (W == 32) {
    return __shfl_sync(0xffffffff, state, y);
  } else {
    int src = (y & 63) >> 1;

    if (y & 1) {
      uint32_t lo = __shfl_sync(0xffffffff, state.z, src);
      uint32_t hi = __shfl_sync(0xffffffff, state.w, src);
      return (uint64_t)hi << 32 | lo;
    } else {
      uint32_t lo = __shfl_sync(0xffffffff, state.x, src);
      uint32_t hi = __shfl_sync(0xffffffff, state.y, src);
      return (uint64_t)hi << 32 | lo;
    }
  }
}

template<unsigned W>
_DI_ typename BitBoard<W>::row_t BitBoard<W>::column(int x) const {
  if constexpr (W == 32) {
    return __ballot_sync(0xffffffff, state & (1<<x));
  } else {
    uint32_t xs, zs;
    if(x < 32) {
      xs = __ballot_sync(0xffffffff, state.x & (1<<x));
      zs = __ballot_sync(0xffffffff, state.z & (1<<x));
    } else {
      xs = __ballot_sync(0xffffffff, state.y & (1<<(x-32)));
      zs = __ballot_sync(0xffffffff, state.w & (1<<(x-32)));
    }

    static const uint64_t B[] = {0x0000FFFF0000FFFF, 0x00FF00FF00FF00FF, 0x0F0F0F0F0F0F0F0F, 0x3333333333333333, 0x5555555555555555};
    static const unsigned S[] = {16, 8, 4, 2, 1};

    uint64_t xsl = xs;
    uint64_t zsl = zs;

    for(unsigned i = 0; i < sizeof(B)/sizeof(B[0]); i++) {
      xsl = (xsl | (xsl << S[i])) & B[i];
      zsl = (zsl | (zsl << S[i])) & B[i];
    }

    return xsl | (zsl << 1);
  }
}

template<unsigned W>
_DI_ bool BitBoard<W>::get(int x, int y) const {
  row_t r = row(y);
  return (r & ((row_t)1 << x)) != 0;
}

template<unsigned W>
_DI_ void BitBoard<W>::set(int x, int y) {
  if constexpr (W == 32) {
    bool should_act = (threadIdx.x & 31) == y;
    unsigned int bit = 1u << (x & 31);

    state |= bit & (should_act ? 0xffffffff : 0);
  } else {
    bool should_act = (threadIdx.x & 31) == (y >> 1);
    unsigned int bit = 1u << (x & 31);

    state.x |= bit & (should_act && !(y & 1) && !(x & 32) ? 0xffffffff : 0);
    state.y |= bit & (should_act && !(y & 1) &&  (x & 32) ? 0xffffffff : 0);
    state.z |= bit & (should_act &&  (y & 1) && !(x & 32) ? 0xffffffff : 0);
    state.w |= bit & (should_act &&  (y & 1) &&  (x & 32) ? 0xffffffff : 0);
  }
}

template<unsigned W>
_DI_ void BitBoard<W>::erase(int x, int y) {
  if constexpr (W == 32) {
    bool should_act = (threadIdx.x & 31) == y;
    unsigned int bit = 1u << (x & 31);

    state &= ~(bit & (should_act ? 0xffffffff : 0));
  } else {
    bool should_act = (threadIdx.x & 31) == (y >> 1);
    unsigned int bit = 1u << (x & 31);

    state.x &= ~(bit & (should_act && !(y & 1) && !(x & 32) ? 0xffffffff : 0));
    state.y &= ~(bit & (should_act && !(y & 1) &&  (x & 32) ? 0xffffffff : 0));
    state.z &= ~(bit & (should_act &&  (y & 1) && !(x & 32) ? 0xffffffff : 0));
    state.w &= ~(bit & (should_act &&  (y & 1) &&  (x & 32) ? 0xffffffff : 0));
  }
}

template<unsigned W>
_DI_ cuda::std::pair<int, int> BitBoard<W>::first_on() const {
  if constexpr (W == 32) {
    unsigned x = find_first_set<32>(state);

    uint32_t mask = __ballot_sync(0xffffffff, state);
    unsigned first_lane = find_first_set<32>(mask);

    x = __shfl_sync(0xffffffff, x, first_lane);

    return {x, first_lane};
  } else {
    unsigned x_low = find_first_set<64>((uint64_t) state.y << 32 | state.x);
    unsigned x_high = find_first_set<64>((uint64_t) state.w << 32 | state.z);

    bool use_high = ((state.x | state.y) == 0);
    unsigned x = use_high ? x_high : x_low;

    unsigned y_base = (threadIdx.x & 31) << 1;
    unsigned y = y_base + (use_high ? 1 : 0);

    uint32_t mask = __ballot_sync(0xffffffff, state.x | state.y | state.z | state.w);
    unsigned first_lane = find_first_set<32>(mask);

    y = __shfl_sync(0xffffffff, y, first_lane);
    x = __shfl_sync(0xffffffff, x, first_lane);

    return {x, y};
  }
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::positions_before(int x, int y) {
  BitBoard<W> result;

  if constexpr (W == 32) {
    int my_row = threadIdx.x & 31;
    if (my_row < y) {
      result.state = 0xffffffff;
    } else if (my_row == y) {
      result.state = (x == 0) ? 0 : (1u << x) - 1;
    } else {
      result.state = 0;
    }
  } else {
    uint32_t full_mask = 0xffffffff;

    uint32_t row_mask_low = (x < 32) ? ((1u << x) - 1) : full_mask;
    uint32_t row_mask_high = (x < 32) ? 0 : ((1u << (x - 32)) - 1);

    int my_row = threadIdx.x & 31;
    bool before_row = (my_row * 2 < y);
    bool at_row_even = (my_row * 2 == y);
    bool at_row_odd = (my_row * 2 + 1 == y);

    result.state.x = before_row ? full_mask : (at_row_even ? row_mask_low : 0);
    result.state.y = before_row ? full_mask : (at_row_even ? row_mask_high : 0);
    result.state.z = (before_row || at_row_even) ? full_mask : (at_row_odd ? row_mask_low : 0);
    result.state.w = (before_row || at_row_even) ? full_mask : (at_row_odd ? row_mask_high : 0);
  }

  return result;
}

template<unsigned W>
_DI_ bool BitBoard<W>::empty() const {
  if constexpr (W == 32) {
    return __ballot_sync(0xffffffff, state) == 0;
  } else {
    return __ballot_sync(0xffffffff, state.x | state.y | state.z | state.w) == 0;
  }
}

template<unsigned W>
_DI_ int BitBoard<W>::pop() const {
  int val;
  if constexpr (W == 32) {
    val = popcount<32>(state);
  } else {
    val = popcount<32>(state.x) + popcount<32>(state.y) + popcount<32>(state.z) + popcount<32>(state.w);
  }
  return __reduce_add_sync(0xffffffff, val);
}


template<unsigned W>
_DI_ bool BitBoard<W>::operator<(BitBoard other) const {
  if constexpr (W == 32) {

    uint32_t lt = __ballot_sync(0xffffffff, state < other.state);
    uint32_t gt = __ballot_sync(0xffffffff, state > other.state);
    uint32_t diff = lt | gt;
    
    if (diff) {
      int first_diff = find_first_set<32>(diff);
      return __shfl_sync(0xffffffff, state < other.state, first_diff);
    }
    
    return false;
  } else {
    uint32_t x_lt = __ballot_sync(0xffffffff, state.x < other.state.x);
    uint32_t x_gt = __ballot_sync(0xffffffff, state.x > other.state.x);
    uint32_t x_diff = x_lt | x_gt;
    
    if (x_diff) {
      int first_diff = find_first_set<32>(x_diff);
      return __shfl_sync(0xffffffff, state.x < other.state.x, first_diff);
    }
    
    uint32_t y_lt = __ballot_sync(0xffffffff, state.y < other.state.y);
    uint32_t y_gt = __ballot_sync(0xffffffff, state.y > other.state.y);
    uint32_t y_diff = y_lt | y_gt;
    
    if (y_diff) {
      int first_diff = find_first_set<32>(y_diff);
      return __shfl_sync(0xffffffff, state.y < other.state.y, first_diff);
    }
    
    uint32_t z_lt = __ballot_sync(0xffffffff, state.z < other.state.z);
    uint32_t z_gt = __ballot_sync(0xffffffff, state.z > other.state.z);
    uint32_t z_diff = z_lt | z_gt;
    
    if (z_diff) {
      int first_diff = find_first_set<32>(z_diff);
      return __shfl_sync(0xffffffff, state.z < other.state.z, first_diff);
    }
    
    uint32_t w_lt = __ballot_sync(0xffffffff, state.w < other.state.w);
    uint32_t w_gt = __ballot_sync(0xffffffff, state.w > other.state.w);
    uint32_t w_diff = w_lt | w_gt;
    
    if (w_diff) {
      int first_diff = find_first_set<32>(w_diff);
      return __shfl_sync(0xffffffff, state.w < other.state.w, first_diff);
    }
    
    return false;
  }
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::rotate_torus(int rh, int rv) const {
  if constexpr (W == 32) {
    uint32_t t = state;
    if (rv & 31) {
      int otherthread = ((-rv) + threadIdx.x) & 31;
      t = __shfl_sync(0xffffffff, t, otherthread);
    }
    if (rh & 31) {
      int sa = rh & 31;
      t = (t << sa) | (t >> (32 - sa));
    }
    return BitBoard<W>(t);
  } else {
    uint4 t = state;
    if (rv & 63) {
      // translate vertically:
      uint4 d;
      d.x = (rv & 1) ? t.z : t.x;
      d.y = (rv & 1) ? t.w : t.y;
      d.z = (rv & 1) ? t.x : t.z;
      d.w = (rv & 1) ? t.y : t.w;
      int upperthread = (((-rv) >> 1) + threadIdx.x) & 31;
      int lowerthread = (((-rv + 1) >> 1) + threadIdx.x) & 31;
      t.x = __shfl_sync(0xffffffff, d.x, upperthread);
      t.y = __shfl_sync(0xffffffff, d.y, upperthread);
      t.z = __shfl_sync(0xffffffff, d.z, lowerthread);
      t.w = __shfl_sync(0xffffffff, d.w, lowerthread);
    }

    if (rh & 63) {
      // translate horizontally:
      uint4 d;
      d.x = (rh & 32) ? t.y : t.x;
      d.y = (rh & 32) ? t.x : t.y;
      d.z = (rh & 32) ? t.w : t.z;
      d.w = (rh & 32) ? t.z : t.w;
      int sa = rh & 31;
      t.x = (d.x << sa) | (d.y >> (32 - sa));
      t.y = (d.y << sa) | (d.x >> (32 - sa));
      t.z = (d.z << sa) | (d.w >> (32 - sa));
      t.w = (d.w << sa) | (d.z >> (32 - sa));
    }
    return BitBoard<W>(t);
  }
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::zoi() const {
  BitBoard<W> vert = rotate_torus(0, -1) | *this | rotate_torus(0, 1);
  return vert.rotate_torus(-1, 0) | vert | vert.rotate_torus(1, 0);
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::flip_horizontal() const {
  BitBoard<W> result;
  if constexpr (W == 32) {
    result.state = __brev(state);
  } else {
    result.state.x = __brev(state.y);
    result.state.y = __brev(state.x);
    result.state.z = __brev(state.w);
    result.state.w = __brev(state.z);
  }
  return result;
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::flip_vertical() const {
  BitBoard<W> result;
  if constexpr (W == 32) {
    int my_row = threadIdx.x & 31;
    int src_row = 31 - my_row;
    result.state = __shfl_sync(0xffffffff, state, src_row);
  } else {
    int my_row = threadIdx.x & 31;
    int src_row = 31 - my_row;
    
    result.state.x = __shfl_sync(0xffffffff, state.z, src_row);
    result.state.y = __shfl_sync(0xffffffff, state.w, src_row);
    result.state.z = __shfl_sync(0xffffffff, state.x, src_row);
    result.state.w = __shfl_sync(0xffffffff, state.y, src_row);
  }
  return result;
}

template<unsigned W>
_DI_ uint32_t shuffle_round(uint32_t a, uint32_t mask, unsigned shift) {
  unsigned lane_shift;
  if constexpr (W == 32) {
    lane_shift = shift;
  } else {
    lane_shift = shift >> 1;
  }

  uint32_t b = __shfl_xor_sync(0xffffffff, a, lane_shift);

  uint32_t c;
  if ((threadIdx.x & lane_shift) == 0) {
    c = b << shift;
  } else {
    mask = ~mask;
    c = b >> shift;
  }
  return (a & mask) | (c & ~mask);
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::flip_diagonal() const {
  // TODO: I think with some effort we could have a tensor core do
  // this for us in a single instruction.

  if constexpr (W == 32) {
    // Transpose bitmask.
    uint32_t result = state;
    result = shuffle_round<32>(result, 0x0000ffff, 16);
    result = shuffle_round<32>(result, 0x00ff00ff, 8);
    result = shuffle_round<32>(result, 0x0f0f0f0f, 4);
    result = shuffle_round<32>(result, 0x33333333, 2);
    result = shuffle_round<32>(result, 0x55555555, 1);

    return BitBoard<W>(result);
  } else {
    uint4 result = state;

    // The first round rearranges the uint4:
    {
      uint32_t other_x = __shfl_xor_sync(0xffffffff, state.x, 16);
      uint32_t other_y = __shfl_xor_sync(0xffffffff, state.y, 16);
      uint32_t other_z = __shfl_xor_sync(0xffffffff, state.z, 16);
      uint32_t other_w = __shfl_xor_sync(0xffffffff, state.w, 16);

      if((threadIdx.x & 16) == 0) {
        result.y = other_x;
        result.w = other_z;
      } else {
        result.x = other_y;
        result.z = other_w;
      }
    }

    // The middle rounds are the same as before
    result.x = shuffle_round<64>(result.x, 0x0000ffff, 16);
    result.x = shuffle_round<64>(result.x, 0x00ff00ff, 8);
    result.x = shuffle_round<64>(result.x, 0x0f0f0f0f, 4);
    result.x = shuffle_round<64>(result.x, 0x33333333, 2);
    result.y = shuffle_round<64>(result.y, 0x0000ffff, 16);
    result.y = shuffle_round<64>(result.y, 0x00ff00ff, 8);
    result.y = shuffle_round<64>(result.y, 0x0f0f0f0f, 4);
    result.y = shuffle_round<64>(result.y, 0x33333333, 2);
    result.z = shuffle_round<64>(result.z, 0x0000ffff, 16);
    result.z = shuffle_round<64>(result.z, 0x00ff00ff, 8);
    result.z = shuffle_round<64>(result.z, 0x0f0f0f0f, 4);
    result.z = shuffle_round<64>(result.z, 0x33333333, 2);
    result.w = shuffle_round<64>(result.w, 0x0000ffff, 16);
    result.w = shuffle_round<64>(result.w, 0x00ff00ff, 8);
    result.w = shuffle_round<64>(result.w, 0x0f0f0f0f, 4);
    result.w = shuffle_round<64>(result.w, 0x33333333, 2);

    // The last round happens within a single lane:
    {
      uint32_t mask = 0x55555555;
      uint32_t final_x = (result.x & mask) | ((result.z << 1) & ~mask);
      uint32_t final_z = (result.z & ~mask) | ((result.x >> 1) & mask);
      uint32_t final_y = (result.y & mask) | ((result.w << 1) & ~mask);
      uint32_t final_w = (result.w & ~mask) | ((result.y >> 1) & mask);
      result = {final_x, final_y, final_z, final_w};
    }

    return BitBoard<W>(result);
  }
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::rotate_90() const {
  return flip_diagonal().flip_vertical();
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::rotate_180() const {
  return flip_horizontal().flip_vertical();
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::rotate_270() const {
  return flip_diagonal().flip_horizontal();
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::flip_anti_diagonal() const {
  return rotate_180().flip_diagonal();
}

template<unsigned W>
_DI_ bool BitBoard<W>::is_canonical() const {
  BitBoard<W> flip_v = flip_vertical();
  if (flip_v < *this) return false;

  BitBoard<W> flip_h = flip_horizontal();
  if (flip_h < *this) return false;

  BitBoard<W> rot180 = flip_h.flip_vertical();
  if (rot180 < *this) return false;
  
  // Compute expensive diagonal flip only once
  BitBoard<W> diag = flip_diagonal();
  if (diag < *this) return false;
  
  BitBoard<W> rot90 = diag.flip_vertical();
  if (rot90 < *this) return false;
  
  BitBoard<W> rot270 = diag.flip_horizontal();
  if (rot270 < *this) return false;
  
  BitBoard<W> anti_diag = rot270.flip_vertical();
  if (anti_diag < *this) return false;
  
  return true;
}

template <unsigned W>
template <unsigned N>
_DI_ bool BitBoard<W>::is_canonical_subsquare<N>() const {
  BitBoard<W> flip_v = flip_vertical().rotate_torus(0, N);
  if (flip_v < *this) return false;

  BitBoard<W> flip_h = flip_horizontal().rotate_torus(N, 0);
  if (flip_h < *this) return false;

  BitBoard<W> rot180 = flip_h.flip_vertical().rotate_torus(0, N);
  if (rot180 < *this) return false;
  
  // Compute expensive diagonal flip once
  BitBoard<W> diag = flip_diagonal();
  if (diag < *this) return false;
  
  BitBoard<W> rot90 = diag.flip_vertical().rotate_torus(0, N);
  if (rot90 < *this) return false;
  
  BitBoard<W> rot270 = diag.flip_horizontal().rotate_torus(N, 0);
  if (rot270 < *this) return false;
  
  BitBoard<W> anti_diag = rot270.flip_vertical().rotate_torus(0, N);
  if (anti_diag < *this) return false;
  
  return true;
}
