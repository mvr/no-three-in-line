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
    if constexpr (W == 64) {
      state = {0, 0, 0, 0};
    } else {
      state = 0;
    }
  }
  
  _DI_ explicit BitBoard(state_t initial_state) : state(initial_state) {}
  
  _DI_ static BitBoard solid() {
    if constexpr (W == 64) {
      return BitBoard({~0U, ~0U, ~0U, ~0U});
    } else {
      return BitBoard(~0U);
    }
  }

  [[nodiscard]] _DI_ static BitBoard load(const row_t *data);
  _DI_ void save(row_t *data) const;

  _DI_ bool operator==(BitBoard other) const { return (*this ^ other).empty(); }

  _DI_ BitBoard operator~() const {
    if constexpr (W == 64) {
      return BitBoard({~state.x, ~state.y, ~state.z, ~state.w});
    } else {
      return BitBoard(~state);
    }
  }
  
  _DI_ BitBoard operator|(const BitBoard other) const {
    if constexpr (W == 64) {
      return BitBoard({state.x | other.state.x, state.y | other.state.y, state.z | other.state.z, state.w | other.state.w});
    } else {
      return BitBoard(state | other.state);
    }
  }
  
  _DI_ BitBoard operator&(const BitBoard other) const {
    if constexpr (W == 64) {
      return BitBoard({state.x & other.state.x, state.y & other.state.y, state.z & other.state.z, state.w & other.state.w});
    } else {
      return BitBoard(state & other.state);
    }
  }
  
  _DI_ BitBoard operator^(const BitBoard other) const {
    if constexpr (W == 64) {
      return BitBoard({state.x ^ other.state.x, state.y ^ other.state.y, state.z ^ other.state.z, state.w ^ other.state.w});
    } else {
      return BitBoard(state ^ other.state);
    }
  }
  
  _DI_ void operator|=(const BitBoard other) {
    if constexpr (W == 64) {
      state.x |= other.state.x; state.y |= other.state.y; state.z |= other.state.z; state.w |= other.state.w;
    } else {
      state |= other.state;
    }
  }
  
  _DI_ void operator&=(const BitBoard other) {
    if constexpr (W == 64) {
      state.x &= other.state.x; state.y &= other.state.y; state.z &= other.state.z; state.w &= other.state.w;
    } else {
      state &= other.state;
    }
  }
  
  _DI_ void operator^=(const BitBoard other) {
    if constexpr (W == 64) {
      state.x ^= other.state.x; state.y ^= other.state.y; state.z ^= other.state.z; state.w ^= other.state.w;
    } else {
      state ^= other.state;
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

  _DI_ bool empty() const;
  _DI_ int pop() const;
};

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::load(const row_t *in) {
  if constexpr (W == 64) {
    const uint4 *u4ptr = (const uint4 *)in;
    uint4 result = u4ptr[threadIdx.x & 31];
    return BitBoard(result);
  } else {
    return BitBoard(in[threadIdx.x & 31]);
  }
}

template<unsigned W>
_DI_ void BitBoard<W>::save(row_t *out) const {
  if constexpr (W == 64) {
    uint4 *u4ptr = (uint4 *)out;
    u4ptr[threadIdx.x & 31] = state;
  } else {
    out[threadIdx.x & 31] = state;
  }
}

template<unsigned W>
_DI_ typename BitBoard<W>::row_t BitBoard<W>::row(int y) const {
  if constexpr (W == 64) {
    int src = (y & 63) >> 1;

    if (y & 1) {
      uint32_t lo = __shfl_sync(0xffffffffu, state.z, src);
      uint32_t hi = __shfl_sync(0xffffffffu, state.w, src);
      return (uint64_t)hi << 32 | lo;
    } else {
      uint32_t lo = __shfl_sync(0xffffffffu, state.x, src);
      uint32_t hi = __shfl_sync(0xffffffffu, state.y, src);
      return (uint64_t)hi << 32 | lo;
    }
  } else {
    return __shfl_sync(0xffffffffu, state, y);
  }
}

template<unsigned W>
_DI_ typename BitBoard<W>::row_t BitBoard<W>::column(int x) const {
  if constexpr (W == 64) {
    uint32_t xs, zs;
    if(x < 32) {
      xs = __ballot_sync(0xffffffffu, state.x & (1<<x));
      zs = __ballot_sync(0xffffffffu, state.z & (1<<x));
    } else {
      xs = __ballot_sync(0xffffffffu, state.y & (1<<(x-32)));
      zs = __ballot_sync(0xffffffffu, state.w & (1<<(x-32)));
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
  } else {
    return __ballot_sync(0xffffffffu, state & (1<<x));
  }
}

template<unsigned W>
_DI_ bool BitBoard<W>::get(int x, int y) const {
  row_t r = row(y);
  return (r & ((row_t)1 << x)) != 0;
}

template<unsigned W>
_DI_ void BitBoard<W>::set(int x, int y) {
  if constexpr (W == 64) {
    bool should_act = (threadIdx.x & 31) == (y >> 1);
    unsigned int bit = 1u << (x & 31);

    state.x |= bit & (should_act && !(y & 1) && !(x & 32) ? 0xFFFFFFFF : 0);
    state.y |= bit & (should_act && !(y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0);
    state.z |= bit & (should_act &&  (y & 1) && !(x & 32) ? 0xFFFFFFFF : 0);
    state.w |= bit & (should_act &&  (y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0);
  } else {
    bool should_act = (threadIdx.x & 31) == y;
    unsigned int bit = 1u << (x & 31);

    if(should_act)
      state |= bit;
  }
}

template<unsigned W>
_DI_ void BitBoard<W>::erase(int x, int y) {
  if constexpr (W == 64) {
    bool should_act = (threadIdx.x & 31) == (y >> 1);
    unsigned int bit = 1u << (x & 31);

    state.x &= ~(bit & (should_act && !(y & 1) && !(x & 32) ? 0xFFFFFFFF : 0));
    state.y &= ~(bit & (should_act && !(y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0));
    state.z &= ~(bit & (should_act &&  (y & 1) && !(x & 32) ? 0xFFFFFFFF : 0));
    state.w &= ~(bit & (should_act &&  (y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0));
  } else {
    bool should_act = (threadIdx.x & 31) == y;
    unsigned int bit = 1u << (x & 31);

    if(should_act)
      state &= ~bit;
  }
}

template<unsigned W>
_DI_ cuda::std::pair<int, int> BitBoard<W>::first_on() const {
  if constexpr (W == 64) {
    int x_low = __ffsll((uint64_t) state.y << 32 | state.x) - 1;
    int x_high = __ffsll((uint64_t) state.w << 32 | state.z) - 1;

    bool use_high = ((state.x | state.y) == 0);
    int x = use_high ? x_high : x_low;

    int y_base = (threadIdx.x & 31) << 1;
    int y = y_base + (use_high ? 1 : 0);

    uint32_t mask = __ballot_sync(0xffffffffu, state.x | state.y | state.z | state.w);
    int first_lane = __ffs(mask) - 1;

    y = __shfl_sync(0xffffffff, y, first_lane);
    x = __shfl_sync(0xffffffff, x, first_lane);

    return {x, y};
  } else {
    int x = __ffsll(state) - 1;
    int y = threadIdx.x;

    uint32_t mask = __ballot_sync(0xffffffffu, state);
    int first_lane = __ffs(mask) - 1;

    y = __shfl_sync(0xffffffff, y, first_lane);
    x = __shfl_sync(0xffffffff, x, first_lane);

    return {x, y};
  }
}

template<unsigned W>
_DI_ bool BitBoard<W>::empty() const {
  if constexpr (W == 64) {
    return __ballot_sync(0xffffffffu, state.x | state.y | state.z | state.w) == 0;
  } else {
    return __ballot_sync(0xffffffffu, state) == 0;
  }
}

template<unsigned W>
_DI_ int BitBoard<W>::pop() const {
  int val;
  if constexpr (W == 64) {
    val = __popc(state.x) + __popc(state.y) + __popc(state.z) + __popc(state.w);
  } else {
    val = __popc(state);
  }
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return __shfl_sync(0xffffffff, val, 0);
}


