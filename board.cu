#pragma once

#include <stdint.h>
#include <cuda/std/utility>
#include <limits>
#include <cmath>

#include "common.hpp"
#include "lookup_tables.cuh"

template<unsigned W>
struct BitBoard {
  board_state_t<W> state;

  _DI_ BitBoard() {
    if constexpr (W == 32) {
      state = 0;
    } else {
      state = {0, 0, 0, 0};
    }
  }
  
  _DI_ explicit BitBoard(board_state_t<W> initial_state) : state(initial_state) {}
  
  _DI_ static BitBoard solid() {
    if constexpr (W == 32) {
      return BitBoard(~0U);
    } else {
      return BitBoard({~0U, ~0U, ~0U, ~0U});
    }
  }

  // Rectangle anchored at origin with width w and height h.
  [[nodiscard]] _DI_ static BitBoard rect(unsigned w, unsigned h);

  [[nodiscard]] _DI_ static BitBoard load(const board_row_t<W> *data);
  _DI_ void save(board_row_t<W> *data) const;

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

  _DI_ board_row_t<W> row(int y) const;
  _DI_ board_row_t<W> column(int x) const;
  _DI_ unsigned column_pop(int x) const;
  _DI_ bool get(int x, int y) const;
  _DI_ bool get(cuda::std::pair<int, int> cell) const { return get(cell.first, cell.second); }
  _DI_ void set(int x, int y);
  _DI_ void set(cuda::std::pair<int, int> cell) { set(cell.first, cell.second); }
  _DI_ void erase(int x, int y);
  _DI_ void erase(cuda::std::pair<int, int> cell) { erase(cell.first, cell.second); }
  _DI_ void erase_row(int y);

  _DI_ cuda::std::pair<int, int> first_on() const;
  _DI_ board_row_t<W> first_row() const;
  _DI_ board_row_t<W> occupied_columns() const;
  _DI_ cuda::std::pair<int, int> some_on() const;
  _DI_ bool pop_on_if_any(cuda::std::pair<int, int> &out);
  _DI_ void on_cells(cuda::std::pair<uint8_t, uint8_t> cells[]) const;

  _DI_ static BitBoard<W> positions_before(int x, int y);
  _DI_ static BitBoard<W> positions_before(cuda::std::pair<int, int> cell) { return positions_before(cell.first, cell.second); }

  _DI_ bool empty() const;
  _DI_ int pop() const;

  _DI_ BitBoard<W> move(int x, int y) const;
  _DI_ BitBoard<W> move(cuda::std::pair<int, int> cell) const { return move(cell.first, cell.second); }
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

  _DI_ BitBoard<W> mirror_around(cuda::std::pair<int, int> cell) const;
  // Pull each set (x, y) down to (x/gcd(x,y), y/gcd(x,y))
  _DI_ BitBoard<W> gcd_reduce() const;


  template<unsigned N>
  _DI_ cuda::std::pair<int, int> first_center_on() const;
  template<unsigned N>
  _DI_ cuda::std::pair<int, int> first_origin_on() const;

  _DI_ void print() const;
};

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::load(const board_row_t<W> *in) {
  if constexpr (W == 32) {
    return BitBoard(in[threadIdx.x & 31]);
  } else {
    const uint4 *u4ptr = (const uint4 *)in;
    uint4 result = u4ptr[threadIdx.x & 31];
    return BitBoard(result);
  }
}

template<unsigned W>
_DI_ void BitBoard<W>::save(board_row_t<W> *out) const {
  if constexpr (W == 32) {
    out[threadIdx.x & 31] = state;
  } else {
    uint4 *u4ptr = (uint4 *)out;
    u4ptr[threadIdx.x & 31] = state;
  }
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::rect(unsigned w, unsigned h) {
  BitBoard<W> result;
  const unsigned lane = threadIdx.x & 31;
  const unsigned width = (w > W) ? W : w;

  if constexpr (W == 32) {
    board_row_t<32> row_mask;
    if (width >= 32) {
      row_mask = ~0u;
    } else if (width == 0) {
      row_mask = 0u;
    } else {
      row_mask = (board_row_t<32>(1) << width) - 1u;
    }
    result.state = (lane < h) ? row_mask : 0u;
  } else {
    board_row_t<64> row_mask;
    if (width >= 64) {
      row_mask = ~board_row_t<64>(0);
    } else if (width == 0) {
      row_mask = 0;
    } else {
      row_mask = (board_row_t<64>(1) << width) - 1;
    }

    const uint32_t lo = static_cast<uint32_t>(row_mask);
    const uint32_t hi = static_cast<uint32_t>(row_mask >> 32);
    const unsigned even_row = 2 * lane;
    const unsigned odd_row = even_row + 1;
    const bool has_even = even_row < h;
    const bool has_odd = odd_row < h;

    result.state.x = has_even ? lo : 0u;
    result.state.y = has_even ? hi : 0u;
    result.state.z = has_odd ? lo : 0u;
    result.state.w = has_odd ? hi : 0u;
  }
  return result;
}

template<unsigned W>
_DI_ board_row_t<W> BitBoard<W>::row(int y) const {
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
_DI_ board_row_t<W> BitBoard<W>::column(int x) const {
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

    return interleave32(xs, zs);
  }
}

template<unsigned W>
_DI_ unsigned BitBoard<W>::column_pop(int x) const {
  if constexpr (W == 32) {
    const uint32_t col_bits = __ballot_sync(0xffffffff, (state & (1u << x)) != 0u);
    return popcount<32>(col_bits);
  } else {
    uint32_t even_rows;
    uint32_t odd_rows;
    if (x < 32) {
      const uint32_t bit = (1u << x);
      even_rows = __ballot_sync(0xffffffff, (state.x & bit) != 0u);
      odd_rows = __ballot_sync(0xffffffff, (state.z & bit) != 0u);
    } else {
      const uint32_t bit = (1u << (x - 32));
      even_rows = __ballot_sync(0xffffffff, (state.y & bit) != 0u);
      odd_rows = __ballot_sync(0xffffffff, (state.w & bit) != 0u);
    }
    return popcount<32>(even_rows) + popcount<32>(odd_rows);
  }
}

template<unsigned W>
_DI_ bool BitBoard<W>::get(int x, int y) const {
  board_row_t<W> r = row(y);
  return (r & ((board_row_t<W>)1 << x)) != 0;
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
_DI_ void BitBoard<W>::erase_row(int y) {
  if constexpr (W == 32) {
    bool should_act = (threadIdx.x & 31) == y;
    if (should_act)
      state = 0;
  } else {
    bool should_act = (threadIdx.x & 31) == (y >> 1);
    if (should_act && !(y & 1)) {
      state.x = 0;
      state.y = 0;
    }
    if (should_act && (y & 1)) {
      state.z = 0;
      state.w = 0;
    }
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
_DI_ board_row_t<W> BitBoard<W>::first_row() const {
  // TODO W=64
  uint32_t mask = __ballot_sync(0xffffffff, state);
  if (mask == 0)
    return 0;
  unsigned first_lane = find_first_set<32>(mask);
  return __shfl_sync(0xffffffff, state, first_lane);
}

template<unsigned W>
_DI_ board_row_t<W> BitBoard<W>::occupied_columns() const {
  // TODO W=64
  return __reduce_or_sync(0xffffffff, state);
}

template<unsigned W>
_DI_ cuda::std::pair<int, int> BitBoard<W>::some_on() const {
  if constexpr (W == 32) {
    unsigned x = find_last_set<32>(state);

    uint32_t mask = __ballot_sync(0xffffffff, state);
    unsigned first_lane = find_last_set<32>(mask);

    x = __shfl_sync(0xffffffff, x, first_lane);

    return {x, first_lane};
  } else {
    uint32_t lane_mask = state.x | state.y | state.z | state.w;
    uint32_t active_lanes = __ballot_sync(0xffffffff, lane_mask != 0);
    unsigned lane = find_last_set<32>(active_lanes);

    unsigned x = 0;
    unsigned y = 0;

    if ((threadIdx.x & 31) == lane) {
      unsigned y_base = lane << 1;
      uint64_t even = ((uint64_t)state.y << 32) | state.x;
      uint64_t odd = ((uint64_t)state.w << 32) | state.z;

      if (odd) {
        x = find_last_set<64>(odd);
        y = y_base + 1;
      } else {
        x = find_last_set<64>(even);
        y = y_base;
      }
    }

    x = __shfl_sync(0xffffffff, x, lane);
    y = __shfl_sync(0xffffffff, y, lane);

    return {x, y};
  }
}

template<unsigned W>
_DI_ bool BitBoard<W>::pop_on_if_any(cuda::std::pair<int, int> &out) {
  if constexpr (W == 32) {
    const uint32_t mask = __ballot_sync(0xffffffff, state);
    if (mask == 0) {
      return false;
    }

    const unsigned lane = find_last_set<32>(mask);
    const uint32_t lane_state = __shfl_sync(0xffffffff, state, lane);
    const unsigned x = find_last_set<32>(lane_state);

    if ((threadIdx.x & 31) == lane) {
      state &= ~(1u << x);
    }

    out = {x, lane};
    return true;
  } else {
    const uint32_t lane_mask = state.x | state.y | state.z | state.w;
    const uint32_t active_lanes = __ballot_sync(0xffffffff, lane_mask != 0);
    if (active_lanes == 0) {
      return false;
    }

    const unsigned lane = find_last_set<32>(active_lanes);
    unsigned x = 0;
    unsigned y = 0;

    if ((threadIdx.x & 31) == lane) {
      const unsigned y_base = lane << 1;
      const uint64_t even = (static_cast<uint64_t>(state.y) << 32) | state.x;
      const uint64_t odd = (static_cast<uint64_t>(state.w) << 32) | state.z;

      if (odd) {
        x = find_last_set<64>(odd);
        y = y_base + 1;
        const uint32_t bit = 1u << (x & 31);
        if (x < 32) {
          state.z &= ~bit;
        } else {
          state.w &= ~bit;
        }
      } else {
        x = find_last_set<64>(even);
        y = y_base;
        const uint32_t bit = 1u << (x & 31);
        if (x < 32) {
          state.x &= ~bit;
        } else {
          state.y &= ~bit;
        }
      }
    }

    x = __shfl_sync(0xffffffff, x, lane);
    y = __shfl_sync(0xffffffff, y, lane);

    out = {x, y};
    return true;
  }
}

template<unsigned W>
_DI_ void BitBoard<W>::on_cells(cuda::std::pair<uint8_t, uint8_t> cells[]) const {
  auto emit_row = [&](board_row_t<W> bits, unsigned y, unsigned &offset) {
    while (bits) {
      unsigned x = count_trailing_zeros<W>(bits);
      bits &= bits - 1;
      cells[offset] = {x, y};
      offset++;
    }
  };

  if constexpr (W == 32) {
    unsigned lane = threadIdx.x & 31;
    board_row_t<W> row_bits = state;
    unsigned row_count = popcount<W>(row_bits);

    unsigned prefix = row_count;
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
      unsigned other = __shfl_up_sync(0xffffffff, prefix, offset);
      if (lane >= offset) {
        prefix += other;
      }
    }

    unsigned start = prefix - row_count;
    emit_row(row_bits, lane, start);
  } else {
    unsigned lane = threadIdx.x & 31;
    board_row_t<W> even_bits = state.x | (static_cast<uint64_t>(state.y) << 32);
    board_row_t<W> odd_bits = state.z | (static_cast<uint64_t>(state.w) << 32);

    unsigned even_count = popcount<W>(even_bits);
    unsigned odd_count = popcount<W>(odd_bits);
    unsigned lane_total = even_count + odd_count;

    unsigned prefix = lane_total;
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
      unsigned other = __shfl_up_sync(0xffffffff, prefix, offset);
      if (lane >= offset) {
        prefix += other;
      }
    }

    unsigned start = prefix - lane_total;
    emit_row(even_bits, lane * 2, start);
    emit_row(odd_bits, lane * 2 + 1, start);
  }
}

template <unsigned W>
template <unsigned N>
_DI_ cuda::std::pair<int, int> BitBoard<W>::first_center_on() const {
  if constexpr (W == 32) {
    // Put the center at 0, 0, then shift at the end

    uint32_t right_shifted = state >> (N/2);
    int right_closest = find_first_set<32>(right_shifted);

    uint32_t left_shifted = __brev(state) >> (32 - (N/2));
    int left_closest = -(int)find_first_set<32>(left_shifted) - 1;

    int row = (int)(threadIdx.x & 31) - (int)(N/2);

    int col = N;
    if (right_shifted != 0 && std::abs(right_closest) < std::abs(col))
      col = right_closest;
    if (left_shifted != 0 && std::abs(left_closest) < std::abs(col))
      col = left_closest;

    unsigned dist2 = row * row + col * col;

    for (int offset = 16; offset > 0; offset /= 2) {
      int other_row = __shfl_down_sync(0xffffffff, row, offset);
      int other_col = __shfl_down_sync(0xffffffff, col, offset);
      unsigned other_dist2 = __shfl_down_sync(0xffffffff, dist2, offset);

      if (other_dist2 < dist2) {
        row = other_row;
        col = other_col;
        dist2 = other_dist2;
      }
    }

    row = __shfl_sync(0xffffffff, row, 0);
    col = __shfl_sync(0xffffffff, col, 0);

    return {col + (N/2), row + (N/2)};
  } else {
    constexpr unsigned center = N / 2;

    auto abs_int = [](int v) { return v < 0 ? -v : v; };

    struct Candidate {
      int row;
      int col;
      unsigned dist2;
    };

    auto make_candidate = [&](uint64_t bits, int row_index) {
      Candidate cand;
      cand.row = row_index - static_cast<int>(center);
      cand.col = static_cast<int>(N);

      if (bits) {
        if constexpr (center < 64) {
          uint64_t right_shifted = bits >> center;
          if (right_shifted) {
            int right = find_first_set<64>(right_shifted);
            if (abs_int(right) < abs_int(cand.col)) {
              cand.col = right;
            }
          }
        }

        if constexpr (center > 0) {
          uint64_t left_mask;
          if constexpr (center >= 64) {
            left_mask = bits;
          } else {
            left_mask = bits & ((1ULL << center) - 1ULL);
          }

          if (left_mask) {
            int left_index = 63 - __clzll(left_mask);
            int left_offset = left_index - static_cast<int>(center);
            if (abs_int(left_offset) < abs_int(cand.col)) {
              cand.col = left_offset;
            }
          }
        }
      }

      cand.dist2 = static_cast<unsigned>(cand.row * cand.row + cand.col * cand.col);
      return cand;
    };

    int lane = threadIdx.x & 31;
    unsigned y_base = lane << 1;

    uint64_t even_row_bits = (static_cast<uint64_t>(state.y) << 32) | state.x;
    uint64_t odd_row_bits = (static_cast<uint64_t>(state.w) << 32) | state.z;

    Candidate even = make_candidate(even_row_bits, y_base);
    Candidate odd = make_candidate(odd_row_bits, y_base + 1);
    Candidate best = (odd.dist2 < even.dist2) ? odd : even;

    int row = best.row;
    int col = best.col;
    unsigned dist2 = best.dist2;

    for (int offset = 16; offset > 0; offset /= 2) {
      int other_row = __shfl_down_sync(0xffffffff, row, offset);
      int other_col = __shfl_down_sync(0xffffffff, col, offset);
      unsigned other_dist2 = __shfl_down_sync(0xffffffff, dist2, offset);

      if (other_dist2 < dist2) {
        row = other_row;
        col = other_col;
        dist2 = other_dist2;
      }
    }

    row = __shfl_sync(0xffffffff, row, 0);
    col = __shfl_sync(0xffffffff, col, 0);

    return {col + static_cast<int>(center), row + static_cast<int>(center)};
  }
}

template <unsigned W>
template <unsigned N>
_DI_ cuda::std::pair<int, int> BitBoard<W>::first_origin_on() const {
  if constexpr (W == 32) {
    int row = threadIdx.x & 31;

    int col = static_cast<int>(N);
    unsigned dist2 = std::numeric_limits<unsigned>::max();

    if (state != 0) {
      col = find_first_set<W>(state);
      dist2 = static_cast<unsigned>(row * row + col * col);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      int other_row = __shfl_down_sync(0xffffffff, row, offset);
      int other_col = __shfl_down_sync(0xffffffff, col, offset);
      unsigned other_dist2 = __shfl_down_sync(0xffffffff, dist2, offset);

      bool take_other = other_dist2 < dist2 ||
                        (other_dist2 == dist2 &&
                         (other_row < row || (other_row == row && other_col < col)));

      if (take_other) {
        row = other_row;
        col = other_col;
        dist2 = other_dist2;
      }
    }

    row = __shfl_sync(0xffffffff, row, 0);
    col = __shfl_sync(0xffffffff, col, 0);
    dist2 = __shfl_sync(0xffffffff, dist2, 0);

    if (dist2 == std::numeric_limits<unsigned>::max())
      return {-1, -1};

    return {col, row};
  } else {
    constexpr uint64_t column_mask = []() {
      if constexpr (N == 0) {
        return 0ULL;
      } else if constexpr (N >= 64) {
        return ~0ULL;
      } else {
        return (1ULL << N) - 1ULL;
      }
    }();

    struct Candidate {
      int row;
      int col;
      unsigned dist2;
    };

    auto make_candidate = [&](uint64_t bits, int row_index) {
      Candidate cand{row_index, static_cast<int>(N), std::numeric_limits<unsigned>::max()};

      if (row_index < static_cast<int>(N)) {
        uint64_t masked = bits & column_mask;
        if (masked) {
          int col_index = find_first_set<W>(masked);
          cand.col = col_index;
          cand.dist2 = static_cast<unsigned>(row_index * row_index + col_index * col_index);
        }
      }

      return cand;
    };

    int lane = threadIdx.x & 31;
    unsigned y_base = lane << 1;

    uint64_t even_row_bits = (static_cast<uint64_t>(state.y) << 32) | state.x;
    uint64_t odd_row_bits = (static_cast<uint64_t>(state.w) << 32) | state.z;

    Candidate even = make_candidate(even_row_bits, y_base);
    Candidate odd = make_candidate(odd_row_bits, y_base + 1);

    auto choose_better = [](const Candidate &a, const Candidate &b) {
      if (b.dist2 < a.dist2)
        return b;
      if (b.dist2 == a.dist2) {
        if (b.row < a.row)
          return b;
        if (b.row == a.row && b.col < a.col)
          return b;
      }
      return a;
    };

    Candidate best = choose_better(even, odd);

    int row = best.row;
    int col = best.col;
    unsigned dist2 = best.dist2;

    for (int offset = 16; offset > 0; offset /= 2) {
      int other_row = __shfl_down_sync(0xffffffff, row, offset);
      int other_col = __shfl_down_sync(0xffffffff, col, offset);
      unsigned other_dist2 = __shfl_down_sync(0xffffffff, dist2, offset);

      bool take_other = other_dist2 < dist2 ||
                        (other_dist2 == dist2 &&
                         (other_row < row || (other_row == row && other_col < col)));

      if (take_other) {
        row = other_row;
        col = other_col;
        dist2 = other_dist2;
      }
    }

    row = __shfl_sync(0xffffffff, row, 0);
    col = __shfl_sync(0xffffffff, col, 0);
    dist2 = __shfl_sync(0xffffffff, dist2, 0);

    if (dist2 == std::numeric_limits<unsigned>::max())
      return {-1, -1};

    return {col, row};
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
    bool before_row = (my_row * 2 + 1 < y);
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
    return !__any_sync(0xffffffff, state);
  } else {
    return !__any_sync(0xffffffff, state.x | state.y | state.z | state.w);
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
_DI_ BitBoard<W> BitBoard<W>::move(int rh, int rv) const {
  if constexpr (W == 32) {
    const int lane = threadIdx.x & 31;
    const int src_lane = lane - rv;

    uint32_t row = __shfl_sync(0xffffffff, state, src_lane);
    if (src_lane < 0 || src_lane >= 32)
      row = 0;

    if (rh > 0) {
      if (rh >= 32)
        row = 0;
      else
        row <<= rh;
    } else if (rh < 0) {
      const unsigned shift = static_cast<unsigned>(-rh);
      if (shift >= 32)
        row = 0;
      else
        row >>= shift;
    }

    return BitBoard<W>(row);
  } else {
    static_assert(W == 32, "BitBoard::move only implemented for BitBoard<32>");
    return BitBoard<W>();
  }
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
_DI_ bool BitBoard<W>::is_canonical_subsquare() const {
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

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::mirror_around(cuda::std::pair<int, int> cell) const {
  auto [x, y] = cell;

  __builtin_assume(x >= 0 && x < W);
  __builtin_assume(y >= 0 && y < W);

  if constexpr (W == 32) {
    // After bit-reversing, x ends up at (31 - x), and we have to
    // shift that back to x. It's fine that this truncates we don't
    // want torus wrapping here.
    uint32_t t;

    if(31 - x > x)
      t = __brev(state) >> ((31 - x) - x);
    else
      t = __brev(state) << (x - (31 - x));

    // We also have to reflect the rows around y. A row k needs to
    // read from 2*y - k
    int k = threadIdx.x & 31;
    int otherlane = 2*y - k;
    t = __shfl_sync(0xffffffff, t, otherlane);

    if(otherlane < 0 || otherlane > 32) // Value of t undefined
      t = 0;

    return BitBoard<W>(t);

  } else {
    int lane = threadIdx.x & 31;
    int shift = 63 - 2 * x;

    uint64_t even_row = (static_cast<uint64_t>(state.y) << 32) | state.x;
    uint64_t odd_row = (static_cast<uint64_t>(state.w) << 32) | state.z;

    auto mirror_row = [&](uint64_t row) -> uint64_t {
      uint64_t reversed = __brevll(row);
      if (shift >= 0) {
        return reversed >> shift;
      } else {
        return reversed << (-shift);
      }
    };

    uint64_t mirrored_even = mirror_row(even_row);
    uint64_t mirrored_odd = mirror_row(odd_row);

    uint32_t even_low = static_cast<uint32_t>(mirrored_even);
    uint32_t even_high = static_cast<uint32_t>(mirrored_even >> 32);
    uint32_t odd_low = static_cast<uint32_t>(mirrored_odd);
    uint32_t odd_high = static_cast<uint32_t>(mirrored_odd >> 32);

    int y_even_out = lane << 1;
    int y_odd_out = y_even_out + 1;

    auto fetch_row = [&](int src_row) -> uint64_t {
      if (src_row < 0 || src_row >= 64) {
        return 0ULL;
      }

      int src_lane = src_row >> 1;
      uint32_t fetched_even_low = __shfl_sync(0xffffffff, even_low, src_lane);
      uint32_t fetched_even_high = __shfl_sync(0xffffffff, even_high, src_lane);
      uint32_t fetched_odd_low = __shfl_sync(0xffffffff, odd_low, src_lane);
      uint32_t fetched_odd_high = __shfl_sync(0xffffffff, odd_high, src_lane);

      if (src_row & 1) {
        return (static_cast<uint64_t>(fetched_odd_high) << 32) | fetched_odd_low;
      } else {
        return (static_cast<uint64_t>(fetched_even_high) << 32) | fetched_even_low;
      }
    };

    uint64_t even_out = fetch_row(2 * y - y_even_out);
    uint64_t odd_out = fetch_row(2 * y - y_odd_out);

    uint4 result_state;
    result_state.x = static_cast<uint32_t>(even_out);
    result_state.y = static_cast<uint32_t>(even_out >> 32);
    result_state.z = static_cast<uint32_t>(odd_out);
    result_state.w = static_cast<uint32_t>(odd_out >> 32);

    return BitBoard<W>(result_state);
  }
}

template<unsigned W>
_DI_ BitBoard<W> BitBoard<W>::gcd_reduce() const {
  constexpr int MAX_POINTS = 64;
  static __shared__ cuda::std::pair<uint8_t, uint8_t> shared_cells[WARPS_PER_BLOCK][MAX_POINTS];

  const unsigned warp = threadIdx.x >> 5;
  const unsigned lane = threadIdx.x & 31;
  auto *cell_buffer = shared_cells[warp];

  this->on_cells(cell_buffer);
  int total = pop();

  for (int idx = lane; idx < total; idx += 32) {
    auto [x, y] = cell_buffer[idx];

    const unsigned x_div = div_gcd_table[x][y];
    const unsigned y_div = div_gcd_table[y][x];

    cell_buffer[idx] = {x_div, y_div};
  }

  __syncwarp();

  BitBoard<W> reduced;
  for (int idx = 0; idx < total; ++idx) {
    reduced.set(cell_buffer[idx]);
  }

  return reduced;
}

template<unsigned W>
_DI_ void BitBoard<W>::print() const {
  unsigned eol_count = 0;

  for (unsigned j = 0; j < W; j++) {
    bool s = get(0, j);
    char last_val = s ? 'o' : 'b';
    unsigned run_count = 0;

    for (unsigned i = 0; i < W; i++) {
      bool s = get(i, j);
      char val = s ? 'o' : 'b';

      // Flush linefeeds if we find a live cell
      if (val != 'b' && eol_count > 0) {
        if (eol_count > 1)
          if ((threadIdx.x & 31) == 0) printf("%d", eol_count);

        if ((threadIdx.x & 31) == 0) printf("$");

        eol_count = 0;
      }

      // Flush current run if val changes
      if (val != last_val) {
        if (run_count > 1)
            if ((threadIdx.x & 31) == 0) printf("%d", run_count);
        if ((threadIdx.x & 31) == 0) printf("%c", last_val);
        run_count = 0;
      }

      run_count++;
      last_val = val;
    }

    // Flush run of live cells at end of line
    if (last_val != 'b') {
      if (run_count > 1)
        if ((threadIdx.x & 31) == 0) printf("%d", run_count);

      if ((threadIdx.x & 31) == 0) printf("%c", last_val);

      run_count = 0;
    }

    eol_count++;
  }
  if ((threadIdx.x & 31) == 0) printf("!\n");
}
