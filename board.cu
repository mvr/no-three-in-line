#include <stdint.h>
#include <cuda/std/utility>

#include "params.hpp"

// an inlined host function:
#define _HI_ __attribute__((always_inline)) inline

// an inlined device function:
#define _DI_ __attribute__((always_inline)) __device__ inline

// an inlined host/device function:
#ifdef __CUDACC__
#define _HD_ __attribute__((always_inline)) __host__ __device__ inline
#else
#define _HD_ _HI_
#endif

struct BitBoard {
  uint4 state;

  _DI_ BitBoard() : state{0, 0, 0, 0} {}
  _DI_ explicit BitBoard(uint4 initial_state) : state(initial_state) {}
  _DI_ static BitBoard solid() {return BitBoard({~0U, ~0U, ~0U, ~0U}); }

  [[nodiscard]] _DI_ static BitBoard load(const uint64_t *state);

  _DI_ void save(uint64_t *state) const;

  _DI_ bool operator==(BitBoard other) const { return (*this ^ other).empty(); }

  _DI_ BitBoard operator~() const { return BitBoard({~state.x, ~state.y, ~state.z, ~state.w}); }
  _DI_ BitBoard operator|(const BitBoard other) const { return BitBoard({state.x | other.state.x, state.y | other.state.y, state.z | other.state.z, state.w | other.state.w }); }
  _DI_ BitBoard operator&(const BitBoard other) const { return BitBoard({state.x & other.state.x, state.y & other.state.y, state.z & other.state.z, state.w & other.state.w }); }
  _DI_ BitBoard operator^(const BitBoard other) const { return BitBoard({state.x ^ other.state.x, state.y ^ other.state.y, state.z ^ other.state.z, state.w ^ other.state.w }); }
  _DI_ void operator|=(const BitBoard other) { state.x |= other.state.x; state.y |= other.state.y; state.z |= other.state.z; state.w |= other.state.w; }
  _DI_ void operator&=(const BitBoard other) { state.x &= other.state.x; state.y &= other.state.y; state.z &= other.state.z; state.w &= other.state.w; }
  _DI_ void operator^=(const BitBoard other) { state.x ^= other.state.x; state.y ^= other.state.y; state.z ^= other.state.z; state.w ^= other.state.w; }

  _DI_ uint64_t row(int y) const;
  _DI_ uint64_t column(int x) const;
  _DI_ bool get(int x, int y) const;
  _DI_ bool get(cuda::std::pair<int, int> cell) const { return get(cell.first, cell.second); }
  _DI_ void set(int x, int y);
  _DI_ void set(cuda::std::pair<int, int> cell) { set(cell.first, cell.second); }
  _DI_ void erase(int x, int y);
  _DI_ void erase(cuda::std::pair<int, int> cell) { erase(cell.first, cell.second); }

  _DI_ cuda::std::pair<int, int> first_on() const;

  _DI_ bool empty() const;
  _DI_ int pop() const;

  static _DI_ BitBoard line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
};

_DI_ BitBoard BitBoard::load(const uint64_t *in) {
  const uint4 *u4ptr = (const uint4 *)in;
  uint4 result = u4ptr[threadIdx.x & 31];
  return BitBoard(result);
}

_DI_ void BitBoard::save(uint64_t *out) const {
  uint4 *u4ptr = (uint4 *)out;
  u4ptr[threadIdx.x & 31] = state;
  // Endianness is fine?
  // out[(threadIdx.x & 31) >> 1] = state.x | (uint64_t)state.y << 32;
  // out[((threadIdx.x & 31) >> 1)+1] = state.z | (uint64_t)state.w << 32;
}

_DI_ uint64_t BitBoard::row(int y) const {
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
}

_DI_ uint64_t BitBoard::column(int x) const {
  uint32_t xs, zs;
  if(x < 32) {
    xs = __ballot_sync(0xffffffffu, state.x & (1<<x));
    zs = __ballot_sync(0xffffffffu, state.z & (1<<x));
  } else {
    xs = __ballot_sync(0xffffffffu, state.y & (1<<(x-32)));
    zs = __ballot_sync(0xffffffffu, state.w & (1<<(x-32)));
  }

  // Now interleave them
  // TODO: can this be avoided using other warp primitives?
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

_DI_ bool BitBoard::get(int x, int y) const {
  uint64_t r = row(y);
  return (r & (1ull << x)) != 0;
}

_DI_ void BitBoard::set(int x, int y) {
  bool should_act = (threadIdx.x & 31) == (y >> 1);
  unsigned int bit = 1u << (x & 31);

  state.x |= bit & (should_act && !(y & 1) && !(x & 32) ? 0xFFFFFFFF : 0);
  state.y |= bit & (should_act && !(y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0);
  state.z |= bit & (should_act &&  (y & 1) && !(x & 32) ? 0xFFFFFFFF : 0);
  state.w |= bit & (should_act &&  (y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0);
}

_DI_ void BitBoard::erase(int x, int y) {
  bool should_act = (threadIdx.x & 31) == (y >> 1);
  unsigned int bit = 1u << (x & 31);

  state.x &= ~(bit & (should_act && !(y & 1) && !(x & 32) ? 0xFFFFFFFF : 0));
  state.y &= ~(bit & (should_act && !(y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0));
  state.z &= ~(bit & (should_act &&  (y & 1) && !(x & 32) ? 0xFFFFFFFF : 0));
  state.w &= ~(bit & (should_act &&  (y & 1) &&  (x & 32) ? 0xFFFFFFFF : 0));
}

_DI_ cuda::std::pair<int, int> BitBoard::first_on() const {
  int x_low = __ffsll((uint64_t) state.y << 32 | state.x) - 1;
  int x_high = __ffsll((uint64_t) state.w << 32 | state.z) - 1;

  bool use_high = ((state.x | state.y) == 0);
  int x = use_high ? x_high : x_low;

  int y_base = (threadIdx.x & 31) << 1;
  int y = y_base + (use_high ? 1 : 0);

  uint32_t mask = __ballot_sync(0xffffffffu, state.x | state.y | state.z | state.w);
  int first_lane = __ffs(mask) - 1;

  y = __shfl_sync(0xffffffff, y, first_lane); // mvrnote: This order might reduce instruction dependency?
  x = __shfl_sync(0xffffffff, x, first_lane);

  return {x, y};
}

_DI_ bool BitBoard::empty() const {
  return __ballot_sync(0xffffffffu, state.x | state.y | state.z | state.w) == 0;
}

_DI_ int BitBoard::pop() const {
  int val = __popc(state.x) + __popc(state.y) + __popc(state.z) + __popc(state.w);
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return __shfl_sync(0xffffffff, val, 0);
}


// Copied from https://gitlab.com/hatsya/open-source/cpads/-/blob/master/include/cpads/core.hpp
// TODO: just make this a lookup table?
/**
 * Fastest runtime implementation of greatest common divisor.
 *
 * This is based on Stein's binary GCD algorithm, but with a modified loop
 * predicate to optimise for the case where the GCD has no odd prime factors
 * (this happens with probability 8/pi^2 = 81% of the time).
 */
_DI_ uint32_t binary_gcd(uint32_t x, uint32_t y) {
    if (x == 0) { return y; }
    if (y == 0) { return x; }
    int i = __ffs(x)-1; uint32_t u = x >> i;
    int j = __ffs(y)-1; uint32_t v = y >> j;
    int k = (i < j) ? i : j;

    while ((u != v) && (v != 1)) { // loop invariant: both u and v are odd
        if (u > v) { auto w = v; v = u; u = w; }
        v -= u; // now v is even
        v >>= __ffs(v)-1;
    }

    return (v << k);
}

_DI_ BitBoard BitBoard::line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q) {
  if (p.first == q.first || p.second == q.second)
    return BitBoard();

  if (p.second > q.second)
    cuda::std::swap(p, q);

  cuda::std::pair<int, unsigned> delta = {(int)q.first - p.first, q.second - p.second};

  {
    int factor = binary_gcd(std::abs(delta.first), delta.second);
    delta.first = delta.first / factor;
    delta.second = delta.second / factor;
  }

  unsigned p_quo = p.second / delta.second;
  unsigned p_rem = p.second % delta.second;

  BitBoard result;

  // TODO: any optimisation tricks?
  {
    unsigned row = 2*threadIdx.x;
    if (row % delta.second == p_rem) {
      int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
      if(col >= 0 && col < 32) result.state.x |= 1 << col;
      else if(col >= 32 && col < 64) result.state.y |= 1 << (col-32);
    }
  }

  {
    unsigned row = 2*threadIdx.x+1;
    if (row % delta.second == p_rem) {
      int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
      // printf("row2: %d, col: %d\n", row, col);
      if(col >= 0 && col < 32) result.state.z |= 1 << col;
      else if(col >= 32 && col < 64) result.state.w |= 1 << (col-32);
    }
  }

  return result;
}

enum struct Axis {
  Vertical,
  Horizontal,
};

template <unsigned N>
struct ThreeBoard {
  BitBoard knownOn;
  BitBoard knownOff;

  _DI_ ThreeBoard() : knownOn{}, knownOff{} {}
  _DI_ explicit ThreeBoard(BitBoard knownOn, BitBoard knownOff) : knownOn{knownOn}, knownOff{knownOff} {}

  _DI_ bool operator==(ThreeBoard<N> other) const { return (knownOn == other.knownOn) && (knownOff == other.knownOff); }
  _DI_ bool operator!=(ThreeBoard<N> other) const { return !(*this == other); }

  static _DI_ BitBoard bounds();
  static _DI_ BitBoard line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);

  _DI_ bool consistent() const;
  _DI_ unsigned unknown_pop() const;

  _DI_ ThreeBoard<N> force_orthogonal_horiz() const;
  _DI_ ThreeBoard<N> force_orthogonal_vert() const;
  _DI_ ThreeBoard<N> force_orthogonal() const { return force_orthogonal_horiz().force_orthogonal_vert(); }

  _DI_ BitBoard eliminate_line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard seed);
  _DI_ void eliminate_all_lines() { eliminate_all_lines(knownOn); } // This shouldn't need to be used, we can just eliminate lines as they are added
  _DI_ void propagate();

  template<Axis d>
  _DI_ void soft_branch(unsigned row);
  _DI_ void soft_branch_all();

  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_row() const;
  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_col() const;
};

// Produces a solid NxN square from (0, 0) to (N - 1, N - 1)
template <unsigned N>
_DI_ BitBoard ThreeBoard<N>::bounds() {
  uint32_t row_bound_x = N >= 32 ? (~0) : (1 << N) - 1;
  uint32_t row_bound_y = N >= 32 ? (1 << (N - 32)) - 1 : 0;
  bool has_half = (threadIdx.x & 31) < ((N + 1) >> 1);
  bool has_full = (threadIdx.x & 31) < (N >> 1);
  BitBoard result;
  result.state.x = has_half ? row_bound_x : 0;
  result.state.y = has_half ? row_bound_y : 0;
  result.state.z = has_full ? row_bound_x : 0;
  result.state.w = has_full ? row_bound_y : 0;
  return result;
}

template <unsigned N>
_DI_ bool ThreeBoard<N>::consistent() const {
  return (knownOn & knownOff).empty();
}

template <unsigned N>
_DI_ unsigned ThreeBoard<N>::unknown_pop() const {
  return N*N - (knownOn | knownOff).pop();
}

// For these,
// Count the knownOn population across the row
// If there's exactly 2, set all others to knownOff
// Count the knownOff population across the row
// If there's exactly N-2, set all others to knownOn

template <unsigned N>
_DI_ ThreeBoard<N> ThreeBoard<N>::force_orthogonal_horiz() const {
  ThreeBoard<N> result = *this;

  int on_pop_x = __popc(knownOn.state.x) + __popc(knownOn.state.y);
  if(on_pop_x == 2) {
    result.knownOff.state.x = ~knownOn.state.x;
    result.knownOff.state.y = ~knownOn.state.y;
  }
  if(on_pop_x > 2) { // Blow everything up
    result.knownOn = BitBoard::solid();
    result.knownOff = BitBoard::solid();
  }

  int on_pop_z = __popc(knownOn.state.z) + __popc(knownOn.state.w);
  if(on_pop_z == 2) {
    result.knownOff.state.z = ~knownOn.state.z;
    result.knownOff.state.w = ~knownOn.state.w;
  }
  if(on_pop_z > 2) {
    result.knownOn = BitBoard::solid();
    result.knownOff = BitBoard::solid();
  }

  int off_pop_x = __popc(knownOff.state.x) + __popc(knownOff.state.y);
  if(off_pop_x == N - 2) {
    result.knownOn.state.x = ~knownOff.state.x;
    result.knownOn.state.y = ~knownOff.state.y;
  }
  if(off_pop_x > N - 2) {
    result.knownOn = BitBoard::solid();
    result.knownOff = BitBoard::solid();
  }

  int off_pop_z = __popc(knownOff.state.z) + __popc(knownOff.state.w);
  if(off_pop_z == N - 2) {
    result.knownOn.state.z = ~knownOff.state.z;
    result.knownOn.state.w = ~knownOff.state.w;
  }
  if(off_pop_z > N - 2) {
    result.knownOn = BitBoard::solid();
    result.knownOff = BitBoard::solid();
  }

  const BitBoard bds = ThreeBoard<N>::bounds();
  result.knownOn &= bds;
  result.knownOff &= bds;

  return result;
}

// To do the same thing vertically across threads, we'll have to do some binary counting.

struct BinaryCount {
  uint32_t bit0;
  uint32_t bit1;
  uint32_t overflow;

  _DI_ BinaryCount operator+(const BinaryCount other) const {
    // TODO: how much of this becomes LOP3s?

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


template <unsigned N>
_DI_ ThreeBoard<N> ThreeBoard<N>::force_orthogonal_vert() const {
  ThreeBoard<N> result = *this;

  const BinaryCount on_count_xz = count_vertically(knownOn.state.x) + count_vertically(knownOn.state.z);
  const uint32_t on_count_xz_eq_2 = ~on_count_xz.overflow & on_count_xz.bit1 & ~on_count_xz.bit0;
  result.knownOff.state.x |= ~knownOn.state.x & on_count_xz_eq_2;
  result.knownOff.state.z |= ~knownOn.state.z & on_count_xz_eq_2;

  // Signal contradiction
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

  BitBoard notKnownOff = ~knownOff & ThreeBoard<N>::bounds();

  const BinaryCount not_off_count_xz = count_vertically(notKnownOff.state.x) + count_vertically(notKnownOff.state.z);
  const uint32_t not_off_count_xz_eq_2 = ~not_off_count_xz.overflow & not_off_count_xz.bit1 & ~not_off_count_xz.bit0;
  result.knownOn.state.x |= ~knownOff.state.x & not_off_count_xz_eq_2;
  result.knownOn.state.z |= ~knownOff.state.z & not_off_count_xz_eq_2;

  // NOTE: this is actually sneaky, because contradiction bits will be
  // set in the region outside the bounds. Restricting to the bounds
  // at the end makes it safe. (But we should probably just mask with
  // (1<<N)...)
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

  const BitBoard bds = ThreeBoard<N>::bounds();
  result.knownOn &= bds;
  result.knownOff &= bds;

  return result;
}

template <unsigned N>
_DI_ BitBoard
ThreeBoard<N>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                              cuda::std::pair<unsigned, unsigned> q) {
  // This situation is handled by `force_orthogonal`
  if (p.first == q.first || p.second == q.second)
    return BitBoard();

  BitBoard line = BitBoard::line(p, q);

  // Remove the rows that have p and q in them
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
  return line;
}

template <unsigned N>
_DI_ void
ThreeBoard<N>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard qs = knownOn;
  for (auto q = qs.first_on(); !qs.empty();
       qs.erase(q), q = qs.first_on()) {
    knownOff |= eliminate_line(p, q);
  }
  knownOff &= bounds();
}

template <unsigned N>
_DI_ void
ThreeBoard<N>::eliminate_all_lines(BitBoard seed) {
  for (auto p = seed.first_on(); !seed.empty();
       seed.erase(p), p = seed.first_on()) {
    eliminate_all_lines(p);
  }
  knownOff &= bounds();
}

template <unsigned N>
_DI_ void ThreeBoard<N>::propagate() {
  ThreeBoard<N> prev;

  // Until stabilised
  do {
    prev = *this;

    auto forced_orthogonal = force_orthogonal();
    if(!consistent()) break;

    // Get new knownOns that need their lines removed:
    BitBoard newOns = forced_orthogonal.knownOn & ~prev.knownOn;
    *this = forced_orthogonal;

    eliminate_all_lines(newOns);
  } while (*this != prev);
}

template <unsigned N>
template <Axis d>
_DI_ void ThreeBoard<N>::soft_branch<d>(unsigned r) {
  uint64_t row_knownOn;
  if constexpr(d == Axis::Horizontal) {
    row_knownOn = knownOn.row(r);
  } else {
    row_knownOn = knownOn.column(r);
  }

  unsigned on_count = __popcll(row_knownOn);
  if(on_count >= 2) return;

  uint64_t row_knownOff;
  if constexpr(d == Axis::Horizontal) {
    row_knownOff = knownOff.row(r);
  } else {
    row_knownOff = knownOff.column(r);
  }

  unsigned off_count = __popcll(row_knownOff);
  unsigned unknown_count = N - on_count - off_count;
  if (on_count == 1 && unknown_count > SOFT_BRANCH_1_THRESHOLD)
    return;
  if (on_count == 0 && unknown_count > SOFT_BRANCH_2_THRESHOLD)
    return;

  // Collect values that are the same in all branches
  ThreeBoard<N> common(BitBoard::solid(), BitBoard::solid());

  uint64_t remaining = ~row_knownOn & ~row_knownOff & ((1ULL << N) - 1);

  if(on_count == 1) {
    // Iterate through possible remaining cell
    for (; remaining; remaining &= remaining - 1) {
      unsigned c = __ffsll(remaining) - 1;

      cuda::std::pair<unsigned, unsigned> cell;
      if constexpr (d == Axis::Horizontal)
        cell = {c, r};
      else
        cell = {r, c};

      ThreeBoard<N> subBoard = *this;
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
    // This is expensive, we shouldn't do it if there are too many unknown cells

    // Iterate through possible remaining cell
    for (; remaining; remaining &= remaining - 1) {
      unsigned c = __ffsll(remaining) - 1;

      cuda::std::pair<unsigned, unsigned> cell;
      if constexpr (d == Axis::Horizontal)
        cell = {c, r};
      else
        cell = {r, c};

      ThreeBoard<N> subBoard = *this;
      subBoard.knownOn.set(cell);
      subBoard.eliminate_all_lines(cell);
      subBoard.propagate();

      if (!subBoard.consistent()) {
        knownOff.set(cell);
      } else {
        uint64_t row_knownOff2;
        if constexpr(d == Axis::Horizontal) {
          row_knownOff2 = subBoard.knownOff.row(r);
        } else {
          row_knownOff2 = subBoard.knownOff.column(r);
        }
        uint64_t remaining2 = ~row_knownOff2 & ((1ULL << N) - 1);
        for (; remaining2; remaining2 &= remaining2 - 1) {
          unsigned c2 = __ffsll(remaining2)-1;

          cuda::std::pair<unsigned, unsigned> cell2;
          if constexpr (d == Axis::Horizontal)
            cell2 = {c2, r};
          else
            cell2 = {r, c2};

          ThreeBoard<N> subBoard2 = *this;
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

template <unsigned N>
_DI_ void ThreeBoard<N>::soft_branch_all() {
  for (int r = 0; r < N; r++) {
    soft_branch<Axis::Horizontal>(r);
  }
  for (int r = 0; r < N; r++) {
    soft_branch<Axis::Vertical>(r);
  }
}

template <unsigned N>
_DI_ cuda::std::pair<unsigned, unsigned>
ThreeBoard<N>::most_constrained_row() const {
  BitBoard known = knownOn | knownOff;
  unsigned unknown_xy = N - __popc(known.state.x) + __popc(known.state.y);
  unsigned unknown_zw = N - __popc(known.state.z) + __popc(known.state.w);

  // Rows with no knownOn have to make two choices
  if(knownOn.state.x == 0 && knownOn.state.y == 0)
    unknown_xy = unknown_xy * (unknown_xy - 1);

  if(knownOn.state.z == 0 && knownOn.state.w == 0)
    unknown_zw = unknown_zw * (unknown_zw - 1);

  // Invalid cases
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
}

template <unsigned N>
_DI_ cuda::std::pair<unsigned, unsigned>
ThreeBoard<N>::most_constrained_col() const {

}

// Last pre-multistream commit:
// https://gitlab.com/apgoucher/silk/-/commit/f4005091b4093f403e62570a44d135347d1f012f
