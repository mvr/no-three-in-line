#pragma once

#include <stdint.h>
#include <type_traits>
#include <array>
#include <numeric>

#include "params.hpp"

template <unsigned W>
using board_row_t = std::conditional_t<W == 64, uint64_t, uint32_t>;

template <unsigned W>
using board_array_t = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;

#ifdef __CUDACC__
template <unsigned W>
using board_state_t = std::conditional_t<W == 64, uint4, uint32_t>;
#endif

enum struct Axis {
  Vertical,
  Horizontal,
};

enum class LexStatus {
  Less,
  Greater,
  Equal,
  Unknown
};

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

template<unsigned W>
_HD_ int popcount(board_row_t<W> x) {
  if constexpr (W == 32) {
    #ifdef __CUDA_ARCH__
    return __popc(x);
    #else
    return __builtin_popcount(x);
    #endif
  } else {
    #ifdef __CUDA_ARCH__
    return __popcll(x);
    #else
    return __builtin_popcountll(x);
    #endif
  }
}

template<unsigned W>
_HD_ int find_first_set(board_row_t<W> x) {
  if constexpr (W == 32) {
    #ifdef __CUDA_ARCH__
    return __ffs(x) - 1;
    #else
    return __builtin_ffs(x) - 1;
    #endif
  } else {
    #ifdef __CUDA_ARCH__
    return __ffsll(x) - 1;
    #else
    return __builtin_ffsll(x) - 1;
    #endif
  }
}

template<unsigned W>
_HD_ int find_last_set(board_row_t<W> x) {
  if constexpr (W == 32) {
    #ifdef __CUDA_ARCH__
    return 31 - __clz(x);
    #else
    return 31 - __builtin_clz(x);
    #endif
  } else {
    #ifdef __CUDA_ARCH__
    return 63 - __clzll(x);
    #else
    return 63 - __builtin_clzll(x);
    #endif
  }
}

template<unsigned W>
_HD_ int count_trailing_zeros(board_row_t<W> x) {
  if constexpr (W == 32) {
    #ifdef __CUDA_ARCH__
    return __clz(__brev(x));
    #else
    return __builtin_ctz(x);
    #endif
  } else {
    #ifdef __CUDA_ARCH__
    return __clzll(__brevll(x));
    #else
    return __builtin_ctzll(x);
    #endif
  }
}

_HD_ uint64_t interleave32(uint32_t even_bits, uint32_t odd_bits) {
  static const uint64_t B[] = {
      0x0000FFFF0000FFFFULL,
      0x00FF00FF00FF00FFULL,
      0x0F0F0F0F0F0F0F0FULL,
      0x3333333333333333ULL,
      0x5555555555555555ULL};
  static const unsigned S[] = {16, 8, 4, 2, 1};

  uint64_t even = even_bits;
  uint64_t odd = odd_bits;
  #pragma unroll
  for (unsigned i = 0; i < sizeof(B) / sizeof(B[0]); ++i) {
    even = (even | (even << S[i])) & B[i];
    odd  = (odd  | (odd  << S[i])) & B[i];
  }

  return even | (odd << 1);
}

template <unsigned N, unsigned W>
_HD_ unsigned pick_center_col(board_row_t<W> bits) {
  constexpr int center_right = static_cast<int>(N / 2);
  constexpr int center_left = static_cast<int>((N - 1) / 2);
  board_row_t<W> right_mask = bits & (~((board_row_t<W>(1) << center_right) - 1));
  board_row_t<W> left_mask = bits & ((board_row_t<W>(1) << (center_left + 1)) - 1);

  int right = find_first_set<W>(right_mask);
  int left = find_last_set<W>(left_mask);

  bool has_right = right_mask != 0;
  bool has_left = left_mask != 0;

  if (!has_left && has_right)
    return right;

  if (!has_right && has_left)
    return left;

  int dist_right = right - center_right;
  int dist_left = center_left - left;
  return static_cast<unsigned>(dist_right <= dist_left ? right : left);
}

// Also stolen from cpads
template<uint8_t Op>
_HD_ uint32_t lop3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t w;
    #ifdef __CUDA_ARCH__
    asm("lop3.b32 %0,%1,%2,%3,%4;\n" : "=r"(w) : "r"(x), "r"(y), "r"(z), "n"(Op));
    #else
    uint32_t normals[8] = {0, z &~ y, y &~ z, y ^ z, y & z, z, y, y | z};
    constexpr bool op0 = (Op & 1);
    constexpr bool op4 = (Op & 16);
    constexpr uint8_t Lo = op0 ? (7 &~ (Op >> 1)) : (7 & (Op >> 1));
    constexpr uint8_t Hi = op4 ? (7 &~ (Op >> 5)) : (7 & (Op >> 5));
    uint32_t wlo = op0 ? (~normals[Lo]) : normals[Lo];
    uint32_t whi = op4 ? (~normals[Hi]) : normals[Hi];
    w = (whi & x) | (wlo &~ x);
    #endif
    return w;
}

template<uint8_t Op>
_HD_ uint64_t lop3(uint64_t x, uint64_t y, uint64_t z) {
  const uint32_t lo = lop3<Op>(static_cast<uint32_t>(x),
                               static_cast<uint32_t>(y),
                               static_cast<uint32_t>(z));
  const uint32_t hi = lop3<Op>(static_cast<uint32_t>(x >> 32),
                               static_cast<uint32_t>(y >> 32),
                               static_cast<uint32_t>(z >> 32));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

_HD_ uint32_t maj3(uint32_t x, uint32_t y, uint32_t z) { return lop3<0xE8>(x, y, z); }
_HD_ uint32_t xor3(uint32_t x, uint32_t y, uint32_t z) { return lop3<0x96>(x, y, z); }
_HD_ uint32_t mux3(uint32_t x, uint32_t y, uint32_t z) { return lop3<0xCA>(x, y, z); }

_HD_ uint64_t maj3(uint64_t x, uint64_t y, uint64_t z) { return lop3<0xE8>(x, y, z); }
_HD_ uint64_t xor3(uint64_t x, uint64_t y, uint64_t z) { return lop3<0x96>(x, y, z); }
_HD_ uint64_t mux3(uint64_t x, uint64_t y, uint64_t z) { return lop3<0xCA>(x, y, z); }

#ifdef __CUDACC__
#endif
