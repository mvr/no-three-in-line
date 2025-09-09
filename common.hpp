#pragma once

#include <stdint.h>
#include <type_traits>
#include <array>

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
    #ifdef __CUDACC__
    return __popc(x);
    #else
    return __builtin_popcount(x);
    #endif
  } else {
    #ifdef __CUDACC__
    return __popcll(x);
    #else
    return __builtin_popcountll(x);
    #endif
  }
}

template<unsigned W>
_HD_ int find_first_set(board_row_t<W> x) {
  if constexpr (W == 32) {
    #ifdef __CUDACC__
    return __ffs(x) - 1;
    #else
    return __builtin_ffs(x);
    #endif
  } else {
    #ifdef __CUDACC__
    return __ffsll(x) - 1;
    #else
    return __builtin_ffsll(x);
    #endif
  }
}

template<unsigned W>
_HD_ int find_last_set(board_row_t<W> x) {
  if constexpr (W == 32) {
    #ifdef __CUDACC__
    return 31 - __clz(x);
    #else
    return 31 - __builtin_clz(x);
    #endif
  } else {
    #ifdef __CUDACC__
    return 63 - __clzll(x);
    #else
    return 63 - __builtin_clzll(x);
    #endif
  }
}

template<unsigned W>
_HD_ int count_trailing_zeros(board_row_t<W> x) {
  if constexpr (W == 32) {
    #ifdef __CUDACC__
    return __clz(__brev(x));
    #else
    return __builtin_ctz(x);
    #endif
  } else {
    #ifdef __CUDACC__
    return __clzll(__brevll(x));
    #else
    return __builtin_ctzll(x);
    #endif
  }
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

_HD_ uint32_t maj3(uint32_t x, uint32_t y, uint32_t z) { return lop3<0xE8>(x, y, z); }
_HD_ uint32_t xor3(uint32_t x, uint32_t y, uint32_t z) { return lop3<0x96>(x, y, z); }
_HD_ uint32_t mux3(uint32_t x, uint32_t y, uint32_t z) { return lop3<0xCA>(x, y, z); }

#ifdef __CUDACC__

__constant__ unsigned char div_gcd_table[64][64];

__constant__ uint64_t relevant_endpoint_table[64];
__constant__ uint64_t relevant_endpoint_table_64[256];

#endif
