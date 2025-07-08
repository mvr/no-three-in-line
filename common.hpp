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

#ifdef __CUDACC__

__constant__ unsigned char div_gcd_table[64][64];

__constant__ uint64_t relevant_endpoint_table[64];
__constant__ uint64_t relevant_endpoint_table_64[256];

#endif
