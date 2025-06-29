#pragma once

#include <stdint.h>
#include <type_traits>

#include "params.hpp"

enum struct Axis {
  Vertical,
  Horizontal,
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

#ifdef __CUDACC__

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

__constant__ unsigned char div_gcd_table[64][64];

__constant__ uint64_t relevant_endpoint_table[64];
__constant__ uint64_t relevant_endpoint_table_64[256];
#endif
