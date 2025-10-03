// NOTE: Derived from silk/cpads/include/cpads/core.hpp in silk project.
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <utility>
#include <type_traits>
#include <cmath>

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

// an inlined constexpr host/device function:
#define _HDC_ _HD_ constexpr

// static assert, but in green:
// #define colour_assert(x, y) static_assert(x, "\033[32;1m" y "\033[0m")
#define colour_assert(x, y) static_assert(x, y)

// useful for defining classes by composition instead of inheritance:
#define INHERIT_COMPARATORS_FROM(ThisType, member, decorator) \
    decorator bool operator==(const ThisType& rhs) const { return member == rhs . member; } \
    decorator bool operator!=(const ThisType& rhs) const { return member != rhs . member; } \
    decorator bool operator<=(const ThisType& rhs) const { return member <= rhs . member; } \
    decorator bool operator>=(const ThisType& rhs) const { return member >= rhs . member; } \
    decorator bool operator< (const ThisType& rhs) const { return member <  rhs . member; } \
    decorator bool operator> (const ThisType& rhs) const { return member >  rhs . member; }

#define INHERIT_ACCESSORS_FROM(ElementType, member, decorator) \
    decorator ElementType& operator[](size_t i) { return member[i]; } \
    decorator const ElementType& operator[](size_t i) const { return member[i]; } 

namespace hh {

template<typename T>
_HDC_ T min(const T &a, const T &b) {
    return (a < b) ? a : b;
}

template<typename T>
_HDC_ T max(const T &a, const T &b) {
    return (a < b) ? b : a;
}

typedef __uint128_t u128;
typedef __int128_t i128;

/// In-place multiplication of 64-bit operands to yield 128-bit result.
_HD_ void mul64x64(uint64_t &low, uint64_t &high) {
    #ifdef __CUDA_ARCH__
    auto product = low * high;
    high = __umul64hi(low, high);
    low = product;
    #else
    u128 product = ((u128) low) * high;
    high = (uint64_t) (product >> 64);
    low = (uint64_t) product;
    #endif
}

/**
 * Rotation intrinsics. Note that there are no range checks, so r should
 * be in the interval [1, w - 1].
 */
_HDC_ uint32_t rotl32(uint32_t input, int r) {
    return (input << r) | (input >> (32 - r));
}

_HDC_ uint32_t rotr32(uint32_t input, int r) {
    return (input >> r) | (input << (32 - r));
}

_HDC_ uint64_t rotl48(uint64_t input, int r) {
    return ((input << r) | (input >> (48 - r))) & 0xffffffffffffull;
}

_HDC_ uint64_t rotr48(uint64_t input, int r) {
    return ((input >> r) | (input << (48 - r))) & 0xffffffffffffull;
}

_HDC_ uint64_t rotl64(uint64_t input, int r) {
    return (input << r) | (input >> (64 - r));
}

_HDC_ uint64_t rotr64(uint64_t input, int r) {
    return (input >> r) | (input << (64 - r));
}

/**
 * Multiply by an invertible circulant 64x64 matrix over F_2.
 * Due to ILP, this should be really fast. Parameters taken from:
 * http://mostlymangling.blogspot.com/2018/07/on-mixing-functions-in-fast-splittable.html
 */
_HDC_ uint64_t mix_circulant(uint64_t input) {
    return input ^ rotr64(input, 49) ^ rotr64(input, 24);
}

/**
 * Quadratic permutation which mixes high bits well.
 * This only uses a single uint64 multiplication plus some cheap ops.
 */
_HDC_ uint64_t mix_quadratic(uint64_t input) {
    return input * (11400714819323198485ull + input + input);
}

/**
 * Function for hashing a 64-bit integer, suitable for hashtables.
 * If the low bits need to be well avalanched, then apply mix_circulant
 * to the output of this.
 */
_HDC_ uint64_t fibmix(uint64_t input) {
    return mix_quadratic(mix_circulant(input));
}

colour_assert(fibmix(0) == 0, "fibmix must be a zero-preserving permutation");

_HD_ uint32_t brev32(uint32_t x) {
    #ifdef __CUDA_ARCH__
    return __brev(x);
    #elif __clang__
    return __builtin_bitreverse32(x);
    #else
    uint32_t y = (x << 16) | (x >> 16);
    y = ((y & 0x00ff00ffu) << 8) | ((y & 0xff00ff00u) >> 8);
    y = ((y & 0x0f0f0f0fu) << 4) | ((y & 0xf0f0f0f0u) >> 4);
    y = ((y & 0x33333333u) << 2) | ((y & 0xccccccccu) >> 2);
    y = ((y & 0x55555555u) << 1) | ((y & 0xaaaaaaaau) >> 1);
    return y;
    #endif
}

_HD_ uint64_t brev64(uint64_t x) {
    #ifdef __CUDA_ARCH__
    return __brevll(x);
    #elif __clang__
    return __builtin_bitreverse64(x);
    #else
    uint64_t y = (x << 32) | (x >> 32);
    y = ((y & 0x0000ffff0000ffffull) << 16) | ((y & 0xffff0000ffff0000ull) >> 16);
    y = ((y & 0x00ff00ff00ff00ffull) <<  8) | ((y & 0xff00ff00ff00ff00ull) >>  8);
    y = ((y & 0x0f0f0f0f0f0f0f0full) <<  4) | ((y & 0xf0f0f0f0f0f0f0f0ull) >>  4);
    y = ((y & 0x3333333333333333ull) <<  2) | ((y & 0xccccccccccccccccull) >>  2);
    y = ((y & 0x5555555555555555ull) <<  1) | ((y & 0xaaaaaaaaaaaaaaaaull) >>  1);
    return y;
    #endif
}

_HD_ int popc32(uint32_t x) {
    #ifdef __CUDA_ARCH__
    return __popc(x);
    #else
    return __builtin_popcount(x);
    #endif
}

_HD_ int popc64(uint64_t x) {
    #ifdef __CUDA_ARCH__
    return __popcll(x);
    #else
    return __builtin_popcountll(x);
    #endif
}

_HD_ int ffs32(uint32_t x) {
    #ifdef __CUDA_ARCH__
    return __ffs(x);
    #else
    return __builtin_ffs(x);
    #endif
}

_HD_ int ffs64(uint64_t x) {
    #ifdef __CUDA_ARCH__
    return __ffsll(x);
    #else
    return __builtin_ffsll(x);
    #endif
}

_HD_ int clz32(uint32_t x) {
    #ifdef __CUDA_ARCH__
    return __clz(x);
    #else
    return __builtin_clz(x);
    #endif
}

_HD_ int clz64(uint64_t x) {
    #ifdef __CUDA_ARCH__
    return __clzll(x);
    #else
    return __builtin_clzll(x);
    #endif
}

_HD_ int ctz32(uint32_t x) {
    #ifdef __CUDA_ARCH__
    return __ffs(x) - 1;
    #else
    return __builtin_ctz(x);
    #endif
}

_HD_ int ctz64(uint64_t x) {
    #ifdef __CUDA_ARCH__
    return __ffsll(x) - 1;
    #else
    return __builtin_ctzll(x);
    #endif
}

/**
 * Computes the integer part of the square-root of the input.
 * This is valid for all 64-bit unsigned integers.
 */
_HD_ uint64_t floor_sqrt(uint64_t x) {
    uint64_t y = ((uint64_t) (std::sqrt((double) x) - 0.5));
    if (2*y < x - y*y) { y++; }
    return y;
}

/*
 * In C++11, a constexpr function can only consist of a single return
 * statement. We implement Gerth Brodal's 'Algorithm B' for computing
 * floor(log2(x)) for any uint64_t x >= 1.
 */

_HDC_ uint64_t constexpr_log2_I(uint64_t t) { return ((t + (t << 30)) >> 60); }
_HDC_ uint64_t constexpr_log2_H(uint64_t t) { return constexpr_log2_I(t + (t << 15)); }
_HDC_ uint64_t constexpr_log2_G(uint64_t x, uint64_t y, uint64_t h) { return constexpr_log2_H(h & (y | ((y | h) - (x ^ y)))); }
_HDC_ uint64_t constexpr_log2_F(uint64_t x) { return constexpr_log2_G(x, x & 0xff00f0f0ccccaaaaull, 0x8000800080008000ull); }
_HDC_ uint64_t constexpr_log2_E(uint64_t input) { return constexpr_log2_F(input | (input << 32)); }
_HDC_ uint64_t constexpr_log2_D(uint64_t input) { return constexpr_log2_E(input | (input << 16)); }
_HDC_ uint64_t constexpr_log2_C(uint64_t input, uint64_t r) { return constexpr_log2_D(input >> r) + r; }
_HDC_ uint64_t constexpr_log2_B(uint64_t input) { return constexpr_log2_C(input, (input >= 0x10000ull) ? 16 : 0); }
_HDC_ uint64_t constexpr_log2_A(uint64_t input, uint64_t r) { return constexpr_log2_B(input >> r) + r; }
_HDC_ uint64_t constexpr_log2(uint64_t input) { return constexpr_log2_A(input, (input >= 0x100000000ull) ? 32 : 0); }

/**
 * Recommended log2(structs_per_alloc) for a struct of a given size.
 */
_HDC_ uint64_t suggest_lowbits(uint64_t byte_size) {
    return constexpr_log2((32768 / byte_size) + 16);
}

/**
 * Fastest runtime implementation of greatest common divisor.
 *
 * This is based on Stein's binary GCD algorithm, but with a modified loop
 * predicate to optimise for the case where the GCD has no odd prime factors
 * (this happens with probability 8/pi^2 = 81% of the time).
 */
_HD_ uint64_t binary_gcd(uint64_t x, uint64_t y) {

    if (x == 0) { return y; }
    if (y == 0) { return x; }
    int i = hh::ctz64(x); uint64_t u = x >> i;
    int j = hh::ctz64(y); uint64_t v = y >> j;
    int k = (i < j) ? i : j;

    while ((u != v) && (v != 1)) { // loop invariant: both u and v are odd
        if (u > v) { auto w = v; v = u; u = w; }
        v -= u; // now v is even
        v >>= hh::ctz64(v);
    }

    return (v << k);
}

} // namespace hh

#ifdef __CUDACC__
#include "gpu_only.hpp"
#endif

