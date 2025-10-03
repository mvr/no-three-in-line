// NOTE: Derived from silk/cpads/include/cpads/sorting/bitonic.hpp in silk project.
#pragma once
#include "../vec.hpp"

namespace hh {

template<size_t E, typename T, size_t N, bool alternate=false>
_DI_ void warp_bitonic_sort(hh::vec<T, N> &x, bool desc=false) {

    bool descending = desc;
    if constexpr (alternate) {
        descending = (threadIdx.x >> E) & 1;
    }

    if constexpr (E > 0) {

        warp_bitonic_sort<E-1, T, N, true>(x);

        for (int i = 0; i < ((int) E); i++) {
            int m = 1 << (E-1-i);

            #pragma unroll
            for (size_t j = 0; j < N; j++) {
                T other = shuffle_xor_32(x[j], m);
                compare_and_swap(x[j], other, descending ^ ((bool) (threadIdx.x & m)));
            }
        }
    }

    x.sort(descending);
}


template<size_t Log2WarpsPerBlock, size_t E, typename T, size_t N, bool alternate=false>
_DI_ void block_bitonic_merge(hh::vec<T, N> &x, T* smem, bool desc=false) {

    constexpr int ThreadsPerBlock = 32 << Log2WarpsPerBlock;

    int cx = threadIdx.x >> (Log2WarpsPerBlock);
    int cy = threadIdx.x & ((1 << Log2WarpsPerBlock) - 1);

    #pragma unroll
    for (size_t j = 0; j < N; j++) {
        smem[(j * ThreadsPerBlock) ^ threadIdx.x ^ (threadIdx.x >> 5)] = x[j];
    }
    __syncthreads();
    #pragma unroll
    for (size_t j = 0; j < N; j++) {
        x[j] = smem[(j * ThreadsPerBlock) ^ (cy << 5) ^ cy ^ cx];
    }

    bool descending = desc;
    if constexpr (alternate) {
        descending = (threadIdx.x >> (E-5)) & 1;
    }

    for (int i = 0; i < ((int) (E-5)); i++) {
        int m = 1 << (E-6-i);

        #pragma unroll
        for (size_t j = 0; j < N; j++) {
            T other = shuffle_xor_32(x[j], m);
            compare_and_swap(x[j], other, descending ^ ((bool) (threadIdx.x & m)));
        }
    }

    #pragma unroll
    for (size_t j = 0; j < N; j++) {
        smem[(j * ThreadsPerBlock) ^ (cy << 5) ^ cy ^ cx] = x[j];
    }
    __syncthreads();
    #pragma unroll
    for (size_t j = 0; j < N; j++) {
        x[j] = smem[(j * ThreadsPerBlock) ^ threadIdx.x ^ (threadIdx.x >> 5)];
    }

    if constexpr (alternate) {
        descending = (threadIdx.x >> E) & 1;
    }

    for (int i = 0; i < 5; i++) {
        int m = 16 >> i;

        #pragma unroll
        for (size_t j = 0; j < N; j++) {
            T other = shuffle_xor_32(x[j], m);
            compare_and_swap(x[j], other, descending ^ ((bool) (threadIdx.x & m)));
        }
    }

    x.sort(descending);
}


template<size_t Log2WarpsPerBlock, size_t E, typename T, size_t N, bool alternate=false>
_DI_ void block_bitonic_sort(hh::vec<T, N> &x, T* smem, bool desc=false) {

    if constexpr (E <= 5) {
        warp_bitonic_sort<E, T, N, alternate>(x, desc);
    } else {
        block_bitonic_sort<Log2WarpsPerBlock, E-1, T, N, true>(x, smem);
        block_bitonic_merge<Log2WarpsPerBlock, E, T, N, alternate>(x, smem, desc);
    }
}

template<size_t E, typename T, size_t N>
_DI_ bool warp_memcmp_leq(const hh::vec<T, N> &x, const hh::vec<T, N> &y) {

    if constexpr (E == 0) {
        return x <= y;
    } else {
        uint32_t mask = ((uint32_t) -1);
        if constexpr (E <= 4) { mask &= ((threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu); }
        if constexpr (E <= 3) { mask &= ((threadIdx.x &  8) ? 0xff00ff00u : 0x00ff00ffu); }
        if constexpr (E <= 2) { mask &= ((threadIdx.x &  4) ? 0xf0f0f0f0u : 0x0f0f0f0fu); }
        if constexpr (E <= 1) { mask &= ((threadIdx.x &  2) ? 0xccccccccu : 0x33333333u); }
        uint32_t lt_mask = ballot_32(x < y) & mask;
        uint32_t gt_mask = ballot_32(x > y) & mask;
        return brev32(gt_mask) <= brev32(lt_mask);
    }
}

template<size_t E, typename T, size_t N>
_DI_ void warp_memcmp_min(const hh::vec<T, N> &src, hh::vec<T, N> &dst) {

    if (warp_memcmp_leq<E>(src, dst)) {
        #pragma unroll
        for (size_t i = 0; i < N; i++) {
            dst[i] = src[i];
        }
    }
}

}
