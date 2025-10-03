// NOTE: Derived from silk/cpads/include/cpads/gpu_only.hpp in silk project.
#pragma once

namespace hh {

inline int reportCudaError(cudaError_t err) {

    if (err != cudaSuccess) {
        int interr = (int) err;
        std::cerr << "\033[31;1mCUDA Error " << interr << " : " << cudaGetErrorString(err) << "\033[0m" << std::endl;
        return interr;
    }

    return 0;
}

inline int getDeviceCount() {

    int count = 0;
    if (reportCudaError(cudaGetDeviceCount(&count))) { return -1; }
    return count;

}

/**
 * Thread-local shared memory allows efficient computed offsets
 * without spilling registers to local memory. This container uses
 * a memory layout that avoids shared memory bank conflicts.
 */
template<typename T, int BlockDim>
struct TLSM {

    T *ptr;

    _DI_ T getOther(int idx, int thread) const {
        return ptr[idx * BlockDim + thread];
    }

    _DI_ T& operator[](int idx) {
        return ptr[idx * BlockDim + threadIdx.x];
    }

    _DI_ T operator[](int idx) const {
        return ptr[idx * BlockDim + threadIdx.x];
    }

    _DI_ TLSM(T* ptr) : ptr(ptr) { }

};

#define allocTLSM(Name, T, BlockDim, Elements) __shared__ T Name ## _underlying_smem[(BlockDim) * (Elements)]; hh::TLSM<T, BlockDim> Name(Name ## _underlying_smem)

/**
 * Wrappers of GPU atomics that work with the stdint.h datatypes
 * for 32-bit and 64-bit signed and unsigned integers.
 */
#define MAKE_ATOMIC(new_name, old_name, new_type, old_type) _DI_ new_type new_name(new_type* location, U value) { return ((new_type) old_name((old_type*) location, (old_type) value)); }

template<typename T, typename U> MAKE_ATOMIC(atomic_add, atomicAdd, T, T)
template<typename U>             MAKE_ATOMIC(atomic_add, atomicAdd, uint64_t, unsigned long long)
template<typename U>             MAKE_ATOMIC(atomic_add, atomicAdd, int64_t, unsigned long long)

template<typename T, typename U>
_DI_ T atomic_sub(T* location, U value) {
    return atomic_add(location, -value);
}

template<typename T, typename U> MAKE_ATOMIC(atomic_and, atomicAnd, T, T)
template<typename U>             MAKE_ATOMIC(atomic_and, atomicAnd, uint64_t, unsigned long long)
template<typename U>             MAKE_ATOMIC(atomic_and, atomicAnd, int64_t, unsigned long long)

template<typename T, typename U> MAKE_ATOMIC(atomic_xor, atomicXor, T, T)
template<typename U>             MAKE_ATOMIC(atomic_xor, atomicXor, uint64_t, unsigned long long)
template<typename U>             MAKE_ATOMIC(atomic_xor, atomicXor, int64_t, unsigned long long)

template<typename T, typename U> MAKE_ATOMIC(atomic_or, atomicOr, T, T)
template<typename U>             MAKE_ATOMIC(atomic_or, atomicOr, uint64_t, unsigned long long)
template<typename U>             MAKE_ATOMIC(atomic_or, atomicOr, int64_t, unsigned long long)

template<typename T, typename U> MAKE_ATOMIC(atomic_min, atomicMin, T, T)
template<typename U>             MAKE_ATOMIC(atomic_min, atomicMin, uint64_t, unsigned long long)
template<typename U>             MAKE_ATOMIC(atomic_min, atomicMin, int64_t, long long)

template<typename T, typename U> MAKE_ATOMIC(atomic_max, atomicMax, T, T)
template<typename U>             MAKE_ATOMIC(atomic_max, atomicMax, uint64_t, unsigned long long)
template<typename U>             MAKE_ATOMIC(atomic_max, atomicMax, int64_t, long long)

#undef MAKE_ATOMIC

#define MAKE_SHUFFLE(new_name, old_name, new_type, old_type) _DI_ new_type new_name(const new_type &x, int y) { return ((new_type) old_name(0xffffffffu, ((old_type) x), y)); }

template<typename T> MAKE_SHUFFLE(shuffle_32, __shfl_sync, T, T)
template<>           MAKE_SHUFFLE(shuffle_32, __shfl_sync, uint64_t, unsigned long long)
template<>           MAKE_SHUFFLE(shuffle_32, __shfl_sync, int64_t, long long)

template<typename T> MAKE_SHUFFLE(shuffle_xor_32, __shfl_xor_sync, T, T)
template<>           MAKE_SHUFFLE(shuffle_xor_32, __shfl_xor_sync, uint64_t, unsigned long long)
template<>           MAKE_SHUFFLE(shuffle_xor_32, __shfl_xor_sync, int64_t, long long)

template<typename T> MAKE_SHUFFLE(shuffle_up_32, __shfl_up_sync, T, T)
template<>           MAKE_SHUFFLE(shuffle_up_32, __shfl_up_sync, uint64_t, unsigned long long)
template<>           MAKE_SHUFFLE(shuffle_up_32, __shfl_up_sync, int64_t, long long)

template<typename T> MAKE_SHUFFLE(shuffle_down_32, __shfl_down_sync, T, T)
template<>           MAKE_SHUFFLE(shuffle_down_32, __shfl_down_sync, uint64_t, unsigned long long)
template<>           MAKE_SHUFFLE(shuffle_down_32, __shfl_down_sync, int64_t, long long)

_DI_ uint32_t ballot_32(bool p) {
    return __ballot_sync(0xffffffffu, (p));
}

#undef MAKE_SHUFFLE

_DI_ uint32_t warp_xor(uint32_t x) {
    #if __CUDA_ARCH__ >= 800
    uint32_t y = __reduce_xor_sync(0xffffffffu, x);
    #else
    uint32_t y = x;
    y ^= shuffle_xor_32(y, 1);
    y ^= shuffle_xor_32(y, 2);
    y ^= shuffle_xor_32(y, 4);
    y ^= shuffle_xor_32(y, 8);
    y ^= shuffle_xor_32(y, 16);
    #endif
    return y;
}

_DI_ uint32_t warp_or(uint32_t x) {
    #if __CUDA_ARCH__ >= 800
    uint32_t y = __reduce_or_sync(0xffffffffu, x);
    #else
    uint32_t y = x;
    y |= shuffle_xor_32(y, 1);
    y |= shuffle_xor_32(y, 2);
    y |= shuffle_xor_32(y, 4);
    y |= shuffle_xor_32(y, 8);
    y |= shuffle_xor_32(y, 16);
    #endif
    return y;
}

_DI_ uint32_t warp_and(uint32_t x) {
    #if __CUDA_ARCH__ >= 800
    uint32_t y = __reduce_and_sync(0xffffffffu, x);
    #else
    uint32_t y = x;
    y &= shuffle_xor_32(y, 1);
    y &= shuffle_xor_32(y, 2);
    y &= shuffle_xor_32(y, 4);
    y &= shuffle_xor_32(y, 8);
    y &= shuffle_xor_32(y, 16);
    #endif
    return y;
}

_DI_ uint32_t warp_add(uint32_t x) {
    #if __CUDA_ARCH__ >= 800
    uint32_t y = __reduce_add_sync(0xffffffffu, x);
    #else
    uint32_t y = x;
    y += shuffle_xor_32(y, 1);
    y += shuffle_xor_32(y, 2);
    y += shuffle_xor_32(y, 4);
    y += shuffle_xor_32(y, 8);
    y += shuffle_xor_32(y, 16);
    #endif
    return y;
}

} // namespace hh
