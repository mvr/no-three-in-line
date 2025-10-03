// NOTE: Derived from silk/cpads/include/cpads/internal/assop.inc in silk project.

template<typename U>
_HDC_ hh::vec<T, N>& operator X (const hh::vec<U, N> &other) {

    #ifdef __CUDA_ARCH__
    #pragma unroll
    for (size_t i = 0; i < N; i++)
    #else
    for (size_t i = 0; i < N; i++)
    #endif
    {
        x[i] X other[i];
    }

    return (*this);
}

template<typename U>
_HDC_ hh::vec<T, N>& operator X (const U& other) {

    #ifdef __CUDA_ARCH__
    #pragma unroll
    for (size_t i = 0; i < N; i++)
    #else
    for (size_t i = 0; i < N; i++)
    #endif
    {
        x[i] X other;
    }

    return (*this);
}

