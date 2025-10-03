// NOTE: Derived from silk/cpads/include/cpads/internal/vec_impl.hpp in silk project.
#pragma once

namespace hh {

_HDC_ uint64_t nextpowerof2(uint64_t x) {

    uint64_t y = x;
    y |= (y >> 1);
    y |= (y >> 2);
    y |= (y >> 4);
    y |= (y >> 8);
    y |= (y >> 16);
    y |= (y >> 32);
    return y + 1;

}

template<typename T>
_HDC_ void compare_and_swap(T &a, T &b, bool descending=false) {

    bool c = (b < a) ^ descending;
    T maximum = c ? a : b;
    T minimum = c ? b : a;
    a = minimum;
    b = maximum;

}

template<typename T, size_t N>
struct vec {

    // array type:
    T x[N];
    constexpr static size_t length = N;

    constexpr static size_t size() { return length; }

    // accessing elements:

    INHERIT_ACCESSORS_FROM(T, x, _HDC_)

    template<typename Fn>
    _HDC_ void modify(Fn lambda) {
        #include "loop.inc"
        { lambda(x[i]); }
    }

    _HDC_ auto reverse() const {
        vec<T, N> result{x[0]};
        #include "loop.inc"
        { result[i] = x[N-1-i]; }
        return result;
    }

    _HDC_ T sum() const {
        T total = 0;
        #include "loop.inc"
        { total += x[i]; }
        return total;
    }

    template<typename U>
    _HDC_ auto astype() const {
        vec<U, N> result{((U) x[0])};
        #include "loop.inc"
        { result[i] = (U) x[i]; }
        return result;
    }

    template<size_t M>
    _HDC_ auto concat(const vec<T, M> &other) const {

        vec<T, M + N> result{x[0]};
        constexpr size_t n = N;
        #include "loop.inc"
        { result[i] = x[i]; }
        #define N M
        #include "loop.inc"
        { result[i + n] = other[i]; }
        #undef N

        return result;
    }

    _HDC_ auto append(const T &last) const {

        vec<T, 1> other{last};
        return concat(other);
    }

    _HDC_ void cswap(const size_t &i, const size_t &j, bool descending=false) {
        compare_and_swap(x[i], x[j], descending);
    }

    _HDC_ void swap(const size_t &i, const size_t &j) {
        auto tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }

    template<size_t X0, size_t NX, size_t NY, size_t Stride>
    _HDC_ void mergeExchange(bool descending=false) {

        if constexpr ((NX >= 1) || (NY >= 1)) {

            // check invariants:
            static_assert((NX & (NX - 1)) == 0, "NX must be a power of 2");
            static_assert(NX <= 2 * NY, "NX must be <= 2 * NY");
            static_assert(NY <= 2 * NX, "NY must be <= 2 * NX");

            if constexpr ((NX == 1) && (NY == 1)) {
                cswap(X0, X0 + Stride, descending);
            } else if constexpr (NX == 1) {
                cswap(X0, X0 + Stride, descending);
                cswap(X0 + Stride, X0 + 2 * Stride, descending);
            } else if constexpr (NY == 1) {
                cswap(X0 + Stride, X0 + 2 * Stride, descending);
                cswap(X0, X0 + Stride, descending);
            } else {
                // NX, NY >= 2
                constexpr size_t halfNX = NX >> 1;
                constexpr size_t floorNY = NY >> 1;
                constexpr size_t ceilNY = NY - floorNY;
                mergeExchange<X0, halfNX, ceilNY, Stride*2>(descending);
                mergeExchange<X0 + Stride, halfNX, floorNY, Stride*2>(descending);

                constexpr size_t numSwaps = (NX + NY - 1) >> 1;

                #define N numSwaps
                #include "loop.inc"
                { cswap(X0 + (2*i+1)*Stride, X0 + (2*i+2)*Stride, descending); }
                #undef N
            }
        }
    }

    template<size_t Start, size_t Length>
    _HDC_ void sortSegment(bool descending=false) {

        if constexpr (Length >= 2) {

            constexpr size_t NX = nextpowerof2(Length / 3);
            constexpr size_t NY = Length - NX;

            sortSegment<Start, NX>(descending);
            sortSegment<Start + NX, NY>(descending);
            mergeExchange<Start, NX, NY, 1>(descending);
        }
    }

    template<size_t Start, size_t FirstIndex, size_t... RemainingIndices>
    _HDC_ void segmentedSort(const hh::vec<T, N> &dst, bool &winning) {

        sortSegment<Start, FirstIndex>();

        if (!winning) {
            for (size_t i = Start; i < Start + FirstIndex; i++) {
                if (dst[i] < x[i]) { return; }
                if (x[i] < dst[i]) { winning = true; break; }
            }
        }

        if constexpr ((sizeof...(RemainingIndices)) > 0) {
            segmentedSort<Start + FirstIndex, RemainingIndices...>(dst, winning);
        }
    }

    template<size_t... Indices>
    _HDC_ bool segmentedSortReplace(hh::vec<T, N> &dst, std::index_sequence<Indices...> /* dummy argument */ ) {

        bool winning = false;
        if constexpr ((sizeof...(Indices)) > 0) {
            segmentedSort<0, Indices...>(dst, winning);
        }
        if (winning) {
            #include "loop.inc"
            { dst[i] = x[i]; }
        }
        return winning;
    }

    _HDC_ void sort(bool descending=false) { sortSegment<0, N>(descending); }

#define X +=
#include "assop.inc"
#undef X
#define X -=
#include "assop.inc"
#undef X
#define X *=
#include "assop.inc"
#undef X
#define X /=
#include "assop.inc"
#undef X
#define X %=
#include "assop.inc"
#undef X
#define X &=
#include "assop.inc"
#undef X
#define X |=
#include "assop.inc"
#undef X
#define X ^=
#include "assop.inc"
#undef X
#define X <<=
#include "assop.inc"
#undef X
#define X >>=
#include "assop.inc"
#undef X

    uint64_t hash() const {

        constexpr size_t nbytes = sizeof(T) * N;
        constexpr size_t k64 = (nbytes + 7) >> 3;

        uint64_t hcopy[k64]; hcopy[k64 - 1] = 0;
        memcpy(hcopy, &x, nbytes);

        uint64_t h = hcopy[0];
        for (size_t i = 1; i < k64; i++) {
            h = h * 6364136223846793005ull + hcopy[i];
        }

        return h;
    }

};

template<typename T, size_t N>
std::ostream& operator<< (std::ostream& os, const hh::vec<T, N>& v) {

    os << "[" << v[0];
    for (size_t i = 1; i < N; i++) {
        os << ", " << v[i];
    }
    os << "]";

    return os;
}

#define binopvv(X) template<typename T, typename U, size_t N> _HDC_ auto X (const vec<T, N> &lhs, const vec<U, N> &rhs)
#define binopvs(X) template<typename T, typename U, size_t N> _HDC_ auto X (const vec<T, N> &lhs, const U &rhs)
#define binopsv(X) template<typename T, typename U, size_t N> _HDC_ auto X (const T &lhs, const vec<U, N> &rhs)

binopvv(operator==) {
    #include "loop.inc"
    { if (lhs[i] != rhs[i]) { return false; } }
    return true;
}

binopvv(operator!=) {
    #include "loop.inc"
    { if (lhs[i] != rhs[i]) { return true; } }
    return false;
}

binopvv(operator<=) {
    #include "loop.inc"
    {
        if (lhs[i] < rhs[i]) { return true; }
        if (rhs[i] < lhs[i]) { return false; }
    }
    return true;
}

binopvv(operator<) {
    #include "loop.inc"
    {
        if (lhs[i] < rhs[i]) { return true; }
        if (rhs[i] < lhs[i]) { return false; }
    }
    return false;
}

binopvv(operator>=) {
    #include "loop.inc"
    {
        if (lhs[i] < rhs[i]) { return false; }
        if (rhs[i] < lhs[i]) { return true; }
    }
    return true;
}

binopvv(operator>) {
    #include "loop.inc"
    {
        if (lhs[i] < rhs[i]) { return false; }
        if (rhs[i] < lhs[i]) { return true; }
    }
    return false;
}

template<typename Fn, size_t... I>
_HDC_ auto materialise_vector_inner(Fn lambda, std::index_sequence<I...> /* dummy argument */ ) {

    constexpr size_t N = sizeof...(I);
    using S = decltype(lambda(0));
    vec<S, N> res{(lambda(I))...};
    return res;

}

template<size_t N, typename Fn>
_HDC_ auto materialise_vector(Fn lambda) {

    return materialise_vector_inner(lambda, std::make_index_sequence<N>{});

}

#define X +
#include "binop.inc"
#undef X
#define X -
#include "binop.inc"
#undef X
#define X *
#include "binop.inc"
#undef X
#define X /
#include "binop.inc"
#undef X
#define X %
#include "binop.inc"
#undef X
#define X &
#include "binop.inc"
#undef X
#define X |
#include "binop.inc"
#undef X
#define X ^
#include "binop.inc"
#undef X
#define X <<
#include "binop.inc"
#undef X
#define X >>
#include "binop.inc"
#undef X

#undef binopvv
#undef binopvs
#undef binopsv

}
