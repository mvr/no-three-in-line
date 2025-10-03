// NOTE: Derived from silk/cpads/include/cpads/internal/binop.inc in silk project.

binopvv(operator X) {
    return materialise_vector<N>([&](size_t i) __attribute__((always_inline)) {
        return lhs[i] X rhs[i];
    });
}

binopvs(operator X) {
    return materialise_vector<N>([&](size_t i) __attribute__((always_inline)) {
        return lhs[i] X rhs;
    });
}

binopsv(operator X) {
    return materialise_vector<N>([&](size_t i) __attribute__((always_inline)) {
        return lhs X rhs[i];
    });
}
