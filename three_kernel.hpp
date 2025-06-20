#include <stdint.h>
#include <array>
#include <utility>

template <unsigned N>
std::pair<std::array<uint64_t, 64>, std::array<uint64_t, 64>>
soft_branch(const std::array<uint64_t, 64> &inputKnownOn,
            const std::array<uint64_t, 64> &inputKnownOff);
