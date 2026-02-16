#pragma once

#include "three_kernel.hpp"

template <unsigned N, unsigned W>
int solve_with_device_stack_c4(const SearchOptions<W> &options);
