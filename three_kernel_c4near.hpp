#pragma once

#include "three_kernel.hpp"

void init_lookup_tables_host();

template <unsigned N>
int solve_with_device_stack_c4near(const SearchOptions<32> &options);
