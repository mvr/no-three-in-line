#pragma once

#include "three_kernel.hpp"

void init_lookup_tables_host();

template <unsigned N>
int solve_with_device_stack_c4near();
template <unsigned N>
int solve_with_device_stack_c4near(const board_array_t<32> *seed_on,
                                   const board_array_t<32> *seed_off);
template <unsigned N>
int solve_with_device_stack_c4near(unsigned frontier_min_on);
