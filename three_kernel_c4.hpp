#pragma once

#include "three_kernel.hpp"

void init_lookup_tables_host();
void init_relevant_endpoint_host(unsigned n);
void init_relevant_endpoint_host_64(unsigned n);

template <unsigned N>
int solve_with_device_stack_c4();
template <unsigned N>
int solve_with_device_stack_c4(unsigned frontier_min_on);
template <unsigned N>
int solve_with_device_stack_c4(const board_array_t<32> *seed_on,
                               const board_array_t<32> *seed_off);
template <unsigned N>
int solve_with_device_stack_c4(const board_array_t<64> *seed_on,
                               const board_array_t<64> *seed_off);
