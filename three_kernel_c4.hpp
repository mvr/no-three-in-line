#pragma once

#include "common.hpp"

struct ProblemC4 {
  board_array_t<32> known_on;
  board_array_t<32> known_off;
};

void init_lookup_tables_host();
void init_relevant_endpoint_host(unsigned n);
void init_relevant_endpoint_host_64(unsigned n);

template <unsigned N>
int solve_with_device_stack_c4();

