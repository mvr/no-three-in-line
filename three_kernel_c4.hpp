#pragma once

#include "common.hpp"

void init_lookup_tables_host();
void init_relevant_endpoint_host();
void init_relevant_endpoint_host_64();

template <unsigned N>
int solve_with_device_stack_c4();

