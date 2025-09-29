#include <stdint.h>
#include <array>
#include <vector>
#include <utility>

#include "common.hpp"

template<unsigned W>
struct Problem {
  board_array_t<W> known_on;
  board_array_t<W> known_off;
};

template <unsigned W>
struct DeviceStack {
  Problem<W> problems[STACK_CAPACITY];
  unsigned size;
};

template <unsigned W>
struct SolutionBuffer {
  board_array_t<W> solutions[SOLUTION_BUFFER_CAPACITY];
  unsigned size;
};

template <unsigned N, unsigned W>
int solve_with_device_stack();

void init_lookup_tables_host();
void init_relevant_endpoint_host(unsigned n);
void init_relevant_endpoint_host_64(unsigned n);
