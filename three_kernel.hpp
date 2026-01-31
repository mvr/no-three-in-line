#include <stdint.h>
#include <array>
#include <vector>
#include <utility>
#include <ostream>

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
  unsigned overflow;
};

template <unsigned W>
struct SolutionBuffer {
  board_array_t<W> solutions[SOLUTION_BUFFER_CAPACITY];
  unsigned size;
};

template <unsigned N, unsigned W>
int solve_with_device_stack();
template <unsigned N, unsigned W>
int solve_with_device_stack(const board_array_t<W> *seed_on,
                            const board_array_t<W> *seed_off);

struct FrontierConfig {
  unsigned min_on = 0;
  unsigned max_on = 0;
  unsigned max_steps = 0;
  unsigned buffer_capacity = 0;
  bool use_on_band = false;
};

template <unsigned N, unsigned W>
int solve_frontier_with_device_stack(const FrontierConfig &config,
                                     std::ostream &out);

void init_lookup_tables_host();
void init_relevant_endpoint_host(unsigned n);
void init_relevant_endpoint_host_64(unsigned n);
