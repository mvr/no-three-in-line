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
struct OutputBuffer {
  Problem<W> *entries;
  unsigned size;
  unsigned overflow;
  unsigned capacity;
};

template <unsigned W>
struct DeviceStack {
  Problem<W> problems[STACK_CAPACITY];
  unsigned size;
  unsigned overflow;
};

template <unsigned N, unsigned W>
int solve_with_device_stack();
template <unsigned N, unsigned W>
int solve_with_device_stack(const board_array_t<W> *seed_on,
                            const board_array_t<W> *seed_off);

struct FrontierConfig {
  unsigned max_on = 0;
  unsigned buffer_capacity = 0;
};

template <unsigned N, unsigned W>
int solve_with_device_stack(const FrontierConfig &config);

void init_lookup_tables_host();
void init_relevant_endpoint_host(unsigned n);
void init_relevant_endpoint_host_64(unsigned n);
