#include <stdint.h>
#include <array>
#include <vector>
#include <utility>

#include "common.hpp"

template<unsigned W>
struct Problem {
  using ArrayType = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;
  ArrayType known_on;
  ArrayType known_off;
  ArrayType seed; // The newly placed, unpropagated ONs
};

template <unsigned W>
struct DeviceStack {
  Problem<W> problems[STACK_CAPACITY];
  unsigned size;
  unsigned lock;
};

template <unsigned W>
struct SolutionBuffer {
  using ArrayType = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;
  ArrayType solutions[SOLUTION_BUFFER_CAPACITY];
  unsigned size;
};

template <unsigned N, unsigned W>
int solve_with_device_stack();

void init_lookup_tables_host();
void init_relevant_endpoint_host();
void init_relevant_endpoint_host_64();
