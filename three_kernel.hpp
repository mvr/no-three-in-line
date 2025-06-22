#include <stdint.h>
#include <array>
#include <vector>
#include <utility>

#include "common.hpp"

template<unsigned W>
struct Problem {
  using ArrayType = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;
  ArrayType knownOn;
  ArrayType knownOff;
  ArrayType seed; // The newly placed, unpropagated ONs
};

template<unsigned W>
struct Outcome {
  using ArrayType = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;
  ArrayType knownOn;
  ArrayType knownOff;
  bool solved;
  bool consistent;
  unsigned unknownPop;
  // If consistent but not solved, what to branch on:
  Axis axis;
  unsigned ix;
};


template <unsigned N, unsigned W>
std::vector<Outcome<W>> launch_work_kernel(unsigned batch_size,
                                           std::vector<Problem<W>> problems);


template <unsigned N, unsigned W>
std::pair<typename Problem<W>::ArrayType, typename Problem<W>::ArrayType>
soft_branch(const typename Problem<W>::ArrayType &inputKnownOn,
            const typename Problem<W>::ArrayType &inputKnownOff);
