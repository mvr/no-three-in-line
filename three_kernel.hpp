#include <stdint.h>
#include <array>
#include <vector>
#include <utility>


struct Problem {
  std::array<uint64_t, 64> knownOn;
  std::array<uint64_t, 64> knownOff;
  std::array<uint64_t, 64> seed; // The newly placed, unpropagated ONs
};

struct Outcome {
  std::array<uint64_t, 64> knownOn;
  std::array<uint64_t, 64> knownOff;
  bool solved;
  bool consistent;
  unsigned unknownPop;
  // If consistent but not solved, what to branch on:
  // Axis axis;
  unsigned ix;
};


template <unsigned N>
std::vector<Outcome> launch_work_kernel(unsigned batch_size,
                                        std::vector<Problem> problems);


template <unsigned N>
std::pair<std::array<uint64_t, 64>, std::array<uint64_t, 64>>
soft_branch(const std::array<uint64_t, 64> &inputKnownOn,
            const std::array<uint64_t, 64> &inputKnownOff);
