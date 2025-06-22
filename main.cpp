#include <iostream>
#include <tuple>

#include "parsing.hpp"

#include "three_kernel.hpp"

#include "params.hpp"

void resolve_outcome(Outcome &outcome, std::vector<Problem> &stack) {
  if (!outcome.consistent) {
    return;
  }
  if (outcome.solved) {
    std::cout << to_rle(outcome.knownOn) << std::endl;
    return;
  }

  uint64_t row_knownOn = outcome.knownOn[outcome.ix];
  uint64_t row_knownOff = outcome.knownOff[outcome.ix];
  uint64_t remaining = ~row_knownOn & ~row_knownOff & ((1ULL << N) - 1);

  if (__builtin_popcount(row_knownOn) == 1) {
    for (; remaining; remaining &= remaining - 1) {
      uint64_t lowest_bit = remaining & -remaining;

      Problem problem = {outcome.knownOn, outcome.knownOff, {0}};
      problem.knownOn[outcome.ix] |= lowest_bit;
      problem.seed[outcome.ix] |= lowest_bit;
      stack.push_back(problem);
    }
  }

  if (__builtin_popcount(row_knownOn) == 0) {
    for (; remaining; remaining &= remaining - 1) {
      uint64_t lowest_bit = remaining & -remaining;

      uint64_t remaining2 = remaining & ~lowest_bit;
      for (; remaining2; remaining2 &= remaining2 - 1) {
        uint64_t lowest_bit2 = remaining2 & -remaining2;

        Problem problem = {outcome.knownOn, outcome.knownOff, {0}};
        problem.knownOn[outcome.ix] |= lowest_bit | lowest_bit2;
        problem.seed[outcome.ix] |= lowest_bit | lowest_bit2;
        stack.push_back(problem);
      }
    }
  }

}

int main() {
  // auto [knownOn, knownOff] = parse_rle_history("");
  // Problem problem = {knownOn, knownOff, {0}};

  std::vector<Problem> stack = {};

  Outcome blank = {{0}, {0}, false, true, N*N, 0};
  resolve_outcome(blank, stack);

  while (!stack.empty()) {
    // std::cout << stack.size() << std::endl;
    // std::cout << "x = 10, y = 10, rule = LifeHistory" << std::endl;
    // std::cout << to_rle_history(stack.back().knownOn, stack.back().knownOff) << std::endl;

    size_t batch_size = std::min(stack.size(), static_cast<size_t>(MAX_BATCH_SIZE));

    std::vector<Problem> batch(stack.end() - batch_size, stack.end());
    stack.resize(stack.size() - batch_size);

    std::vector<Outcome> outcomes = launch_work_kernel<N>(batch_size, batch);

    for (auto &outcome : outcomes) {
      // std::cout << "x = 10, y = 10, rule = LifeHistory" << std::endl;
      // std::cout << to_rle_history(outcome.knownOn, outcome.knownOff) << std::endl;
      // std::cout << "solved: " << outcome.solved << std::endl;
      // std::cout << "consistent: " << outcome.consistent << std::endl;
      // std::cout << "unknown_pop: " << outcome.unknownPop << std::endl;
      // std::cout << "ix: " << outcome.ix << std::endl;

      resolve_outcome(outcome, stack);
    }
  }
  return 0;

  // std::cout << "x = 10, y = 10, rule = LifeHistory" << std::endl;
  // std::cout << to_rle_history(result.knownOn, result.knownOff) << std::endl;
  // std::cout << "solved: " << result.solved << std::endl;
  // std::cout << "consistent: " << result.consistent << std::endl;
  // std::cout << "unknown_pop: " << result.unknownPop << std::endl;
  // std::cout << "ix: " << result.ix << std::endl;
}
