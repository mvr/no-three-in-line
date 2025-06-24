#include <iostream>
#include <tuple>

#include "parsing.hpp"

#include "three_kernel.hpp"

#include "params.hpp"

template<unsigned W>
void resolve_outcome(Outcome<W> &outcome, std::vector<Problem<W>> &stack) {
  if (!outcome.consistent) {
    return;
  }
  if (outcome.solved) {
    std::cout << to_rle<N, W>(outcome.knownOn) << std::endl;
    return;
  }

  using row_t = std::conditional_t<W == 64, uint64_t, uint32_t>;
  
  row_t line_knownOn, line_knownOff;
  if (outcome.axis == Axis::Horizontal) {
    line_knownOn = outcome.knownOn[outcome.ix];
    line_knownOff = outcome.knownOff[outcome.ix];
  } else { // Axis::Vertical
    line_knownOn = 0;
    line_knownOff = 0;
    for (unsigned r = 0; r < N; r++) {
      if (outcome.knownOn[r] & ((row_t)1 << outcome.ix)) {
        line_knownOn |= (row_t)1 << r;
      }
      if (outcome.knownOff[r] & ((row_t)1 << outcome.ix)) {
        line_knownOff |= (row_t)1 << r;
      }
    }
  }
  
  row_t remaining = ~line_knownOn & ~line_knownOff & (((row_t)1 << N) - 1);

  unsigned on_count;
  if constexpr (W == 64) {
    on_count = __builtin_popcountll(line_knownOn);
  } else {
    on_count = __builtin_popcount(line_knownOn);
  }

  if (on_count == 1) {
    for (; remaining; remaining &= remaining - 1) {
      row_t lowest_bit = remaining & -remaining;

      Problem<W> problem = {outcome.knownOn, outcome.knownOff, {}};
      if (outcome.axis == Axis::Horizontal) {
        problem.knownOn[outcome.ix] |= lowest_bit;
        problem.seed[outcome.ix] |= lowest_bit;
      } else { // Axis::Vertical
        unsigned r;
        if constexpr (W == 64) {
          r = __builtin_ctzll(lowest_bit);
        } else {
          r = __builtin_ctz(lowest_bit);
        }
        problem.knownOn[r] |= (row_t)1 << outcome.ix;
        problem.seed[r] |= (row_t)1 << outcome.ix;
      }
      stack.push_back(problem);
    }
  }

  if (on_count == 0) {
    for (; remaining; remaining &= remaining - 1) {
      row_t lowest_bit = remaining & -remaining;

      row_t remaining2 = remaining & ~lowest_bit;
      for (; remaining2; remaining2 &= remaining2 - 1) {
        row_t lowest_bit2 = remaining2 & -remaining2;

        Problem<W> problem = {outcome.knownOn, outcome.knownOff, {}};
        if (outcome.axis == Axis::Horizontal) {
          problem.knownOn[outcome.ix] |= lowest_bit | lowest_bit2;
          problem.seed[outcome.ix] |= lowest_bit | lowest_bit2;
        } else { // Axis::Vertical
          unsigned r1, r2;
          if constexpr (W == 64) {
            r1 = __builtin_ctzll(lowest_bit);
            r2 = __builtin_ctzll(lowest_bit2);
          } else {
            r1 = __builtin_ctz(lowest_bit);
            r2 = __builtin_ctz(lowest_bit2);
          }
          problem.knownOn[r1] |= (row_t)1 << outcome.ix;
          problem.knownOn[r2] |= (row_t)1 << outcome.ix;
          problem.seed[r1] |= (row_t)1 << outcome.ix;
          problem.seed[r2] |= (row_t)1 << outcome.ix;
        }
        stack.push_back(problem);
      }
    }
  }

}

template<unsigned W>
int solve_main() {
  init_lookup_tables_host();

  // auto [knownOn, knownOff] = parse_rle_history("");
  // Problem<W> problem = {knownOn, knownOff, {}};

  std::vector<Problem<W>> stack = {};

  Outcome<W> blank = {{}, {}, false, true, N*N, Axis::Horizontal, 0};
  resolve_outcome<W>(blank, stack);

  while (!stack.empty()) {
    // std::cout << stack.size() << std::endl;
    // std::cout << "x = 10, y = 10, rule = LifeHistory" << std::endl;
    // std::cout << to_rle_history(stack.back().knownOn, stack.back().knownOff) << std::endl;

    size_t batch_size = std::min(stack.size(), static_cast<size_t>(MAX_BATCH_SIZE));

    std::vector<Problem<W>> batch(stack.end() - batch_size, stack.end());
    stack.resize(stack.size() - batch_size);

    std::vector<Outcome<W>> outcomes = launch_work_kernel<N, W>(batch_size, batch);

    for (auto &outcome : outcomes) {
      // std::cout << "x = 10, y = 10, rule = LifeHistory" << std::endl;
      // std::cout << to_rle_history<N, W>(outcome.knownOn, outcome.knownOff) << std::endl;
      // std::cout << "solved: " << outcome.solved << std::endl;
      // std::cout << "consistent: " << outcome.consistent << std::endl;
      // std::cout << "unknown_pop: " << outcome.unknownPop << std::endl;
      // std::cout << "ix: " << outcome.ix << std::endl;

      resolve_outcome<W>(outcome, stack);
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

int main() {
  // Choose bit width based on problem size or configuration
  if (N > 32) {
    return solve_main<64>();
  } else {
    return solve_main<32>();
  }
}

// Last pre-multistream commit:
// https://gitlab.com/apgoucher/silk/-/commit/f4005091b4093f403e62570a44d135347d1f012f
