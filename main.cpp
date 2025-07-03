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
    std::cout << to_rle<N, W>(outcome.known_on) << std::endl;
    return;
  }

  using row_t = std::conditional_t<W == 64, uint64_t, uint32_t>;
  
  row_t line_known_on, line_known_off;
  if (outcome.axis == Axis::Horizontal) {
    line_known_on = outcome.known_on[outcome.ix];
    line_known_off = outcome.known_off[outcome.ix];
  } else { // Axis::Vertical
    line_known_on = 0;
    line_known_off = 0;
    for (unsigned r = 0; r < N; r++) {
      if (outcome.known_on[r] & ((row_t)1 << outcome.ix)) {
        line_known_on |= (row_t)1 << r;
      }
      if (outcome.known_off[r] & ((row_t)1 << outcome.ix)) {
        line_known_off |= (row_t)1 << r;
      }
    }
  }

  row_t remaining = ~line_known_on & ~line_known_off & (((row_t)1 << N) - 1);

  unsigned on_count = popcount<W>(line_known_on);

  if (on_count == 1) {
    for (; remaining; remaining &= remaining - 1) {
      row_t lowest_bit = remaining & -remaining;

      Problem<W> problem = {outcome.known_on, outcome.known_off, {}};
      if (outcome.axis == Axis::Horizontal) {
        problem.known_on[outcome.ix] |= lowest_bit;
        problem.seed[outcome.ix] |= lowest_bit;
      } else { // Axis::Vertical
        unsigned r = count_trailing_zeros<W>(lowest_bit);
        problem.known_on[r] |= (row_t)1 << outcome.ix;
        problem.seed[r] |= (row_t)1 << outcome.ix;
      }
      stack.push_back(problem);
    }
  }

  if (on_count == 0) {
    row_t prev_offs = 0;
    for (; remaining; remaining &= remaining - 1) {
      row_t lowest_bit = remaining & -remaining;

      Problem<W> problem = {outcome.known_on, outcome.known_off, {}};

      if (outcome.axis == Axis::Horizontal) {
        problem.known_on[outcome.ix] |= lowest_bit;
        problem.known_off[outcome.ix] |= prev_offs;
        problem.seed[outcome.ix] |= lowest_bit;
      } else { // Axis::Vertical
        unsigned r = count_trailing_zeros<W>(lowest_bit);
        problem.known_on[r] |= (row_t)1 << outcome.ix;
        problem.seed[r] |= (row_t)1 << outcome.ix;

        row_t prev_offs_remaining = prev_offs;
        for (; prev_offs_remaining; prev_offs_remaining &= prev_offs_remaining - 1) {
          row_t lowest_bit = prev_offs_remaining & -prev_offs_remaining;
          unsigned r = count_trailing_zeros<W>(lowest_bit);
          problem.known_off[r] |= (row_t)1 << outcome.ix;
        }
      }

      stack.push_back(problem);

      prev_offs |= lowest_bit;
    }
  }

}

template<unsigned W>
int solve_main() {
  init_lookup_tables_host();
  init_relevant_endpoint_host();
  init_relevant_endpoint_host_64();

  std::vector<Problem<W>> stack = {};

  Outcome<W> blank = {{}, {}, false, true, N*N, Axis::Horizontal, 0};
  resolve_outcome<W>(blank, stack);

  DeviceMemory<W> device_mem(MAX_BATCH_SIZE);

  while (!stack.empty()) {
    size_t batch_size = std::min(stack.size(), static_cast<size_t>(MAX_BATCH_SIZE));

    std::vector<Problem<W>> batch(stack.end() - batch_size, stack.end());
    stack.resize(stack.size() - batch_size);

    std::vector<Outcome<W>> outcomes = launch_work_kernel<N, W>(batch_size, batch, device_mem);

    for (auto &outcome : outcomes) {
      // std::cout << "x = 12, y = 12, rule = LifeHistory" << std::endl;
      // std::cout << to_rle_history<N, W>(outcome.known_on, outcome.known_off) << std::endl;
      // std::cout << "solved: " << outcome.solved << std::endl;
      // std::cout << "consistent: " << outcome.consistent << std::endl;
      // std::cout << "unknown_pop: " << outcome.unknownPop << std::endl;
      // std::cout << "ix: " << outcome.ix << std::endl;

      resolve_outcome<W>(outcome, stack);
    }
  }

  return 0;
}

int main() {
  if (N > 32) {
    return solve_main<64>();
  } else {
    return solve_main<32>();
  }
}

// Last pre-multistream commit:
// https://gitlab.com/apgoucher/silk/-/commit/f4005091b4093f403e62570a44d135347d1f012f
