#include "three_kernel.hpp"

#include "params.hpp"
#include "parsing.hpp"
#include "search_cli.hpp"

#include <iostream>

int main(int argc, char **argv) {
  SearchCliOptions options{};
  if (!parse_search_cli(argc, argv, "three", options)) {
    return options.show_help ? 0 : 1;
  }
  if (options.show_help) {
    return 0;
  }
  if (apply_search_gpu(options) != 0) {
    return 1;
  }

  board_array_t<32> seed_on =
      options.seed_on_rle.empty() ? board_array_t<32>{} : parse_rle<32>(options.seed_on_rle);
  board_array_t<32> seed_off =
      options.seed_off_rle.empty() ? board_array_t<32>{} : parse_rle<32>(options.seed_off_rle);

  if (options.use_frontier) {
    return solve_with_device_stack<N, 32>(options.frontier_min_on);
  }

  if (options.has_seed) {
    return solve_with_device_stack<N, 32>(&seed_on, &seed_off);
  }
  return solve_with_device_stack<N, 32>();
}
