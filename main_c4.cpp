#include "three_kernel_c4.hpp"

#include "params.hpp"
#include "search_cli.hpp"

int main(int argc, char **argv) {
  return run_search_cli_32_or_64(
      argc,
      argv,
      "three_c4",
      N > 32,
      [](const SearchOptions<32> &options) {
        return solve_with_device_stack_c4<N, 32>(options);
      },
      [](const SearchOptions<64> &options) {
        return solve_with_device_stack_c4<N, 64>(options);
      });
}
