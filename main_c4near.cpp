#include "three_kernel_c4near.hpp"

#include "params.hpp"
#include "search_cli.hpp"

int main(int argc, char **argv) {
  return run_search_cli_32(
      argc,
      argv,
      "three_c4near",
      [](const SearchOptions<32> &options) {
        return solve_with_device_stack_c4near<N>(options);
      });
}
