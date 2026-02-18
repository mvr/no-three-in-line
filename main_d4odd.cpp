#include "three_kernel_d4odd.hpp"

#include "params.hpp"
#include "search_cli.hpp"

int main(int argc, char **argv) {
  return run_search_cli_32(
      argc,
      argv,
      "three_d4odd",
      [](const SearchOptions<32> &options) {
        return solve_with_device_stack_d4odd<N>(options);
      });
}

