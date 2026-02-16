#include "three_kernel.hpp"

#include "params.hpp"
#include "search_cli.hpp"

int main(int argc, char **argv) {
  return run_search_cli_32(
      argc,
      argv,
      "three",
      [](const SearchOptions<32> &options) {
        return solve_with_device_stack<N, 32>(options);
      });
}
