#include "three_kernel_c4.hpp"
#include "params.hpp"

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  int gpu = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--gpu") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --gpu\n";
        return 1;
      }
      gpu = std::stoi(argv[++i]);
      continue;
    }
    if (arg.rfind("--gpu=", 0) == 0) {
      gpu = std::stoi(arg.substr(6));
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      std::cerr << "Usage: three_c4 [--gpu N]\n";
      return 0;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    return 1;
  }

  if (gpu >= 0) {
    auto err = cudaSetDevice(gpu);
    if (err != cudaSuccess) {
      std::cerr << "cudaSetDevice(" << gpu << ") failed: "
                << cudaGetErrorString(err) << "\n";
      return 1;
    }
  }

  return solve_with_device_stack_c4<N>();
}
