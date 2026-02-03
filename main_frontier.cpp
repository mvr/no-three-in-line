#include "three_kernel.hpp"
#include "params.hpp"

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  FrontierConfig config{};
  int gpu = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto consume_uint = [&](const char *flag, unsigned &out) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      out = static_cast<unsigned>(std::stoul(argv[++i]));
      return true;
    };

    if (arg == "--min-on") {
      if (!consume_uint("--min-on", config.min_on)) return 1;
      config.use_min_on = true;
      continue;
    }
    if (arg.rfind("--min-on=", 0) == 0) {
      config.min_on = static_cast<unsigned>(std::stoul(arg.substr(9)));
      config.use_min_on = true;
      continue;
    }
    if (arg == "--steps") {
      if (!consume_uint("--steps", config.max_steps)) return 1;
      continue;
    }
    if (arg.rfind("--steps=", 0) == 0) {
      config.max_steps = static_cast<unsigned>(std::stoul(arg.substr(8)));
      continue;
    }
    if (arg == "--buffer-cap") {
      if (!consume_uint("--buffer-cap", config.buffer_capacity)) return 1;
      continue;
    }
    if (arg.rfind("--buffer-cap=", 0) == 0) {
      config.buffer_capacity = static_cast<unsigned>(std::stoul(arg.substr(13)));
      continue;
    }
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
      std::cerr << "Usage: three_frontier [--min-on N] [--steps N] "
                   "[--buffer-cap N] [--gpu N]\n";
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

  if (N > 32) {
    return solve_with_device_stack<N, 64>(config);
  }
  return solve_with_device_stack<N, 32>(config);
}
