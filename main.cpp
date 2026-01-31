#include "three_kernel.hpp"

#include "params.hpp"
#include "parsing.hpp"

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::string seed_on_rle;
  std::string seed_off_rle;
  bool has_seed = false;
  int gpu = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto consume_value = [&](const char *flag, std::string &out) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      out = argv[++i];
      has_seed = true;
      return true;
    };

    if (arg == "--seed-on") {
      if (!consume_value("--seed-on", seed_on_rle)) {
        std::cerr << "Failed to consume --seed-on\n";
        return 1;
      }
      continue;
    }
    if (arg.rfind("--seed-on=", 0) == 0) {
      seed_on_rle = arg.substr(10);
      has_seed = true;
      continue;
    }
    if (arg == "--seed-off") {
      if (!consume_value("--seed-off", seed_off_rle)) {
        std::cerr << "Failed to consume --seed-off\n";
        return 1;
      }
      continue;
    }
    if (arg.rfind("--seed-off=", 0) == 0) {
      seed_off_rle = arg.substr(11);
      has_seed = true;
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
      std::cerr << "Usage: three [--seed-on RLE] [--seed-off RLE] [--gpu N]\n";
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

  // if (N > 32) {
  //   if (has_seed) {
  //     board_array_t<64> seed_on = seed_on_rle.empty() ? board_array_t<64>{} : parse_rle<64>(seed_on_rle);
  //     board_array_t<64> seed_off = seed_off_rle.empty() ? board_array_t<64>{} : parse_rle<64>(seed_off_rle);
  //     return solve_with_device_stack<N, 64>(&seed_on, &seed_off);
  //   }
  //   return solve_with_device_stack<N, 64>();
  // } else {
    if (has_seed) {
      board_array_t<32> seed_on = seed_on_rle.empty() ? board_array_t<32>{} : parse_rle<32>(seed_on_rle);
      board_array_t<32> seed_off = seed_off_rle.empty() ? board_array_t<32>{} : parse_rle<32>(seed_off_rle);
      return solve_with_device_stack<N, 32>(&seed_on, &seed_off);
    }
    return solve_with_device_stack<N, 32>();
  // }
}

// Last pre-multistream commit:
// https://gitlab.com/apgoucher/silk/-/commit/f4005091b4093f403e62570a44d135347d1f012f
