#pragma once

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>

struct SearchCliOptions {
  std::string seed_rle;
  std::string seed_on_rle;
  std::string seed_off_rle;
  bool has_seed = false;
  bool use_frontier = false;
  unsigned frontier_min_on = 0;
  int gpu = -1;
  bool show_help = false;
};

inline bool parse_search_cli(int argc,
                             char **argv,
                             const char *prog,
                             SearchCliOptions &out) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto consume_value = [&](const char *flag, std::string &value) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      value = argv[++i];
      return true;
    };

    auto consume_uint = [&](const char *flag, unsigned &value) -> bool {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      value = static_cast<unsigned>(std::stoul(argv[++i]));
      return true;
    };

    auto parse_seed = [&](const std::string &value) {
      out.seed_rle = value;
      const size_t sep = value.find('|');
      if (sep == std::string::npos) {
        out.seed_on_rle = value;
        out.seed_off_rle.clear();
      } else {
        out.seed_on_rle = value.substr(0, sep);
        out.seed_off_rle = value.substr(sep + 1);
      }
      out.has_seed = true;
    };

    if (arg == "--seed") {
      std::string value;
      if (!consume_value("--seed", value)) {
        return false;
      }
      parse_seed(value);
      continue;
    }
    if (arg.rfind("--seed=", 0) == 0) {
      parse_seed(arg.substr(7));
      continue;
    }
    if (arg == "--frontier") {
      if (!consume_uint("--frontier", out.frontier_min_on)) {
        return false;
      }
      out.use_frontier = true;
      continue;
    }
    if (arg.rfind("--frontier=", 0) == 0) {
      out.frontier_min_on = static_cast<unsigned>(std::stoul(arg.substr(11)));
      out.use_frontier = true;
      continue;
    }
    if (arg == "--gpu") {
      std::string value;
      if (!consume_value("--gpu", value)) {
        return false;
      }
      out.gpu = std::stoi(value);
      continue;
    }
    if (arg.rfind("--gpu=", 0) == 0) {
      out.gpu = std::stoi(arg.substr(6));
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      std::cerr << "Usage: " << prog
                << " [--frontier MIN_ON] [--seed ON_RLE[|OFF_RLE]] [--gpu N]\n";
      out.show_help = true;
      return true;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    return false;
  }

  if (out.use_frontier && out.has_seed) {
    std::cerr << "Arguments --frontier and --seed are mutually exclusive\n";
    return false;
  }
  return true;
}

inline int apply_search_gpu(const SearchCliOptions &options) {
  if (options.gpu < 0) {
    return 0;
  }
  auto err = cudaSetDevice(options.gpu);
  if (err != cudaSuccess) {
    std::cerr << "cudaSetDevice(" << options.gpu << ") failed: "
              << cudaGetErrorString(err) << "\n";
    return 1;
  }
  return 0;
}
