#pragma once

#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "parsing.hpp"
#include "three_kernel.hpp"

struct SearchCliOptions {
  std::string seed_on_rle;
  std::string seed_off_rle;
  bool has_seed = false;
  bool use_frontier = false;
  unsigned frontier_min_on = 0;
  bool first_solution = false;
  unsigned time_limit_seconds = 0;
  unsigned stats_interval_seconds = 10;
  int gpu = -1;
  bool show_help = false;
};

inline void print_search_cli_usage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " [--frontier MIN_ON] [--seed ON_RLE[|OFF_RLE]] [--gpu N]"
            << " [--first-solution] [--time-limit SECONDS]"
            << " [--stats-interval SECONDS]\n";
}

inline bool parse_unsigned_arg(const std::string &s, unsigned &out) {
  if (s.empty()) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long v = std::strtoul(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0' ||
      v > std::numeric_limits<unsigned>::max()) {
    return false;
  }
  out = static_cast<unsigned>(v);
  return true;
}

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
      if (!parse_unsigned_arg(argv[++i], value)) {
        std::cerr << "Invalid value for " << flag << "\n";
        return false;
      }
      return true;
    };

    auto parse_seed = [&](const std::string &value) {
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
    if (arg == "--frontier") {
      if (!consume_uint("--frontier", out.frontier_min_on)) {
        return false;
      }
      out.use_frontier = true;
      continue;
    }
    if (arg == "--gpu") {
      unsigned gpu = 0;
      if (!consume_uint("--gpu", gpu)) {
        return false;
      }
      if (gpu > static_cast<unsigned>(std::numeric_limits<int>::max())) {
        std::cerr << "Invalid value for --gpu\n";
        return false;
      }
      out.gpu = static_cast<int>(gpu);
      continue;
    }
    if (arg == "--first-solution") {
      out.first_solution = true;
      continue;
    }
    if (arg == "--time-limit") {
      if (!consume_uint("--time-limit", out.time_limit_seconds)) {
        return false;
      }
      continue;
    }
    if (arg == "--stats-interval") {
      if (!consume_uint("--stats-interval", out.stats_interval_seconds)) {
        return false;
      }
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      print_search_cli_usage(prog);
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
  if (out.use_frontier && out.first_solution) {
    std::cerr << "Arguments --frontier and --first-solution are mutually exclusive\n";
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

template <unsigned W>
inline SearchOptions<W> build_search_options(const SearchCliOptions &cli) {
  SearchOptions<W> out{};
  out.first_solution = cli.first_solution;
  out.time_limit_seconds = cli.time_limit_seconds;
  out.stats_interval_seconds = cli.stats_interval_seconds;
  if (cli.use_frontier) {
    out.mode = SearchMode::Frontier;
    out.frontier_min_on = cli.frontier_min_on;
  } else if (cli.has_seed) {
    out.mode = SearchMode::Seed;
    out.seed_on = parse_rle<W>(cli.seed_on_rle);
    out.seed_off =
        cli.seed_off_rle.empty() ? board_array_t<W>{} : parse_rle<W>(cli.seed_off_rle);
  } else {
    out.mode = SearchMode::Normal;
  }
  return out;
}

template <typename Solve32>
inline int run_search_cli_32(int argc,
                             char **argv,
                             const char *prog,
                             Solve32 &&solve32) {
  SearchCliOptions options{};
  if (!parse_search_cli(argc, argv, prog, options)) {
    return options.show_help ? 0 : 1;
  }
  if (options.show_help) {
    return 0;
  }
  if (apply_search_gpu(options) != 0) {
    return 1;
  }
  const SearchOptions<32> search_options = build_search_options<32>(options);
  return solve32(search_options);
}

template <typename Solve32, typename Solve64>
inline int run_search_cli_32_or_64(int argc,
                                   char **argv,
                                   const char *prog,
                                   bool use_w64,
                                   Solve32 &&solve32,
                                   Solve64 &&solve64) {
  SearchCliOptions options{};
  if (!parse_search_cli(argc, argv, prog, options)) {
    return options.show_help ? 0 : 1;
  }
  if (options.show_help) {
    return 0;
  }
  if (apply_search_gpu(options) != 0) {
    return 1;
  }
  if (use_w64) {
    const SearchOptions<64> search_options = build_search_options<64>(options);
    return solve64(search_options);
  }
  const SearchOptions<32> search_options = build_search_options<32>(options);
  return solve32(search_options);
}
