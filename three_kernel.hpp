#pragma once

#include <stdint.h>

#include "common.hpp"

template<unsigned W>
struct Problem {
  board_array_t<W> known_on;
  board_array_t<W> known_off;
};

enum class StatId : unsigned {
  NodesVisited,
  VulnerableBranches,
  SymmetryForced,
  CellBranches,
  RowBranches,
  CanonicalSkips,
  Solutions,
  InconsistentNodes,
  Count
};

enum class SearchMode : unsigned {
  Normal,
  Frontier,
  Seed
};

template <unsigned W>
struct SearchOptions {
  SearchMode mode = SearchMode::Normal;
  unsigned frontier_min_on = 0;
  board_array_t<W> seed_on{};
  board_array_t<W> seed_off{};
  bool first_solution = false;
  unsigned time_limit_seconds = 0;
  unsigned stats_interval_seconds = 10;
};

template <unsigned W>
struct OutputBuffer {
  Problem<W> *entries;
  unsigned size;
  unsigned overflow;
  unsigned capacity;
};

template <unsigned W>
struct DeviceStack {
  Problem<W> problems[STACK_CAPACITY];
  unsigned size;
  unsigned overflow;
};

template <unsigned N, unsigned W>
int solve_with_device_stack(const SearchOptions<W> &options);
