#pragma once

#include "common.hpp"
#include "board.cu"

struct ForceCandidate {
  bool has_force = false;
  bool force_on = false;
  bool on_b = false;
  cuda::std::pair<int, int> cell{0, 0};
};

template<unsigned W>
_DI_ LexStatus compare_with_unknowns(const BitBoard<W> a_on, const BitBoard<W> a_off,
                                     const BitBoard<W> b_on, const BitBoard<W> b_off,
                                     const BitBoard<W> bounds) {
  // We need to find the first differing bit position where:
  // - Both are known (not unknown)
  // - They differ in value

  BitBoard<W> a_unknown = ~(a_on | a_off);
  BitBoard<W> b_unknown = ~(b_on | b_off);

  BitBoard<W> both_known = ~(a_unknown | b_unknown);
  BitBoard<W> diff = (a_on ^ b_on) & both_known;

  if (diff.empty()) {
    // Check if any unknowns exist that could change the comparison
    BitBoard<W> critical_unknowns = (a_unknown | b_unknown) & bounds;
    return critical_unknowns.empty() ? LexStatus::Equal : LexStatus::Unknown;
  }

  auto cell = diff.first_on();

  BitBoard<W> before_mask = BitBoard<W>::positions_before(cell) & bounds;

  if (a_on.get(cell)) {
    // a = 1, b = 0 at first difference
    // But we need to check if there's an earlier unknown that could flip this

    BitBoard<W> critical_before = before_mask & (
        (a_unknown & b_on) |   // a unknown, b = 1 (could make a < b)
        (a_off & b_unknown) |  // a = 0, b unknown (could make a < b)
        (a_unknown & b_unknown) // both unknown (could become different)
    );

    return critical_before.empty() ? LexStatus::Greater : LexStatus::Unknown;
  } else {
    // a = 0, b = 1 at first difference

    BitBoard<W> critical_before = before_mask & (
        (a_unknown & b_off) |   // a unknown, b = 0 (could make a > b)
        (a_on & b_unknown) |    // a = 1, b unknown (could make a > b)
        (a_unknown & b_unknown) // both unknown (could become different)
    );

    return critical_before.empty() ? LexStatus::Less : LexStatus::Unknown;
  }
}

template<unsigned W>
_DI_ LexStatus compare_with_unknowns_forced(const BitBoard<W> a_on, const BitBoard<W> a_off,
                                            const BitBoard<W> b_on, const BitBoard<W> b_off,
                                            const BitBoard<W> bounds,
                                            ForceCandidate &forced) {
  BitBoard<W> a_unknown = ~(a_on | a_off);
  BitBoard<W> b_unknown = ~(b_on | b_off);

  LexStatus order = compare_with_unknowns<W>(a_on, a_off, b_on, b_off, bounds);
  if (order == LexStatus::Unknown) {
    BitBoard<W> critical_unknowns = (a_unknown | b_unknown) & bounds;
    if (!critical_unknowns.empty()) {
      auto candidate = critical_unknowns.first_on();
      if (a_on.get(candidate) && b_unknown.get(candidate)) {
        forced.has_force = true;
        forced.force_on = true;
        forced.on_b = true;
        forced.cell = candidate;
      } else if (a_unknown.get(candidate) && b_off.get(candidate)) {
        forced.has_force = true;
        forced.force_on = false;
        forced.on_b = false;
        forced.cell = candidate;
      }
    }
  }
  return order;
}
