#include <cuda/std/array>
#include <cuda/std/utility>

#include "gtest/gtest.h"

#include "parsing.hpp"
#include "board.cu"
#include "three_board.cu"
#include "three_board_c4.cu"

namespace {

template <unsigned N>
std::string rle_from_points(std::initializer_list<cuda::std::pair<int, int>> pts) {
  board_array_t<32> arr{};
  for (const auto &pt : pts) {
    arr[pt.second] |= board_row_t<32>(1u) << pt.first;
  }
  return to_rle<N, 32>(arr);
}

template <unsigned N>
std::string empty_rle() {
  board_array_t<32> arr{};
  return to_rle<N, 32>(arr);
}

template <unsigned N>
std::string first_arc_choice(const std::vector<cuda::std::pair<int, int>> &candidates) {
  board_array_t<32> arr{};
  for (const auto &pt : candidates) {
    arr[pt.second] |= board_row_t<32>(1u) << pt.first;
  }
  return to_rle<N, 32>(arr);
}

// TODO: use the symmetry helpers
template <unsigned N>
__device__ void expand_to_full(const ThreeBoardC4<N> &source,
                               typename ThreeBoardC4<N>::FullBoard &destination) {
  destination.known_on = typename ThreeBoardC4<N>::FullBitBoard();
  destination.known_off = typename ThreeBoardC4<N>::FullBitBoard();

  for (int y = 0; y < static_cast<int>(N); ++y) {
    const board_row_t<32> row_on = source.known_on.row(y);
    const board_row_t<32> row_off = source.known_off.row(y);

    for (int x = 0; x < static_cast<int>(N); ++x) {
      const board_row_t<32> bit = board_row_t<32>(1u) << x;
      const bool is_on = (row_on & bit) != 0;
      const bool is_off = (row_off & bit) != 0;

      if (!is_on && !is_off) {
        continue;
      }

      const auto orbit = ThreeBoardC4<N>::orbit({x, y});
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int actual_x = orbit[r].first;
        const int actual_y = orbit[r].second;
        const int board_x = actual_x + static_cast<int>(N);
        const int board_y = actual_y + static_cast<int>(N);

        if (is_on) {
          destination.known_on.set(board_x, board_y);
        }
        if (is_off) {
          destination.known_off.set(board_x, board_y);
        }
      }
    }
  }
}

// TODO: use the symmetry helpers
template <unsigned N>
__device__ void project_to_fundamental(const typename ThreeBoardC4<N>::FullBoard &full_board,
                                       BitBoard<32> &proj_on,
                                       BitBoard<32> &proj_off) {
  proj_on = BitBoard<32>();
  proj_off = BitBoard<32>();

  for (int y = 0; y < static_cast<int>(N); ++y) {
    for (int x = 0; x < static_cast<int>(N); ++x) {
      const int board_x = x + static_cast<int>(N);
      const int board_y = y + static_cast<int>(N);

      if (full_board.known_on.get(board_x, board_y)) {
        proj_on.set(x, y);
      }
      if (full_board.known_off.get(board_x, board_y)) {
        proj_off.set(x, y);
      }
    }
  }

  const BitBoard<32> bds = ThreeBoardC4<N>::bounds();
  proj_on &= bds;
  proj_off &= bds;
}

// TODO: use the symmetry helpers
template <unsigned N>
__device__ BitBoard<32> project_mask_to_fundamental(const typename ThreeBoardC4<N>::FullBitBoard &mask) {
  BitBoard<32> proj;

  for (int y = 0; y < static_cast<int>(N); ++y) {
    for (int x = 0; x < static_cast<int>(N); ++x) {
      const unsigned fx = static_cast<unsigned>(x + N);
      const unsigned fy = static_cast<unsigned>(y + N);

      if (mask.get(fx, fy)) {
        proj.set(x, y);
      }
    }
  }

  proj &= ThreeBoardC4<N>::bounds();
  return proj;
}

template <unsigned N>
__global__ void force_compare_kernel(board_row_t<32> *known_on,
                                     board_row_t<32> *known_off,
                                     board_row_t<32> *c4_on_out,
                                     board_row_t<32> *c4_off_out,
                                     board_row_t<32> *full_on_out,
                                     board_row_t<32> *full_off_out,
                                     bool *c4_consistency,
                                     bool *full_consistency) {
  ThreeBoardC4<N> c4_board;
  c4_board.known_on = BitBoard<32>::load(known_on);
  c4_board.known_off = BitBoard<32>::load(known_off);

  ThreeBoardC4<N> forced_c4 = c4_board.force_orthogonal();
  forced_c4.known_on.save(c4_on_out);
  forced_c4.known_off.save(c4_off_out);
  const bool c4_ok = forced_c4.consistent();
  if ((threadIdx.x & 31) == 0) {
    *c4_consistency = c4_ok;
  }

  typename ThreeBoardC4<N>::FullBoard full_board;
  expand_to_full(c4_board, full_board);

  typename ThreeBoardC4<N>::FullBoard forced_full = full_board.force_orthogonal();

  BitBoard<32> proj_on;
  BitBoard<32> proj_off;
  project_to_fundamental<N>(forced_full, proj_on, proj_off);

  proj_on.save(full_on_out);
  proj_off.save(full_off_out);
  const bool full_ok = forced_full.consistent();
  if ((threadIdx.x & 31) == 0) {
    *full_consistency = full_ok;
  }
}

template <unsigned N>
__global__ void vulnerable_compare_kernel(board_row_t<32> *known_on,
                                          board_row_t<32> *known_off,
                                          board_row_t<32> *c4_vulnerable_out,
                                          board_row_t<32> *full_vulnerable_out) {
  ThreeBoardC4<N> board;
  board.known_on = BitBoard<32>::load(known_on);
  board.known_off = BitBoard<32>::load(known_off);

  BitBoard<32> c4_vulnerable = board.vulnerable();
  c4_vulnerable.save(c4_vulnerable_out);

  typename ThreeBoardC4<N>::FullBoard full_board = board.expand_to_full();
  auto full_vulnerable = full_board.vulnerable();
  BitBoard<32> projected = project_mask_to_fundamental<N>(full_vulnerable);
  projected.save(full_vulnerable_out);
}

template <unsigned N>
void expect_force_matches(const std::string &known_on_rle,
                          const std::string &known_off_rle) {
  const auto host_known_on = parse_rle<32>(known_on_rle);
  const auto host_known_off = parse_rle<32>(known_off_rle);

  board_row_t<32> *d_known_on = nullptr;
  board_row_t<32> *d_known_off = nullptr;
  board_row_t<32> *d_c4_on = nullptr;
  board_row_t<32> *d_c4_off = nullptr;
  board_row_t<32> *d_full_on = nullptr;
  board_row_t<32> *d_full_off = nullptr;
  bool *d_c4_consistent = nullptr;
  bool *d_full_consistent = nullptr;

  const size_t bytes = sizeof(board_array_t<32>);

  cudaMalloc(&d_known_on, bytes);
  cudaMalloc(&d_known_off, bytes);
  cudaMalloc(&d_c4_on, bytes);
  cudaMalloc(&d_c4_off, bytes);
  cudaMalloc(&d_full_on, bytes);
  cudaMalloc(&d_full_off, bytes);
  cudaMalloc(&d_c4_consistent, sizeof(bool));
  cudaMalloc(&d_full_consistent, sizeof(bool));

  cudaMemcpy(d_known_on, host_known_on.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_known_off, host_known_off.data(), bytes, cudaMemcpyHostToDevice);

  force_compare_kernel<N><<<1, 32>>>(d_known_on, d_known_off, d_c4_on, d_c4_off, d_full_on, d_full_off,
                                     d_c4_consistent, d_full_consistent);
  cudaDeviceSynchronize();

  board_array_t<32> c4_on;
  board_array_t<32> c4_off;
  board_array_t<32> full_on;
  board_array_t<32> full_off;
  bool c4_consistent = true;
  bool full_consistent = true;

  cudaMemcpy(c4_on.data(), d_c4_on, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(c4_off.data(), d_c4_off, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(full_on.data(), d_full_on, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(full_off.data(), d_full_off, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(&c4_consistent, d_c4_consistent, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(&full_consistent, d_full_consistent, sizeof(bool), cudaMemcpyDeviceToHost);

  if (full_consistent) {
    EXPECT_TRUE(c4_consistent);
    if (c4_consistent) {
      EXPECT_EQ((to_rle<N, 32>(full_on)), (to_rle<N, 32>(c4_on)));
      EXPECT_EQ((to_rle<N, 32>(full_off)), (to_rle<N, 32>(c4_off)));
    }
  } else {
    EXPECT_FALSE(c4_consistent);
  }

  cudaFree(d_known_on);
  cudaFree(d_known_off);
  cudaFree(d_c4_on);
  cudaFree(d_c4_off);
  cudaFree(d_full_on);
  cudaFree(d_full_off);
  cudaFree(d_c4_consistent);
  cudaFree(d_full_consistent);
}

template <unsigned N>
__global__ void eliminate_compare_kernel(board_row_t<32> *known_on,
                                         board_row_t<32> *known_off,
                                         board_row_t<32> *seed_mask,
                                         board_row_t<32> *c4_on_out,
                                         board_row_t<32> *c4_off_out,
                                         board_row_t<32> *full_on_out,
                                         board_row_t<32> *full_off_out,
                                         bool *c4_consistency,
                                         bool *full_consistency) {
  ThreeBoardC4<N> base;
  base.known_on = BitBoard<32>::load(known_on);
  base.known_off = BitBoard<32>::load(known_off);

  BitBoard<32> seeds = BitBoard<32>::load(seed_mask);

  ThreeBoardC4<N> c4_after = base;
  c4_after.eliminate_all_lines(seeds);
  c4_after.known_on.save(c4_on_out);
  c4_after.known_off.save(c4_off_out);
  const bool c4_ok = c4_after.consistent();
  if ((threadIdx.x & 31) == 0) {
    *c4_consistency = c4_ok;
  }

  typename ThreeBoardC4<N>::FullBoard full = base.expand_to_full();
  typename ThreeBoardC4<N>::FullBitBoard full_seeds = base.expand_mask(seeds);
  full.eliminate_all_lines_unfiltered(full_seeds);

  ThreeBoardC4<N> projected;
  projected.project_from_full(full);
  projected.known_on.save(full_on_out);
  projected.known_off.save(full_off_out);
  const bool full_ok = full.consistent();
  if ((threadIdx.x & 31) == 0) {
    *full_consistency = full_ok;
  }
}

template <unsigned N>
void expect_eliminate_matches(const std::string &known_on_rle,
                              const std::string &known_off_rle,
                              const std::string &seed_rle) {
  init_lookup_tables_host();
  init_relevant_endpoint_host(ThreeBoardC4<N>::FULL_N);
  init_relevant_endpoint_host_64(ThreeBoardC4<N>::FULL_N);

  const auto host_known_on = parse_rle<32>(known_on_rle);
  const auto host_known_off = parse_rle<32>(known_off_rle);
  const auto host_seed = parse_rle<32>(seed_rle);

  board_row_t<32> *d_known_on = nullptr;
  board_row_t<32> *d_known_off = nullptr;
  board_row_t<32> *d_seed = nullptr;
  board_row_t<32> *d_c4_on = nullptr;
  board_row_t<32> *d_c4_off = nullptr;
  board_row_t<32> *d_full_on = nullptr;
  board_row_t<32> *d_full_off = nullptr;
  bool *d_c4_consistent = nullptr;
  bool *d_full_consistent = nullptr;

  const size_t bytes = sizeof(board_array_t<32>);

  cudaMalloc(&d_known_on, bytes);
  cudaMalloc(&d_known_off, bytes);
  cudaMalloc(&d_seed, bytes);
  cudaMalloc(&d_c4_on, bytes);
  cudaMalloc(&d_c4_off, bytes);
  cudaMalloc(&d_full_on, bytes);
  cudaMalloc(&d_full_off, bytes);
  cudaMalloc(&d_c4_consistent, sizeof(bool));
  cudaMalloc(&d_full_consistent, sizeof(bool));

  cudaMemcpy(d_known_on, host_known_on.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_known_off, host_known_off.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_seed, host_seed.data(), bytes, cudaMemcpyHostToDevice);

  eliminate_compare_kernel<N><<<1, 32>>>(d_known_on, d_known_off, d_seed,
                                          d_c4_on, d_c4_off, d_full_on, d_full_off,
                                          d_c4_consistent, d_full_consistent);
  cudaDeviceSynchronize();

  board_array_t<32> c4_on;
  board_array_t<32> c4_off;
  board_array_t<32> full_on;
  board_array_t<32> full_off;
  bool c4_consistent = true;
  bool full_consistent = true;

  cudaMemcpy(c4_on.data(), d_c4_on, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(c4_off.data(), d_c4_off, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(full_on.data(), d_full_on, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(full_off.data(), d_full_off, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(&c4_consistent, d_c4_consistent, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(&full_consistent, d_full_consistent, sizeof(bool), cudaMemcpyDeviceToHost);

  if (full_consistent) {
    EXPECT_TRUE(c4_consistent);
    if (c4_consistent) {
      EXPECT_EQ((to_rle<N, 32>(full_on)), (to_rle<N, 32>(c4_on)));
      EXPECT_EQ((to_rle<N, 32>(full_off)), (to_rle<N, 32>(c4_off)));
    }
  } else {
    EXPECT_FALSE(c4_consistent);
  }

  cudaFree(d_known_on);
  cudaFree(d_known_off);
  cudaFree(d_seed);
  cudaFree(d_c4_on);
  cudaFree(d_c4_off);
  cudaFree(d_full_on);
  cudaFree(d_full_off);
  cudaFree(d_c4_consistent);
  cudaFree(d_full_consistent);
}

template <unsigned N>
void compute_vulnerable_rles(const std::string &known_on_rle,
                             const std::string &known_off_rle,
                             std::string &c4_rle,
                             std::string &full_rle) {
  const auto host_known_on = parse_rle<32>(known_on_rle);
  const auto host_known_off = parse_rle<32>(known_off_rle);

  board_row_t<32> *d_known_on = nullptr;
  board_row_t<32> *d_known_off = nullptr;
  board_row_t<32> *d_c4_vulnerable = nullptr;
  board_row_t<32> *d_full_vulnerable = nullptr;

  const size_t bytes = sizeof(board_array_t<32>);

  cudaMalloc(&d_known_on, bytes);
  cudaMalloc(&d_known_off, bytes);
  cudaMalloc(&d_c4_vulnerable, bytes);
  cudaMalloc(&d_full_vulnerable, bytes);

  cudaMemcpy(d_known_on, host_known_on.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_known_off, host_known_off.data(), bytes, cudaMemcpyHostToDevice);

  vulnerable_compare_kernel<N><<<1, 32>>>(d_known_on, d_known_off,
                                          d_c4_vulnerable, d_full_vulnerable);
  cudaDeviceSynchronize();

  board_array_t<32> c4_vulnerable;
  board_array_t<32> full_vulnerable;
  cudaMemcpy(c4_vulnerable.data(), d_c4_vulnerable, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(full_vulnerable.data(), d_full_vulnerable, bytes, cudaMemcpyDeviceToHost);

  c4_rle = to_rle<N, 32>(c4_vulnerable);
  full_rle = to_rle<N, 32>(full_vulnerable);

  cudaFree(d_known_on);
  cudaFree(d_known_off);
  cudaFree(d_c4_vulnerable);
  cudaFree(d_full_vulnerable);
}

template <unsigned N>
__global__ void first_near_radius_on_kernel(const board_row_t<32> *mask,
                                         cuda::std::pair<int, int> *out) {
  BitBoard<32> board = BitBoard<32>::load(mask);
  *out = board.first_near_radius_on<N>();
}

template <unsigned N>
void expect_first_near_radius_on(const std::string &mask_rle,
                              cuda::std::pair<int, int> expected) {
  board_array_t<32> host_mask = parse_rle<32>(mask_rle);
  board_row_t<32> *d_mask;
  cuda::std::pair<int, int> *d_out;
  cudaMalloc(&d_mask, sizeof(board_array_t<32>));
  cudaMalloc(&d_out, sizeof(cuda::std::pair<int, int>));
  cudaMemcpy(d_mask, host_mask.data(), sizeof(board_array_t<32>), cudaMemcpyHostToDevice);
  first_near_radius_on_kernel<N><<<1, 32>>>(d_mask, d_out);
  cuda::std::pair<int, int> result;
  cudaMemcpy(&result, d_out, sizeof(result), cudaMemcpyDeviceToHost);
  EXPECT_EQ(expected.first, result.first);
  EXPECT_EQ(expected.second, result.second);
  cudaFree(d_mask);
  cudaFree(d_out);
}

template <unsigned N>
__global__ void expand_project_kernel(board_row_t<32> *known_on,
                                      board_row_t<32> *known_off,
                                      board_row_t<32> *projected_on,
                                      board_row_t<32> *projected_off) {
  ThreeBoardC4<N> base;
  base.known_on = BitBoard<32>::load(known_on);
  base.known_off = BitBoard<32>::load(known_off);

  typename ThreeBoardC4<N>::FullBoard full;
  expand_to_full(base, full);

  BitBoard<32> proj_on;
  BitBoard<32> proj_off;
  project_to_fundamental<N>(full, proj_on, proj_off);

  proj_on.save(projected_on);
  proj_off.save(projected_off);
}

template <unsigned N>
void expect_round_trip(const std::string &known_on_rle,
                       const std::string &known_off_rle) {
  const auto host_known_on = parse_rle<32>(known_on_rle);
  const auto host_known_off = parse_rle<32>(known_off_rle);

  board_row_t<32> *d_known_on = nullptr;
  board_row_t<32> *d_known_off = nullptr;
  board_row_t<32> *d_proj_on = nullptr;
  board_row_t<32> *d_proj_off = nullptr;

  const size_t bytes = sizeof(board_array_t<32>);

  cudaMalloc(&d_known_on, bytes);
  cudaMalloc(&d_known_off, bytes);
  cudaMalloc(&d_proj_on, bytes);
  cudaMalloc(&d_proj_off, bytes);

  cudaMemcpy(d_known_on, host_known_on.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_known_off, host_known_off.data(), bytes, cudaMemcpyHostToDevice);

  expand_project_kernel<N><<<1, 32>>>(d_known_on, d_known_off, d_proj_on, d_proj_off);
  cudaDeviceSynchronize();

  board_array_t<32> proj_on;
  board_array_t<32> proj_off;

  cudaMemcpy(proj_on.data(), d_proj_on, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(proj_off.data(), d_proj_off, bytes, cudaMemcpyDeviceToHost);

  EXPECT_EQ((to_rle<N, 32>(host_known_on)), (to_rle<N, 32>(proj_on)));
  EXPECT_EQ((to_rle<N, 32>(host_known_off)), (to_rle<N, 32>(proj_off)));

  cudaFree(d_known_on);
  cudaFree(d_known_off);
  cudaFree(d_proj_on);
  cudaFree(d_proj_off);
}

}  // namespace

TEST(ThreeBoardC4Helpers, ExpandProjectRoundTrip_SingleOn) {
  expect_round_trip<4>("o$4b$4b$4b!", "!");
}

TEST(ThreeBoardC4Helpers, ExpandProjectRoundTrip_MixedOnOff) {
  expect_round_trip<4>("o2bo$2b$2bo$4b!", "bobo$bob$4b$4b!");
}

TEST(ThreeBoardC4, ForceMatchesFullBoard_Dot) {
  expect_force_matches<4>("o!", "!");
  expect_force_matches<4>("3$3bo!", "!");
  expect_force_matches<4>("2$3bo!", "!");
}

TEST(ThreeBoardC4, ForceMatchesFullBoard_Larger) {
  expect_force_matches<4>("$3bo$3bo!", "!");
  expect_force_matches<6>("bo$2bo!", "!");
  expect_force_matches<6>("$3bo$5bo!", "!");
}

TEST(ThreeBoardC4, EliminateMatchesFullBoard_Diagonal) {
  expect_eliminate_matches<4>("o3b$bo2b$4b$4b!", "!", "o3b$bo2b$4b$4b!");
}

TEST(ThreeBoardC4, EliminateMatchesFullBoard_Offset) {
  expect_eliminate_matches<6>("o5b$6b$bo4b$6b$6b$6b!", "!", "o5b$6b$bo4b$6b$6b$6b!");
}

TEST(ThreeBoardC4, EliminateMatchesFullBoard_SelfOrbit) {
  expect_eliminate_matches<6>(rle_from_points<6>({{1, 1}}), empty_rle<6>(), rle_from_points<6>({{1, 1}}));
}

TEST(ThreeBoardC4, EliminateMatchesFullBoard_TwoSeeds) {
  expect_eliminate_matches<6>(rle_from_points<6>({{1, 1}, {4, 3}}), empty_rle<6>(),
                              rle_from_points<6>({{1, 1}, {4, 3}}));
}

TEST(ThreeBoardC4, EliminateMatchesFullBoard_WithKnownOff) {
  expect_eliminate_matches<6>(rle_from_points<6>({{1, 1}}), rle_from_points<6>({{0, 0}, {2, 4}}),
                              rle_from_points<6>({{3, 2}}));
}

TEST(BitBoardClosestToRadius, PrefersOuterArc) {
  // row 0: bits at columns 1 (close to centre) and 3 (closer to radius for N=6)
  expect_first_near_radius_on<6>(first_arc_choice<6>({{1, 0}, {3, 0}}), {3, 0});
}

TEST(BitBoardClosestToRadius, EmptyRow) {
  expect_first_near_radius_on<6>(empty_rle<6>(), {-1, -1});
}

TEST(BitBoardClosestToRadius, ChoosesBetterRow) {
  // row 0 @ col 4 vs row 1 @ col 5; row 1 is closer to the radius arc
  expect_first_near_radius_on<6>(first_arc_choice<6>({{4, 0}, {5, 1}}), {5, 1});
}

TEST(BitBoardClosestToRadius, BreaksTieWithinRow) {
  // row 2 picks column 3 over column 2 because it sits closer to the arc distance
  expect_first_near_radius_on<6>(first_arc_choice<6>({{2, 2}, {3, 2}}), {3, 2});
}

TEST(ThreeBoardC4, VulnerableMatchesFullBoard_Empty) {
  std::string c4, full;
  compute_vulnerable_rles<6>("!", "!", c4, full);
  EXPECT_EQ(c4, full);
  EXPECT_EQ(c4, "!");
}

TEST(ThreeBoardC4, VulnerableMatchesFullBoard_ZeroTriggered) {
  std::string c4, full;
  compute_vulnerable_rles<6>("!", "o2b3o$o2$o$o$o!", c4, full);

  EXPECT_EQ(c4, full);
  EXPECT_NE(c4, "!");
}

TEST(ThreeBoardC4, VulnerableMatchesFullBoard_OneTriggered) {
  std::string c4, full;
  compute_vulnerable_rles<6>("2bo!", "o2b3o$o2$o$o$o!", c4, full);

  EXPECT_EQ(c4, full);
  EXPECT_NE(c4, "!");
}
