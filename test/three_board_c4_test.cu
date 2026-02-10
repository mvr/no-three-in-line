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
using FullBoardT = ThreeBoard<ThreeBoardC4<N>::FULL_N, ThreeBoardC4<N>::FULL_W>;

template <unsigned N>
using FullBitBoardT = BitBoard<ThreeBoardC4<N>::FULL_W>;

template <unsigned N>
void init_c4_tables_for_test() {
  ThreeBoardC4<N, 32>::init_tables_host();
  FullBoardT<N>::init_tables_host();
}

template <unsigned N>
_DI_ FullBoardT<N> expand_to_full_test(const ThreeBoardC4<N> &base) {
  FullBoardT<N> full;
  for (int y = 0; y < static_cast<int>(N); ++y) {
    const board_row_t<32> row_on = base.known_on.row(y);
    const board_row_t<32> row_off = base.known_off.row(y);
    for (int x = 0; x < static_cast<int>(N); ++x) {
      const board_row_t<32> bit = board_row_t<32>(1u) << x;
      const bool is_on = (row_on & bit) != 0;
      const bool is_off = (row_off & bit) != 0;
      if (!is_on && !is_off) {
        continue;
      }
      ThreeBoardC4<N>::for_each_orbit_point({x, y}, [&](cuda::std::pair<int, int> pt) {
        const int full_x = pt.first + static_cast<int>(N);
        const int full_y = pt.second + static_cast<int>(N);
        if (is_on) {
          full.known_on.set(full_x, full_y);
        }
        if (is_off) {
          full.known_off.set(full_x, full_y);
        }
      });
    }
  }
  return full;
}

template <unsigned N>
_DI_ void project_from_full_test(const FullBoardT<N> &full, ThreeBoardC4<N> &out) {
  BitBoard<32> proj_on;
  BitBoard<32> proj_off;
  for (int y = 0; y < static_cast<int>(N); ++y) {
    for (int x = 0; x < static_cast<int>(N); ++x) {
      const unsigned full_x = static_cast<unsigned>(x + N);
      const unsigned full_y = static_cast<unsigned>(y + N);
      if (full.known_on.get(full_x, full_y)) {
        proj_on.set(x, y);
      }
      if (full.known_off.get(full_x, full_y)) {
        proj_off.set(x, y);
      }
    }
  }
  out.known_on = proj_on;
  out.known_off = proj_off;
  out.apply_bounds();
}

template <unsigned N>
_DI_ FullBitBoardT<N> expand_mask_test(BitBoard<32> mask) {
  FullBitBoardT<N> result;
  cuda::std::pair<int, int> cell;
  while (mask.pop_on_if_any(cell)) {
    ThreeBoardC4<N>::for_each_orbit_point(cell, [&](cuda::std::pair<int, int> pt) {
      const int full_x = pt.first + static_cast<int>(N);
      const int full_y = pt.second + static_cast<int>(N);
      result.set(full_x, full_y);
    });
  }
  return result;
}

template <unsigned N>
_DI_ BitBoard<32> project_mask_test(const FullBitBoardT<N> &mask) {
  BitBoard<32> result;
  for (int y = 0; y < static_cast<int>(N); ++y) {
    for (int x = 0; x < static_cast<int>(N); ++x) {
      bool set = false;
      ThreeBoardC4<N>::for_each_orbit_point({x, y}, [&](cuda::std::pair<int, int> pt) {
        if (set) {
          return;
        }
        const int full_x = pt.first + static_cast<int>(N);
        const int full_y = pt.second + static_cast<int>(N);
        if (mask.get(full_x, full_y)) {
          set = true;
        }
      });
      if (set) {
        result.set(x, y);
      }
    }
  }
  result &= ThreeBoardC4<N>::bounds();
  return result;
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

  FullBoardT<N> full_board = expand_to_full_test(c4_board);
  FullBoardT<N> forced_full = full_board.force_orthogonal();

  ThreeBoardC4<N> projected;
  project_from_full_test(forced_full, projected);
  projected.known_on.save(full_on_out);
  projected.known_off.save(full_off_out);
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

  FullBoardT<N> full_board = expand_to_full_test(board);
  auto full_vulnerable = full_board.vulnerable();
  BitBoard<32> projected = project_mask_test<N>(full_vulnerable);
  projected.save(full_vulnerable_out);
}

template <unsigned N>
void expect_force_matches(const std::string &known_on_rle,
                          const std::string &known_off_rle) {
  init_c4_tables_for_test<N>();

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

  FullBoardT<N> full = expand_to_full_test(base);
  FullBitBoardT<N> full_seeds = expand_mask_test<N>(seeds);
  full.eliminate_all_lines(full_seeds);

  ThreeBoardC4<N> projected;
  project_from_full_test(full, projected);
  projected.known_on.save(full_on_out);
  projected.known_off.save(full_off_out);
  const bool full_ok = full.consistent();
  if ((threadIdx.x & 31) == 0) {
    *full_consistency = full_ok;
  }
}

template <unsigned N>
void expect_eliminate_superset(const std::string &known_on_rle,
                               const std::string &known_off_rle,
                               const std::string &seed_rle) {
  init_c4_tables_for_test<N>();

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

  const std::string full_on_rle_out = to_rle<N, 32>(full_on);
  const std::string c4_on_rle_out = to_rle<N, 32>(c4_on);
  const std::string full_off_rle_out = to_rle<N, 32>(full_off);
  const std::string c4_off_rle_out = to_rle<N, 32>(c4_off);

  EXPECT_EQ(full_on_rle_out, c4_on_rle_out);

  bool off_superset_ok = true;
  for (unsigned y = 0; y < N; ++y) {
    if ((full_off[y] & ~c4_off[y]) != 0u) {
      off_superset_ok = false;
      break;
    }
  }
  EXPECT_TRUE(off_superset_ok) << "full_off=" << full_off_rle_out << " c4_off=" << c4_off_rle_out;

  if (!full_consistent) {
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
  init_c4_tables_for_test<N>();

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
__global__ void expand_project_kernel(board_row_t<32> *known_on,
                                      board_row_t<32> *known_off,
                                      board_row_t<32> *projected_on,
                                      board_row_t<32> *projected_off) {
  ThreeBoardC4<N> base;
  base.known_on = BitBoard<32>::load(known_on);
  base.known_off = BitBoard<32>::load(known_off);

  FullBoardT<N> full = expand_to_full_test(base);

  ThreeBoardC4<N> projected;
  project_from_full_test(full, projected);
  projected.known_on.save(projected_on);
  projected.known_off.save(projected_off);
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

TEST(ThreeBoardC4, EliminateSupersetOfFullBoard_Diagonal) {
  expect_eliminate_superset<4>("o3b$bo2b$4b$4b!", "!", "o3b$bo2b$4b$4b!");
}

TEST(ThreeBoardC4, EliminateSupersetOfFullBoard_Offset) {
  expect_eliminate_superset<6>("o5b$6b$bo4b$6b$6b$6b!", "!", "o5b$6b$bo4b$6b$6b$6b!");
}

TEST(ThreeBoardC4, EliminateSupersetOfFullBoard_SelfOrbit) {
  expect_eliminate_superset<6>(rle_from_points<6>({{1, 1}}), empty_rle<6>(), rle_from_points<6>({{1, 1}}));
}

TEST(ThreeBoardC4, EliminateSupersetOfFullBoard_TwoSeeds) {
  expect_eliminate_superset<6>(rle_from_points<6>({{1, 1}, {4, 3}}), empty_rle<6>(),
                               rle_from_points<6>({{1, 1}, {4, 3}}));
}

TEST(ThreeBoardC4, EliminateSupersetOfFullBoard_WithKnownOff) {
  expect_eliminate_superset<6>(rle_from_points<6>({{1, 1}}), rle_from_points<6>({{0, 0}, {2, 4}}),
                               rle_from_points<6>({{3, 2}}));
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
