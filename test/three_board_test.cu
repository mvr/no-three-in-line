#include <cuda/std/utility>
#include <array>

#include "gtest/gtest.h"

#include "parsing.hpp"
#include "board.cu"
#include "three_board.cu"

// Common type aliases for W-dependent types
template<unsigned W>
using board_t = std::conditional_t<W == 64, std::array<uint64_t, 64>, std::array<uint32_t, 32>>;

template<unsigned W>
using row_t = std::conditional_t<W == 64, uint64_t, uint32_t>;

template <unsigned N, unsigned W>
__global__ void three_bounds_kernel(row_t<W> *a) {
  BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  bds.save(a);
}

template <unsigned N, unsigned W>
void test_bounds(const board_t<W> &expected) {
  row_t<W> *d_a;
  board_t<W> h_a;

  cudaMalloc((void**) &d_a, W * sizeof(row_t<W>));

  three_bounds_kernel<N, W><<<1, 32>>>(d_a);
  cudaMemcpy(h_a.data(), d_a, W * sizeof(row_t<W>), cudaMemcpyDeviceToHost);

  EXPECT_EQ((to_rle<N, W>(expected)), (to_rle<N, W>(h_a)));

  cudaFree(d_a);
}

template <unsigned N>
void test_bounds_both(const std::string &expected_rle) {
  test_bounds<N, 32>(parse_rle<32>(expected_rle));
  test_bounds<N, 64>(parse_rle<64>(expected_rle));
}

TEST(ThreeBoard, Bounds) {
  test_bounds_both<2>("2o$2o!");
  test_bounds_both<3>("3o$3o$3o!");
  test_bounds_both<4>("4o$4o$4o$4o!");
  test_bounds_both<5>("5o$5o$5o$5o$5o!");
  test_bounds_both<31>("31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o!");
  test_bounds_both<32>("32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o!");
  test_bounds<33, 64>(parse_rle<64>("33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o!"));
  test_bounds<34, 64>(parse_rle<64>("34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o!"));
}

template <unsigned N, unsigned W>
__global__ void force_horiz_kernel(row_t<W> *knownOn, row_t<W> *knownOff) {
  ThreeBoard<N, W> board;
  board.knownOn = BitBoard<W>::load(knownOn);
  board.knownOff = BitBoard<W>::load(knownOff);
  board = board.force_orthogonal_horiz();
  board.knownOn.save(knownOn);
  board.knownOff.save(knownOff);
}

template <unsigned N, unsigned W>
__global__ void force_vert_kernel(row_t<W> *knownOn, row_t<W> *knownOff) {
  ThreeBoard<N, W> board;
  board.knownOn = BitBoard<W>::load(knownOn);
  board.knownOff = BitBoard<W>::load(knownOff);
  board = board.force_orthogonal_vert();
  board.knownOn.save(knownOn);
  board.knownOff.save(knownOff);
}

template <unsigned N, unsigned W, Axis type>
void test_force(const board_t<W> &inputKnownOn, const board_t<W> &inputKnownOff,
                const board_t<W> &expectedKnownOn, const board_t<W> &expectedKnownOff) {
  row_t<W> *d_knownOn, *d_knownOff;
  board_t<W> h_knownOn, h_knownOff;

  cudaMalloc((void**) &d_knownOn, W * sizeof(row_t<W>));
  cudaMemcpy(d_knownOn, inputKnownOn.data(), W * sizeof(row_t<W>), cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_knownOff, W * sizeof(row_t<W>));
  cudaMemcpy(d_knownOff, inputKnownOff.data(), W * sizeof(row_t<W>), cudaMemcpyHostToDevice);

  if constexpr (type == Axis::Horizontal) {
    force_horiz_kernel<N, W><<<1, 64>>>(d_knownOn, d_knownOff);
  } else {
    force_vert_kernel<N, W><<<1, 32>>>(d_knownOn, d_knownOff);
  }

  cudaMemcpy(h_knownOn.data(), d_knownOn, W * sizeof(row_t<W>), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_knownOff.data(), d_knownOff, W * sizeof(row_t<W>), cudaMemcpyDeviceToHost);

  EXPECT_EQ((to_rle<N, W>(expectedKnownOn)), (to_rle<N, W>(h_knownOn)));
  EXPECT_EQ((to_rle<N, W>(expectedKnownOff)), (to_rle<N, W>(h_knownOff)));

  cudaFree(d_knownOn);
  cudaFree(d_knownOff);
}

template <unsigned N, Axis type>
void test_force_both(const std::string &inputKnownOnRle,
                     const std::string &inputKnownOffRle,
                     const std::string &expectedKnownOnRle,
                     const std::string &expectedKnownOffRle) {
  test_force<N, 32, type>(parse_rle<32>(inputKnownOnRle), parse_rle<32>(inputKnownOffRle), 
                          parse_rle<32>(expectedKnownOnRle), parse_rle<32>(expectedKnownOffRle));
  test_force<N, 64, type>(parse_rle<64>(inputKnownOnRle), parse_rle<64>(inputKnownOffRle), 
                          parse_rle<64>(expectedKnownOnRle), parse_rle<64>(expectedKnownOffRle));
}

TEST(ThreeBoard, ForceHoriVert) {
  test_force_both<4, Axis::Horizontal>("2o!", "!", "2o!", "2b2o!");
  test_force_both<5, Axis::Horizontal>("2o!", "!", "2o!", "2b3o!");

  test_force_both<4, Axis::Horizontal>("!", "2o!", "2b2o!", "2o!");

  test_force_both<4, Axis::Vertical>("o$o!", "!", "o$o!", "2$o$o!");
  test_force_both<5, Axis::Vertical>("o$o!", "!", "o$o!", "2$o$o$o!");

  test_force_both<4, Axis::Vertical>("!", "o$o!", "2$o$o!", "o$o!");
}


// template <unsigned N, unsigned W>
// __global__ void line_kernel(row_t<W> *a, cuda::std::pair<int, int> p, cuda::std::pair<int, int> q) {
//   BitBoard<W> b = ThreeBoard<N, W>::line({(unsigned)p.first, (unsigned)p.second}, {(unsigned)q.first, (unsigned)q.second}) & ThreeBoard<N, W>::bounds();
//   b.save(a);
// }

// template <unsigned N, unsigned W>
// void test_line(cuda::std::pair<int, int> p, cuda::std::pair<int, int> q, const board_t<W> &expected) {
//   row_t<W> *d_a;
//   board_t<W> h_a;

//   cudaMalloc((void**) &d_a, W * sizeof(row_t<W>));
//   line_kernel<N, W><<<1, 32>>>(d_a, p, q);
//   cudaMemcpy(h_a.data(), d_a, W * sizeof(row_t<W>), cudaMemcpyDeviceToHost);

//   EXPECT_EQ((to_rle<N, W>(expected)), (to_rle<N, W>(h_a)));
//   cudaFree(d_a);
// }

// template <unsigned N>
// void test_line_both(cuda::std::pair<int, int> p, cuda::std::pair<int, int> q, const std::string &expectedRle) {
//   test_line<N, 32>(p, q, parse_rle<32>(expectedRle));
//   test_line<N, 64>(p, q, parse_rle<64>(expectedRle));
// }

// TEST(ThreeBoard, Line) {
//   test_line_both<4>({0,0}, {1,1}, "o$bo$2bo$3bo!");
//   test_line_both<4>({0,0}, {2,2}, "o$bo$2bo$3bo!");
//   test_line_both<4>({0,0}, {3,3}, "o$bo$2bo$3bo!");
//   test_line_both<4>({1,1}, {0,0}, "o$bo$2bo$3bo!");
//   test_line_both<4>({1,1}, {2,2}, "o$bo$2bo$3bo!");
//   test_line_both<4>({1,1}, {3,3}, "o$bo$2bo$3bo!");
//   test_line_both<4>({2,2}, {3,3}, "o$bo$2bo$3bo!");

//   test_line_both<5>({0,0}, {1,2}, "o2$bo2$2bo!");
//   test_line_both<5>({0,0}, {2,1}, "o$2bo$4bo!");
//   test_line_both<5>({0,0}, {2,4}, "o2$bo2$2bo!");
//   test_line_both<5>({0,0}, {4,2}, "o$2bo$4bo!");
// }


template <unsigned N, unsigned W>
__global__ void consistent_kernel(row_t<W> *a, bool *result) {
  ThreeBoard<N, W> board;
  board.knownOn = BitBoard<W>::load(a);
  board.eliminate_all_lines();
  board.propagate();
  board.soft_branch_all();
  *result = board.consistent();
}

template <unsigned N, unsigned W>
void test_consistent(const board_t<W> &inputKnownOn, bool expectedConsistent) {
  row_t<W> *d_a;
  bool *d_result;

  cudaMalloc((void**) &d_a, W * sizeof(row_t<W>));
  cudaMemcpy(d_a, inputKnownOn.data(), W * sizeof(row_t<W>), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_result, sizeof(bool));

  consistent_kernel<N, W><<<1, 32>>>(d_a, d_result);
  bool consistent;
  cudaMemcpy(&consistent, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

  EXPECT_EQ(expectedConsistent, consistent);
  
  cudaFree(d_a);
  cudaFree(d_result);
}

template <unsigned N>
void test_consistent_both(const std::string &inputKnownOnRle, bool expectedConsistent) {
  test_consistent<N, 32>(parse_rle<32>(inputKnownOnRle), expectedConsistent);
  test_consistent<N, 64>(parse_rle<64>(inputKnownOnRle), expectedConsistent);
}

TEST(ThreeBoard, Consistent) {
  test_consistent_both<4>("b2o$o2bo$b2o$o2bo!", true);
  test_consistent_both<4>("obo$bobo$bobo$obo!", true);
  test_consistent_both<4>("o2bo$b2o$o2bo$b2o!", true);
  test_consistent_both<4>("bobo$obo$obo$bobo!", true);

  test_consistent_both<10>("o4bo$2bobo$5bobo$bo7bo$2b2o$7bobo$o7bo$bo4bo$3bo4bo$4bobo!", true);
  test_consistent_both<10>("o4bo$3bo2bo$2bo5bo$2bobo$7b2o$6bo2bo$o8bo$bobo$bo5bo$4b2o!", true);
}

TEST(ThreeBoard, Inconsistent) {
  test_consistent_both<10>("o4bo$2bobo$5bobo$2o7bo$2b2o!", false);
}
