#include <cuda/std/utility>
#include <array>

#include "gtest/gtest.h"

#include "parsing.hpp"
#include "board.cu"

template <unsigned N>
__global__ void three_bounds_kernel(uint64_t *a) {
  BitBoard bds = ThreeBoard<N>::bounds();
  bds.save(a);
}

template <unsigned N>
void test_bounds(const std::array<uint64_t, 64> &expected) {
  uint64_t *d_a;
  std::array<uint64_t, 64> h_a;

  cudaMalloc((void**) &d_a, 64 * sizeof(uint64_t));

  three_bounds_kernel<N><<<1, 32>>>(d_a);
  cudaMemcpy(h_a.data(), d_a, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  EXPECT_EQ(to_rle(expected), to_rle(h_a));

  cudaFree(d_a);
}

TEST(ThreeBoard, Bounds) {
  test_bounds<2>(parse_rle("2o$2o!"));
  test_bounds<3>(parse_rle("3o$3o$3o!"));
  test_bounds<4>(parse_rle("4o$4o$4o$4o!"));
  test_bounds<5>(parse_rle("5o$5o$5o$5o$5o!"));

  test_bounds<31>(parse_rle("31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o$31o!"));
  test_bounds<32>(parse_rle("32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o$32o!"));
  test_bounds<33>(parse_rle("33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o$33o!"));
  test_bounds<34>(parse_rle("34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o$34o!"));
}

template <unsigned N>
__global__ void force_horiz_kernel(uint64_t *knownOn, uint64_t *knownOff) {
  ThreeBoard<N> board;
  board.knownOn = BitBoard::load(knownOn);
  board.knownOff = BitBoard::load(knownOff);
  board = board.force_orthogonal_horiz();
  board.knownOn.save(knownOn);
  board.knownOff.save(knownOff);
}

template <unsigned N>
__global__ void force_vert_kernel(uint64_t *knownOn, uint64_t *knownOff) {
  ThreeBoard<N> board;
  board.knownOn = BitBoard::load(knownOn);
  board.knownOff = BitBoard::load(knownOff);
  board = board.force_orthogonal_vert();
  board.knownOn.save(knownOn);
  board.knownOff.save(knownOff);
}

enum struct TestType {
  Hori,
  Vert,
};

template <unsigned N, TestType type>
void test_force(const std::array<uint64_t, 64> &inputKnownOn,
                const std::array<uint64_t, 64> &inputKnownOff,
                const std::array<uint64_t, 64> &expectedKnownOn,
                const std::array<uint64_t, 64> &expectedKnownOff) {
  uint64_t *d_knownOn;
  uint64_t *h_knownOn;

  cudaMalloc((void**) &d_knownOn, 64 * sizeof(uint64_t));
  cudaMallocHost((void**) &h_knownOn, 64 * sizeof(uint64_t));
  cudaMemcpy(d_knownOn, inputKnownOn.data(), 64 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  uint64_t *d_knownOff;
  uint64_t *h_knownOff;

  cudaMalloc((void**) &d_knownOff, 64 * sizeof(uint64_t));
  cudaMallocHost((void**) &h_knownOff, 64 * sizeof(uint64_t));
  cudaMemcpy(d_knownOff, inputKnownOff.data(), 64 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  if constexpr (type == TestType::Hori) {
    force_horiz_kernel<N><<<1, 32>>>(d_knownOn, d_knownOff);
  } else if constexpr (type == TestType::Vert) {
    force_vert_kernel<N><<<1, 32>>>(d_knownOn, d_knownOff);
  }

  cudaMemcpy(h_knownOn, d_knownOn, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_knownOff, d_knownOff, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(expectedKnownOn[i], h_knownOn[i]) << "knownOn wrong on row " << i;
  }

  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(expectedKnownOff[i], h_knownOff[i]) << "knownOff wrong on row " << i;
  }

  cudaFree(d_knownOn);
  cudaFreeHost(h_knownOn);
  cudaFree(d_knownOff);
  cudaFreeHost(h_knownOff);
}

TEST(ThreeBoard, ForceHoriVert) {
  test_force<4, TestType::Hori>(parse_rle("2o!"), parse_rle("!"),
                                 parse_rle("2o!"), parse_rle("2b2o!"));
  test_force<5, TestType::Hori>(parse_rle("2o!"), parse_rle("!"),
                                 parse_rle("2o!"), parse_rle("2b3o!"));

  test_force<4, TestType::Hori>(parse_rle("!"), parse_rle("2o!"),
                                 parse_rle("2b2o!"), parse_rle("2o!"));

  test_force<4, TestType::Vert>(parse_rle("o$o!"), parse_rle("!"),
                                parse_rle("o$o!"), parse_rle("2$o$o!"));
  test_force<5, TestType::Vert>(parse_rle("o$o!"), parse_rle("!"),
                                parse_rle("o$o!"), parse_rle("2$o$o$o!"));

  test_force<4, TestType::Vert>(parse_rle("!"), parse_rle("o$o!"),
                                parse_rle("2$o$o!"), parse_rle("o$o!"));
}

template <unsigned N>
__global__ void line_kernel(uint64_t *a, cuda::std::pair<int, int> p, cuda::std::pair<int, int> q) {
  BitBoard b = BitBoard::line(p, q) & ThreeBoard<N>::bounds();
  b.save(a);
}

template <unsigned N>
void test_line(cuda::std::pair<int, int> p, cuda::std::pair<int, int> q, const std::array<uint64_t, 64> &expected) {
  uint64_t *d_a;
  std::array<uint64_t, 64> h_a;

  cudaMalloc((void**) &d_a, 64 * sizeof(uint64_t));

  line_kernel<N><<<1, 32>>>(d_a, p, q);
  cudaMemcpy(h_a.data(), d_a, 64 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  EXPECT_EQ(to_rle(expected), to_rle(h_a));

  cudaFree(d_a);
}

TEST(ThreeBoard, Line) {
  test_line<4>({0,0}, {1,1}, parse_rle("o$bo$2bo$3bo!"));
  test_line<4>({0,0}, {2,2}, parse_rle("o$bo$2bo$3bo!"));
  test_line<4>({0,0}, {3,3}, parse_rle("o$bo$2bo$3bo!"));
  test_line<4>({1,1}, {0,0}, parse_rle("o$bo$2bo$3bo!"));
  test_line<4>({1,1}, {2,2}, parse_rle("o$bo$2bo$3bo!"));
  test_line<4>({1,1}, {3,3}, parse_rle("o$bo$2bo$3bo!"));
  test_line<4>({2,2}, {3,3}, parse_rle("o$bo$2bo$3bo!"));

  test_line<5>({0,0}, {1,2}, parse_rle("o2$bo2$2bo!"));
  test_line<5>({0,0}, {2,1}, parse_rle("o$2bo$4bo!"));
  test_line<5>({0,0}, {2,4}, parse_rle("o2$bo2$2bo!"));
  test_line<5>({0,0}, {4,2}, parse_rle("o$2bo$4bo!"));
}

template <unsigned N>
__global__ void contradiction_kernel(uint64_t *a, bool *result) {
  ThreeBoard<N> board;
  board.knownOn = BitBoard::load(a);
  board.eliminate_all_lines();
  board.propagate();
  board.soft_branch_all();
  *result = board.contradiction();
}

template <unsigned N>
void test_consistent(const std::array<uint64_t, 64> &inputKnownOn, bool expectedConsistent) {
  uint64_t *d_a;

  cudaMalloc((void**) &d_a, 64 * sizeof(uint64_t));
  cudaMemcpy(d_a, inputKnownOn.data(), 64 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  bool *d_result;
  cudaMalloc((void**) &d_result, sizeof(bool));

  contradiction_kernel<N><<<1, 32>>>(d_a, d_result);
  bool contradiction;
  cudaMemcpy(&contradiction, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

  if(expectedConsistent) {
    EXPECT_FALSE(contradiction);
  } else {
    EXPECT_TRUE(contradiction);
  }

  cudaFree(d_a);
}

TEST(ThreeBoard, Consistent) {
  test_consistent<10>(parse_rle("o4bo$2bobo$5bobo$bo7bo$2b2o$7bobo$o7bo$bo4bo$3bo4bo$4bobo!"), true);
  test_consistent<10>(parse_rle("o4bo$3bo2bo$2bo5bo$2bobo$7b2o$6bo2bo$o8bo$bobo$bo5bo$4b2o!"), true);
}

TEST(ThreeBoard, Inconsistent) {
  test_consistent<10>(parse_rle("o4bo$2bobo$5bobo$2o7bo$2b2o!"), false);
}
