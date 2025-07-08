#include <cuda/std/utility>
#include <array>

#include "gtest/gtest.h"

#include "parsing.hpp"
#include "board.cu"
#include "three_board.cu"

template <unsigned N, unsigned W>
__global__ void three_bounds_kernel(board_row_t<W> *a) {
  BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  bds.save(a);
}

template <unsigned N, unsigned W>
void test_bounds(const board_array_t<W> &expected) {
  board_row_t<W> *d_a;
  board_array_t<W> h_a;

  cudaMalloc((void**) &d_a, sizeof(board_array_t<W>));

  three_bounds_kernel<N, W><<<1, 32>>>(d_a);
  cudaMemcpy(h_a.data(), d_a, sizeof(board_array_t<W>), cudaMemcpyDeviceToHost);

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
__global__ void force_horiz_kernel(board_row_t<W> *known_on, board_row_t<W> *known_off) {
  ThreeBoard<N, W> board;
  board.known_on = BitBoard<W>::load(known_on);
  board.known_off = BitBoard<W>::load(known_off);
  board = board.force_orthogonal_horiz();
  board.known_on.save(known_on);
  board.known_off.save(known_off);
}

template <unsigned N, unsigned W>
__global__ void force_vert_kernel(board_row_t<W> *known_on, board_row_t<W> *known_off) {
  ThreeBoard<N, W> board;
  board.known_on = BitBoard<W>::load(known_on);
  board.known_off = BitBoard<W>::load(known_off);
  board = board.force_orthogonal_vert();
  board.known_on.save(known_on);
  board.known_off.save(known_off);
}

template <unsigned N, unsigned W, Axis type>
void test_force(const board_array_t<W> &input_known_on, const board_array_t<W> &input_known_off,
                const board_array_t<W> &expected_known_on, const board_array_t<W> &expected_known_off) {
  board_row_t<W> *d_known_on, *d_known_off;
  board_array_t<W> h_known_on, h_known_off;

  cudaMalloc((void**) &d_known_on, sizeof(board_array_t<W>));
  cudaMemcpy(d_known_on, input_known_on.data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_known_off, sizeof(board_array_t<W>));
  cudaMemcpy(d_known_off, input_known_off.data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);

  if constexpr (type == Axis::Horizontal) {
    force_horiz_kernel<N, W><<<1, 64>>>(d_known_on, d_known_off);
  } else {
    force_vert_kernel<N, W><<<1, 32>>>(d_known_on, d_known_off);
  }

  cudaMemcpy(h_known_on.data(), d_known_on, sizeof(board_array_t<W>), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_known_off.data(), d_known_off, sizeof(board_array_t<W>), cudaMemcpyDeviceToHost);

  EXPECT_EQ((to_rle<N, W>(expected_known_on)), (to_rle<N, W>(h_known_on)));
  EXPECT_EQ((to_rle<N, W>(expected_known_off)), (to_rle<N, W>(h_known_off)));

  cudaFree(d_known_on);
  cudaFree(d_known_off);
}

template <unsigned N, Axis type>
void test_force_both(const std::string &input_known_on_rle,
                     const std::string &input_known_off_rle,
                     const std::string &expected_known_on_rle,
                     const std::string &expected_known_off_rle) {
  test_force<N, 32, type>(parse_rle<32>(input_known_on_rle), parse_rle<32>(input_known_off_rle),
                          parse_rle<32>(expected_known_on_rle), parse_rle<32>(expected_known_off_rle));
  test_force<N, 64, type>(parse_rle<64>(input_known_on_rle), parse_rle<64>(input_known_off_rle),
                          parse_rle<64>(expected_known_on_rle), parse_rle<64>(expected_known_off_rle));
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
// __global__ void line_kernel(board_row_t<W> *a, cuda::std::pair<int, int> p, cuda::std::pair<int, int> q) {
//   BitBoard<W> b = ThreeBoard<N, W>::line({(unsigned)p.first, (unsigned)p.second}, {(unsigned)q.first, (unsigned)q.second}) & ThreeBoard<N, W>::bounds();
//   b.save(a);
// }

// template <unsigned N, unsigned W>
// void test_line(cuda::std::pair<int, int> p, cuda::std::pair<int, int> q, const board_array_t<W> &expected) {
//   board_row_t<W> *d_a;
//   board_array_t<W> h_a;

//   cudaMalloc((void**) &d_a, sizeof(board_array_t<W>));
//   line_kernel<N, W><<<1, 32>>>(d_a, p, q);
//   cudaMemcpy(h_a.data(), d_a, sizeof(board_array_t<W>), cudaMemcpyDeviceToHost);

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
__global__ void consistent_kernel(board_row_t<W> *a, bool *result) {
  ThreeBoard<N, W> board;
  board.known_on = BitBoard<W>::load(a);
  board.eliminate_all_lines();
  board.propagate();
  board.soft_branch_all();
  *result = board.consistent();
}

template <unsigned N, unsigned W>
void test_consistent(const board_array_t<W> &input_known_on, bool expected_consistent) {
  board_row_t<W> *d_a;
  bool *d_result;

  cudaMalloc((void**) &d_a, sizeof(board_array_t<W>));
  cudaMemcpy(d_a, input_known_on.data(), sizeof(board_array_t<W>), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_result, sizeof(bool));

  consistent_kernel<N, W><<<1, 32>>>(d_a, d_result);
  bool consistent;
  cudaMemcpy(&consistent, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected_consistent, consistent);
  
  cudaFree(d_a);
  cudaFree(d_result);
}

template <unsigned N>
void test_consistent_both(const std::string &input_known_on_rle, bool expected_consistent) {
  test_consistent<N, 32>(parse_rle<32>(input_known_on_rle), expected_consistent);
  test_consistent<N, 64>(parse_rle<64>(input_known_on_rle), expected_consistent);
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
