#include <cuda/std/utility>
#include <array>

#include "gtest/gtest.h"

#include "parsing.hpp"
#include "board.cu"

enum class SymmetryTransformType {
  ROTATE_90,
  ROTATE_180,
  ROTATE_270,
  FLIP_HORIZONTAL,
  FLIP_VERTICAL,
  FLIP_DIAGONAL,
  FLIP_ANTI_DIAGONAL
};

template <unsigned W>
__global__ void symmetry_transform_kernel(board_row_t<W> *input_board_data, board_row_t<W> *output_board_data, SymmetryTransformType transform_type) {
  BitBoard<W> input_board = BitBoard<W>::load(input_board_data);
  BitBoard<W> result_board;

  switch (transform_type) {
  case SymmetryTransformType::ROTATE_90:
    result_board = input_board.rotate_90();
    break;
  case SymmetryTransformType::ROTATE_180:
    result_board = input_board.rotate_180();
    break;
  case SymmetryTransformType::ROTATE_270:
    result_board = input_board.rotate_270();
    break;
  case SymmetryTransformType::FLIP_HORIZONTAL:
    result_board = input_board.flip_horizontal();
    break;
  case SymmetryTransformType::FLIP_VERTICAL:
    result_board = input_board.flip_vertical();
    break;
  case SymmetryTransformType::FLIP_DIAGONAL:
    result_board = input_board.flip_diagonal();
    break;
  case SymmetryTransformType::FLIP_ANTI_DIAGONAL:
    result_board = input_board.flip_anti_diagonal();
    break;
  }
  result_board.save(output_board_data);
}

template <unsigned W>
void test_symmetry_transform(const std::string &input_rle, const std::string &expected_rle, SymmetryTransformType transform_type) {
  board_row_t<W> *d_input, *d_output;
  board_array_t<W> h_output;

  cudaMalloc((void**) &d_input, W * sizeof(board_row_t<W>));
  cudaMemcpy(d_input, parse_rle<W>(input_rle).data(), W * sizeof(board_row_t<W>), cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_output, W * sizeof(board_row_t<W>));

  symmetry_transform_kernel<W><<<1, W>>>(d_input, d_output, transform_type);
  cudaMemcpy(h_output.data(), d_output, W * sizeof(board_row_t<W>), cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected_rle, (to_rle<W, W>(h_output)));

  cudaFree(d_input);
  cudaFree(d_output);
}

TEST(BitBoardSymmetry, FlipHorizontal) {
  test_symmetry_transform<32>("3o$o$3o$o$o!", "29b3o$31bo$29b3o$31bo$31bo!",
                              SymmetryTransformType::FLIP_HORIZONTAL);
  test_symmetry_transform<64>("3o$o$3o$o$o!", "61b3o$63bo$61b3o$63bo$63bo!",
                              SymmetryTransformType::FLIP_HORIZONTAL);
}

TEST(BitBoardSymmetry, FlipVertical) {
  test_symmetry_transform<32>("3o$o$3o$o$o!", "27$o$o$3o$o$3o!",
                              SymmetryTransformType::FLIP_VERTICAL);
  test_symmetry_transform<64>("3o$o$3o$o$o!", "59$o$o$3o$o$3o!",
                              SymmetryTransformType::FLIP_VERTICAL);
}

TEST(BitBoardSymmetry, Rotate180) {
  test_symmetry_transform<32>("3o$o$3o$o$o!", "27$31bo$31bo$29b3o$31bo$29b3o!",
                              SymmetryTransformType::ROTATE_180);
  test_symmetry_transform<64>("3o$o$3o$o$o!", "59$63bo$63bo$61b3o$63bo$61b3o!",
                              SymmetryTransformType::ROTATE_180);
}

TEST(BitBoardSymmetry, FlipDiagonal) {
  test_symmetry_transform<32>("3o$o$3o$o$o!", "5o$obo$obo!",
                              SymmetryTransformType::FLIP_DIAGONAL);
  test_symmetry_transform<64>("3o$o$3o$o$o!", "5o$obo$obo!",
                              SymmetryTransformType::FLIP_DIAGONAL);
}

TEST(BitBoardSymmetry, Rotate90) {
  test_symmetry_transform<32>("3o$o$3o$o$o!", "29$obo$obo$5o!",
                              SymmetryTransformType::ROTATE_90);
  test_symmetry_transform<64>("3o$o$3o$o$o!", "61$obo$obo$5o!",
                              SymmetryTransformType::ROTATE_90);
}

TEST(BitBoardSymmetry, Rotate270) {
  test_symmetry_transform<32>("3o$o$3o$o$o!", "27b5o$29bobo$29bobo!",
                              SymmetryTransformType::ROTATE_270);
  test_symmetry_transform<64>("3o$o$3o$o$o!", "59b5o$61bobo$61bobo!",
                              SymmetryTransformType::ROTATE_270);
}

TEST(BitBoardSymmetry, FlipAntiDiagonal) {
  test_symmetry_transform<32>("3o$o$3o$o$o!", "29$29bobo$29bobo$27b5o!",
                              SymmetryTransformType::FLIP_ANTI_DIAGONAL);
  test_symmetry_transform<64>("3o$o$3o$o$o!", "61$61bobo$61bobo$59b5o!",
                              SymmetryTransformType::FLIP_ANTI_DIAGONAL);
}
