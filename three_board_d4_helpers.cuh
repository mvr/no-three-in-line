#pragma once

#include "common.hpp"

template <unsigned H>
struct D4CompactTriangle32 {
  static _DI_ constexpr board_row_t<32> low_mask(unsigned bits) {
    if (bits == 0u) {
      return 0u;
    }
    if (bits >= 32u) {
      return 0xffffffffu;
    }
    return (board_row_t<32>(1u) << bits) - 1u;
  }

  static _DI_ constexpr unsigned anti_width(unsigned ly) {
    return (ly < H) ? (H - ly) : 0u;
  }

  static _DI_ constexpr board_row_t<32> main_triangle_mask(unsigned ly, unsigned store_h) {
    return (ly < store_h) ? low_mask(ly + 1u) : 0u;
  }

  static _DI_ constexpr board_row_t<32> anti_triangle_mask(board_row_t<32> row_mask,
                                                            unsigned ly,
                                                            unsigned store_h) {
    return row_mask & ~main_triangle_mask(ly, store_h);
  }

  // Convert compact anti-side bits to matrix row columns [ly, H-1].
  static _DI_ board_row_t<32> unpack_anti_compact_matrix(board_row_t<32> compact, unsigned ly) {
    const unsigned width = anti_width(ly);
    if (width == 0u) {
      return 0u;
    }
    const board_row_t<32> compact_mask = low_mask(width);
    const board_row_t<32> compact_bits = compact & compact_mask;
    const board_row_t<32> rev = __brev(compact_bits) >> (32u - width);
    return rev << ly;
  }

  static _DI_ board_row_t<32> pack_anti_compact_matrix(board_row_t<32> matrix_row, unsigned ly) {
    const unsigned width = anti_width(ly);
    if (width == 0u) {
      return 0u;
    }
    const board_row_t<32> compact_mask = low_mask(width);
    const board_row_t<32> seg = (matrix_row >> ly) & compact_mask;
    return __brev(seg) >> (32u - width);
  }

  // Align compact anti bits to row/column family index space with chosen offset.
  static _DI_ board_row_t<32> compact_to_family_aligned(board_row_t<32> compact,
                                                         unsigned ly,
                                                         unsigned family_offset) {
    const unsigned width = anti_width(ly);
    if (width == 0u) {
      return 0u;
    }
    const board_row_t<32> compact_mask = low_mask(width);
    const board_row_t<32> compact_bits = compact & compact_mask;
    const board_row_t<32> rev = __brev(compact_bits) >> (32u - width);
    return rev << (ly + family_offset);
  }

  static _DI_ board_row_t<32> reverse_low_bits(board_row_t<32> value) {
    if constexpr (H == 0) {
      return 0u;
    } else {
      return __brev(value) >> (32u - H);
    }
  }
};
