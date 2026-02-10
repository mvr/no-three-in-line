#pragma once

#include "common.hpp"
#include "params.hpp"

#include "board.cu"
#include "compare_with_unknowns.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

__device__ uint32_t *g_line_table_32 = nullptr;

struct ForcedCell {
  bool has_force = false;
  bool force_on = false;
  cuda::std::pair<int, int> cell{0, 0};
};

template <unsigned N, unsigned W>
struct ThreeBoard {
  static constexpr unsigned LINE_ROWS = LINE_TABLE_FULL_WARP_LOAD ? 32 : ((N + 7u) & ~7u);
  static_assert(W != 32 || LINE_ROWS <= 32, "ThreeBoard line table rows must fit one warp");

  BitBoard<W> known_on;
  BitBoard<W> known_off;

  _DI_ ThreeBoard() : known_on{}, known_off{} {}
  _DI_ explicit ThreeBoard(BitBoard<W> known_on, BitBoard<W> known_off) : known_on{known_on}, known_off{known_off} {}

  _DI_ bool operator==(ThreeBoard<N, W> other) const { return (known_on == other.known_on) && (known_off == other.known_off); }
  _DI_ bool operator!=(ThreeBoard<N, W> other) const { return !(*this == other); }

  static _DI_ BitBoard<W> bounds();
  static _DI_ BitBoard<W> relevant_endpoint(cuda::std::pair<unsigned, unsigned> p);
  static void init_line_table_host();
  static void init_tables_host();

  _DI_ bool consistent() const;
  _DI_ bool complete() const;
  _DI_ unsigned unknown_pop() const;
  _DI_ LexStatus is_canonical_orientation() const;
  _DI_ LexStatus is_canonical_orientation_with_forced(ForcedCell &forced) const;
  _DI_ LexStatus canonical_with_forced(ForcedCell &forced) const;
  static _DI_ ThreeBoard<N, W> load_from(const board_array_t<W> &on,
                                         const board_array_t<W> &off);

  _DI_ ThreeBoard<N, W> force_orthogonal_horiz() const;
  _DI_ ThreeBoard<N, W> force_orthogonal_vert() const;
  _DI_ ThreeBoard<N, W> force_orthogonal() const { return force_orthogonal_horiz().force_orthogonal_vert(); }

  _DI_ BitBoard<W> vulnerable() const;
  _DI_ BitBoard<W> semivulnerable() const;
  _DI_ BitBoard<W> quasivulnerable() const;
  template <unsigned UnknownTarget>
  _DI_ BitBoard<W> semivulnerable_like() const;

  _DI_ BitBoard<W> eliminate_line_inner(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q, cuda::std::pair<unsigned, unsigned> delta);
  _DI_ BitBoard<W> eliminate_line(cuda::std::pair<unsigned, unsigned> p, cuda::std::pair<unsigned, unsigned> q);
  _DI_ void eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_all_lines(BitBoard<W> seed);
  _DI_ void eliminate_all_lines() { eliminate_all_lines(known_on); }

  _DI_ void eliminate_one_hop(cuda::std::pair<unsigned, unsigned> p);
  _DI_ void eliminate_one_hop(BitBoard<W> seed);

  _DI_ void propagate();

  _DI_ cuda::std::pair<unsigned, unsigned> most_constrained_row() const;
};

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::bounds() {
  return BitBoard<W>::rect(N, N);
}

template <unsigned N, unsigned W>
inline void ThreeBoard<N, W>::init_line_table_host() {
  if constexpr (W == 32) {
    static uint32_t *d_line_table = nullptr;
    const unsigned cell_count = N * N;
    const unsigned rows = LINE_ROWS;
    const size_t total_entries = static_cast<size_t>(cell_count) * cell_count;
    const size_t total_rows = total_entries * rows;

    std::vector<uint32_t> host_table(total_rows, 0u);

    for (unsigned py = 0; py < N; ++py) {
      for (unsigned px = 0; px < N; ++px) {
        const unsigned p_idx = py * N + px;
        for (unsigned qy = 0; qy < N; ++qy) {
          for (unsigned qx = 0; qx < N; ++qx) {
            const unsigned q_idx = qy * N + qx;
            if (p_idx == q_idx) {
              continue;
            }

            unsigned pyy = py;
            unsigned pxx = px;
            unsigned qyy = qy;
            unsigned qxx = qx;
            if (pyy > qyy) {
              std::swap(pyy, qyy);
              std::swap(pxx, qxx);
            }

            const int dx = static_cast<int>(qxx) - static_cast<int>(pxx);
            const unsigned dy = qyy - pyy;
            if (dy == 0) {
              continue;
            }

            const unsigned adx = static_cast<unsigned>(std::abs(dx));
            const unsigned g = std::gcd(adx, dy);
            const unsigned delta_y = dy / g;
            const int delta_x = (dx < 0 ? -1 : 1) * static_cast<int>(adx / g);

            const unsigned p_quo = pyy / delta_y;
            const unsigned p_rem = pyy % delta_y;

            uint32_t *mask = &host_table[(static_cast<size_t>(p_idx) * cell_count + q_idx) * rows];
            for (unsigned r = 0; r < N; ++r) {
              if (r % delta_y != p_rem) {
                continue;
              }
              const int col =
                  static_cast<int>(pxx) + (static_cast<int>(r / delta_y) - static_cast<int>(p_quo)) * delta_x;
              if (col >= 0 && col < static_cast<int>(N)) {
                mask[r] |= (uint32_t(1) << col);
              }
            }

            mask[py] &= ~(uint32_t(1) << px);
            mask[qy] &= ~(uint32_t(1) << qx);
          }
        }
      }
    }

    if (d_line_table != nullptr) {
      cudaFree(d_line_table);
      d_line_table = nullptr;
    }
    cudaMalloc((void **)&d_line_table, total_rows * sizeof(uint32_t));
    cudaMemcpy(d_line_table, host_table.data(), total_rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_line_table_32, &d_line_table, sizeof(d_line_table));
  }
}

template <unsigned N, unsigned W>
inline void ThreeBoard<N, W>::init_tables_host() {
  if constexpr (W == 32) {
    init_lookup_tables_host();
    init_relevant_endpoint_host(N);
    init_relevant_endpoint_host_64(N);
    init_line_table_host();
  } else {
    init_lookup_tables_host();
    init_relevant_endpoint_host_64(N);
  }
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::relevant_endpoint(cuda::std::pair<unsigned, unsigned> p) {
  if constexpr (W == 32) {
    uint64_t fullrow = relevant_endpoint_table[32-p.second+(threadIdx.x & 31)];
    uint32_t moved_row = fullrow >> (32-p.first); // And truncated
    return BitBoard<W>(moved_row);
  } else {
    BitBoard<W> result;

    // For row threadIdx.x * 2
    {
      unsigned row_idx = (64 - p.second + ((threadIdx.x & 31) * 2));
      uint64_t full_low_bits = relevant_endpoint_table_64[row_idx * 2];
      uint64_t full_high_bits = relevant_endpoint_table_64[row_idx * 2 + 1];
      if(p.first < 32) {
        // Origin ends up in state.x
        result.state.x = (full_low_bits >> (64 - p.first)) | (full_high_bits << p.first);
        result.state.y = full_high_bits >> (32 - p.first);
      } else {
        // Origin ends up in state.y
        result.state.x = full_low_bits >> (64 - p.first);
        result.state.y = (full_low_bits >> (64 - (p.first - 32))) | (full_high_bits << (p.first - 32));
      }
    }

    // For row threadIdx.x * 2 + 1
    {
      unsigned row_idx = 64 - p.second + ((threadIdx.x & 31) * 2 + 1);
      uint64_t full_low_bits = relevant_endpoint_table_64[row_idx * 2];
      uint64_t full_high_bits = relevant_endpoint_table_64[row_idx * 2 + 1];
      if(p.first < 32) {
        // Origin ends up in state.z
        result.state.z = (full_low_bits >> (64 - p.first)) | (full_high_bits << p.first);
        result.state.w = full_high_bits >> (32 - p.first);
      } else {
        // Origin ends up in state.w
        result.state.z = full_low_bits >> (64 - p.first);
        result.state.w = (full_low_bits >> (64 - (p.first - 32))) | (full_high_bits << (p.first - 32));
      }
    }

    return result;
  }
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoard<N, W>::consistent() const {
  return (known_on & known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ bool ThreeBoard<N, W>::complete() const {
  return (bounds() & ~known_on & ~known_off).empty();
}

template <unsigned N, unsigned W>
_DI_ unsigned ThreeBoard<N, W>::unknown_pop() const {
  return N*N - (known_on | known_off).pop();
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoard<N, W>::is_canonical_orientation() const {
  bool any_unknown = false;
  LexStatus order;

  const BitBoard<W> bounds = ThreeBoard<N, W>::bounds();
  BitBoard<W> flip_h_on = known_on.flip_horizontal().rotate_torus(N, 0);
  BitBoard<W> flip_h_off = known_off.flip_horizontal().rotate_torus(N, 0);
  order = compare_with_unknowns<W>(known_on, known_off, flip_h_on, flip_h_off, bounds);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> flip_v_on = known_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> flip_v_off = known_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<W>(known_on, known_off, flip_v_on, flip_v_off, bounds);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> rot180_on = flip_h_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> rot180_off = flip_h_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<W>(known_on, known_off, rot180_on, rot180_off, bounds);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> diag_on = known_on.flip_diagonal();
  BitBoard<W> diag_off = known_off.flip_diagonal();
  order = compare_with_unknowns<W>(known_on, known_off, diag_on, diag_off, bounds);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> rot90_on = diag_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> rot90_off = diag_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<W>(known_on, known_off, rot90_on, rot90_off, bounds);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> rot270_on = diag_on.flip_horizontal().rotate_torus(N, 0);
  BitBoard<W> rot270_off = diag_off.flip_horizontal().rotate_torus(N, 0);
  order = compare_with_unknowns<W>(known_on, known_off, rot270_on, rot270_off, bounds);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  BitBoard<W> anti_diag_on = rot270_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> anti_diag_off = rot270_off.flip_vertical().rotate_torus(0, N);
  order = compare_with_unknowns<W>(known_on, known_off, anti_diag_on, anti_diag_off, bounds);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) any_unknown = true;

  if (any_unknown)
    return LexStatus::Unknown;

  return LexStatus::Less;
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoard<N, W>::is_canonical_orientation_with_forced(ForcedCell &forced) const {
  bool any_unknown = false;
  bool has_force = false;
  LexStatus order;
  ForceCandidate local_force;

  const BitBoard<W> bounds = ThreeBoard<N, W>::bounds();
  constexpr int kSize = (W == 32) ? 32 : 64;
  constexpr int kMask = kSize - 1;
  auto wrap = [&](int v) { return v & kMask; };

  auto rot_inv = [&](cuda::std::pair<int, int> cell, int rh, int rv) {
    cell.first = wrap(cell.first - rh);
    cell.second = wrap(cell.second - rv);
    return cell;
  };
  auto flip_h = [&](cuda::std::pair<int, int> cell) {
    cell.first = kMask - cell.first;
    return cell;
  };
  auto flip_v = [&](cuda::std::pair<int, int> cell) {
    cell.second = kMask - cell.second;
    return cell;
  };
  auto flip_d = [&](cuda::std::pair<int, int> cell) {
    return cuda::std::pair<int, int>{cell.second, cell.first};
  };

  auto maybe_set_force = [&](const ForceCandidate &cand, auto inv_map) {
    if (!has_force && cand.has_force) {
      auto cell = cand.cell;
      if (cand.on_b) {
        cell = inv_map(cell);
      }
      forced.has_force = true;
      forced.force_on = cand.force_on;
      forced.cell = cell;
      has_force = true;
    }
  };

  BitBoard<W> flip_h_on = known_on.flip_horizontal().rotate_torus(N, 0);
  BitBoard<W> flip_h_off = known_off.flip_horizontal().rotate_torus(N, 0);
  local_force = ForceCandidate{};
  order = compare_with_unknowns_forced<W>(known_on, known_off, flip_h_on, flip_h_off, bounds, local_force);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) {
    any_unknown = true;
    auto inv_map = [&](cuda::std::pair<int, int> cell) {
      cell = rot_inv(cell, static_cast<int>(N), 0);
      cell = flip_h(cell);
      return cell;
    };
    maybe_set_force(local_force, inv_map);
  }

  BitBoard<W> flip_v_on = known_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> flip_v_off = known_off.flip_vertical().rotate_torus(0, N);
  local_force = ForceCandidate{};
  order = compare_with_unknowns_forced<W>(known_on, known_off, flip_v_on, flip_v_off, bounds, local_force);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) {
    any_unknown = true;
    auto inv_map = [&](cuda::std::pair<int, int> cell) {
      cell = rot_inv(cell, 0, static_cast<int>(N));
      cell = flip_v(cell);
      return cell;
    };
    maybe_set_force(local_force, inv_map);
  }

  BitBoard<W> rot180_on = flip_h_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> rot180_off = flip_h_off.flip_vertical().rotate_torus(0, N);
  local_force = ForceCandidate{};
  order = compare_with_unknowns_forced<W>(known_on, known_off, rot180_on, rot180_off, bounds, local_force);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) {
    any_unknown = true;
    auto inv_map = [&](cuda::std::pair<int, int> cell) {
      cell = rot_inv(cell, 0, static_cast<int>(N));
      cell = flip_v(cell);
      cell = rot_inv(cell, static_cast<int>(N), 0);
      cell = flip_h(cell);
      return cell;
    };
    maybe_set_force(local_force, inv_map);
  }

  BitBoard<W> diag_on = known_on.flip_diagonal();
  BitBoard<W> diag_off = known_off.flip_diagonal();
  local_force = ForceCandidate{};
  order = compare_with_unknowns_forced<W>(known_on, known_off, diag_on, diag_off, bounds, local_force);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) {
    any_unknown = true;
    auto inv_map = [&](cuda::std::pair<int, int> cell) {
      return flip_d(cell);
    };
    maybe_set_force(local_force, inv_map);
  }

  BitBoard<W> rot90_on = diag_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> rot90_off = diag_off.flip_vertical().rotate_torus(0, N);
  local_force = ForceCandidate{};
  order = compare_with_unknowns_forced<W>(known_on, known_off, rot90_on, rot90_off, bounds, local_force);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) {
    any_unknown = true;
    auto inv_map = [&](cuda::std::pair<int, int> cell) {
      cell = rot_inv(cell, 0, static_cast<int>(N));
      cell = flip_v(cell);
      cell = flip_d(cell);
      return cell;
    };
    maybe_set_force(local_force, inv_map);
  }

  BitBoard<W> rot270_on = diag_on.flip_horizontal().rotate_torus(N, 0);
  BitBoard<W> rot270_off = diag_off.flip_horizontal().rotate_torus(N, 0);
  local_force = ForceCandidate{};
  order = compare_with_unknowns_forced<W>(known_on, known_off, rot270_on, rot270_off, bounds, local_force);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) {
    any_unknown = true;
    auto inv_map = [&](cuda::std::pair<int, int> cell) {
      cell = rot_inv(cell, static_cast<int>(N), 0);
      cell = flip_h(cell);
      cell = flip_d(cell);
      return cell;
    };
    maybe_set_force(local_force, inv_map);
  }

  BitBoard<W> anti_diag_on = rot270_on.flip_vertical().rotate_torus(0, N);
  BitBoard<W> anti_diag_off = rot270_off.flip_vertical().rotate_torus(0, N);
  local_force = ForceCandidate{};
  order = compare_with_unknowns_forced<W>(known_on, known_off, anti_diag_on, anti_diag_off, bounds, local_force);
  if (order == LexStatus::Greater) return LexStatus::Greater;
  if (order == LexStatus::Unknown) {
    any_unknown = true;
    auto inv_map = [&](cuda::std::pair<int, int> cell) {
      cell = rot_inv(cell, 0, static_cast<int>(N));
      cell = flip_v(cell);
      cell = rot_inv(cell, static_cast<int>(N), 0);
      cell = flip_h(cell);
      cell = flip_d(cell);
      return cell;
    };
    maybe_set_force(local_force, inv_map);
  }

  if (any_unknown) {
    return LexStatus::Unknown;
  }

  return LexStatus::Less;
}

template <unsigned N, unsigned W>
_DI_ LexStatus ThreeBoard<N, W>::canonical_with_forced(ForcedCell &forced) const {
  return is_canonical_orientation_with_forced(forced);
}

template <unsigned N, unsigned W>
_DI_ ThreeBoard<N, W> ThreeBoard<N, W>::load_from(const board_array_t<W> &on,
                                                  const board_array_t<W> &off) {
  ThreeBoard<N, W> board;
  board.known_on = BitBoard<W>::load(on.data());
  board.known_off = BitBoard<W>::load(off.data());
  return board;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoard<N, W> ThreeBoard<N, W>::force_orthogonal_horiz() const {
  ThreeBoard<N, W> result = *this;

  if constexpr (W == 32) {
    int on_pop = popcount<32>(known_on.state);
    if(on_pop == 2) {
      result.known_off.state = ~known_on.state;
    }
    if(on_pop > 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int off_pop = popcount<32>(known_off.state);
    if(off_pop == N - 2) {
      result.known_on.state = ~known_off.state;
    }
    if(off_pop > N - 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }
  } else {
    int on_pop_x = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
    if(on_pop_x == 2) {
      result.known_off.state.x = ~known_on.state.x;
      result.known_off.state.y = ~known_on.state.y;
    }
    if(on_pop_x > 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int on_pop_z = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
    if(on_pop_z == 2) {
      result.known_off.state.z = ~known_on.state.z;
      result.known_off.state.w = ~known_on.state.w;
    }
    if(on_pop_z > 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int off_pop_x = popcount<32>(known_off.state.x) + popcount<32>(known_off.state.y);
    if(off_pop_x == N - 2) {
      result.known_on.state.x = ~known_off.state.x;
      result.known_on.state.y = ~known_off.state.y;
    }
    if(off_pop_x > N - 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }

    int off_pop_z = popcount<32>(known_off.state.z) + popcount<32>(known_off.state.w);
    if(off_pop_z == N - 2) {
      result.known_on.state.z = ~known_off.state.z;
      result.known_on.state.w = ~known_off.state.w;
    }
    if(off_pop_z > N - 2) {
      result.known_on = BitBoard<W>::solid();
      result.known_off = BitBoard<W>::solid();
    }
  }

  const BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  result.known_on &= bds;
  result.known_off &= bds;

  return result;
}

template <unsigned N, unsigned W>
_DI_ ThreeBoard<N, W> ThreeBoard<N, W>::force_orthogonal_vert() const {
  ThreeBoard<N, W> result = *this;

  if constexpr (W == 32) {
    const BinaryCountSaturating on_count = BinaryCountSaturating<32>::vertical(known_on.state);
    const uint32_t on_count_eq_2 = on_count.bit1 & ~on_count.bit0;
    result.known_off.state |= ~known_on.state & on_count_eq_2;

    const uint32_t on_count_gt_2 = on_count.bit1 & on_count.bit0;
    result.known_on.state |= on_count_gt_2;
    result.known_off.state |= on_count_gt_2;

    BitBoard<W> notKnownOff = ~known_off & ThreeBoard<N, W>::bounds();

    const BinaryCountSaturating not_off_count = BinaryCountSaturating<32>::vertical(notKnownOff.state);
    const uint32_t not_off_count_eq_2 = not_off_count.bit1 & ~not_off_count.bit0;
    result.known_on.state |= ~known_off.state & not_off_count_eq_2;

    const uint32_t not_off_count_lt_2 = ~not_off_count.bit1;
    result.known_on.state |= not_off_count_lt_2;
    result.known_off.state |= not_off_count_lt_2;
  } else {
    const BinaryCountSaturating on_count_xz = BinaryCountSaturating<32>::vertical(known_on.state.x) + BinaryCountSaturating<32>::vertical(known_on.state.z);
    const uint32_t on_count_xz_eq_2 = on_count_xz.bit1 & ~on_count_xz.bit0;
    result.known_off.state.x |= ~known_on.state.x & on_count_xz_eq_2;
    result.known_off.state.z |= ~known_on.state.z & on_count_xz_eq_2;

    const uint32_t on_count_xz_gt_2 = on_count_xz.bit1 & on_count_xz.bit0;
    result.known_on.state.x |= on_count_xz_gt_2;
    result.known_off.state.x |= on_count_xz_gt_2;

    const BinaryCountSaturating on_count_yw = BinaryCountSaturating<32>::vertical(known_on.state.y) + BinaryCountSaturating<32>::vertical(known_on.state.w);
    const uint32_t on_count_yw_eq_2 = on_count_yw.bit1 & ~on_count_yw.bit0;
    result.known_off.state.y |= ~known_on.state.y & on_count_yw_eq_2;
    result.known_off.state.w |= ~known_on.state.w & on_count_yw_eq_2;

    const uint32_t on_count_yw_gt_2 = on_count_yw.bit1 & on_count_yw.bit0;
    result.known_on.state.y |= on_count_yw_gt_2;
    result.known_off.state.y |= on_count_yw_gt_2;

    BitBoard<W> notKnownOff = ~known_off & ThreeBoard<N, W>::bounds();

    const BinaryCountSaturating not_off_count_xz = BinaryCountSaturating<32>::vertical(notKnownOff.state.x) + BinaryCountSaturating<32>::vertical(notKnownOff.state.z);
    const uint32_t not_off_count_xz_eq_2 = not_off_count_xz.bit1 & ~not_off_count_xz.bit0;
    result.known_on.state.x |= ~known_off.state.x & not_off_count_xz_eq_2;
    result.known_on.state.z |= ~known_off.state.z & not_off_count_xz_eq_2;

    const uint32_t not_off_count_xz_lt_2 = ~not_off_count_xz.bit1;
    result.known_on.state.x |= not_off_count_xz_lt_2;
    result.known_off.state.x |= not_off_count_xz_lt_2;

    const BinaryCountSaturating not_off_count_yw = BinaryCountSaturating<32>::vertical(notKnownOff.state.y) + BinaryCountSaturating<32>::vertical(notKnownOff.state.w);
    const uint32_t not_off_count_yw_eq_2 = not_off_count_yw.bit1 & ~not_off_count_yw.bit0;
    result.known_on.state.y |= ~known_off.state.y & not_off_count_yw_eq_2;
    result.known_on.state.w |= ~known_off.state.w & not_off_count_yw_eq_2;

    const uint32_t not_off_count_yw_lt_2 = ~not_off_count_yw.bit1;
    result.known_on.state.y |= not_off_count_yw_lt_2;
    result.known_off.state.y |= not_off_count_yw_lt_2;
  }

  const BitBoard<W> bds = ThreeBoard<N, W>::bounds();
  result.known_on &= bds;
  result.known_off &= bds;

  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::vulnerable() const {
  BitBoard<W> result;

  if constexpr (W == 32) {
    // Vulnerable horizontally
    {
      unsigned on_pop = popcount<32>(known_on.state);
      unsigned off_pop = popcount<32>(known_off.state);
      unsigned unknown_pop = N - on_pop - off_pop;
      bool vulnerable_row =
        (on_pop == 1 && unknown_pop == 2) || (on_pop == 0 && unknown_pop == 3);

      if (vulnerable_row)
        result.state = ~(board_row_t<W>)0;
    }

    // Vulnerable vertically
    {
      const BinaryCount<32> on_count = BinaryCount<32>::vertical(known_on.state);
      BitBoard<32> unknown = ~known_on & ~known_off & ThreeBoard<N, W>::bounds();
      const BinaryCount<32> unknown_count = BinaryCount<32>::vertical(unknown.state);

      uint32_t vulnerable_column =
          (on_count.bit0 & ~on_count.bit1 & ~on_count.overflow &
           ~unknown_count.bit0 & unknown_count.bit1 & ~unknown_count.overflow)
        | (~on_count.bit0 & ~on_count.bit1 & ~on_count.overflow &
           unknown_count.bit0 & unknown_count.bit1 & ~unknown_count.overflow);

      result.state |= vulnerable_column;
    }
  } else {
    // Vulnerable horizontally
    {
      unsigned on_pop_xy = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
      unsigned off_pop_xy = popcount<32>(known_off.state.x) + popcount<32>(known_off.state.y);
      unsigned unknown_pop_xy = N - on_pop_xy - off_pop_xy;
      bool vulnerable_row_xy =
        (on_pop_xy == 1 && unknown_pop_xy == 2) || (on_pop_xy == 0 && unknown_pop_xy == 3);

      if (vulnerable_row_xy) {
        result.state.x = ~(board_row_t<W>)0;
        result.state.y = ~(board_row_t<W>)0;
      }

      unsigned on_pop_zw = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
      unsigned off_pop_zw = popcount<32>(known_off.state.z) + popcount<32>(known_off.state.w);
      unsigned unknown_pop_zw = N - on_pop_zw - off_pop_zw;
      bool vulnerable_row_zw =
        (on_pop_zw == 1 && unknown_pop_zw == 2) || (on_pop_zw == 0 && unknown_pop_zw == 3);

      if (vulnerable_row_zw) {
        result.state.z = ~(board_row_t<W>)0;
        result.state.w = ~(board_row_t<W>)0;
      }
    }

    // Vulnerable vertically
    {
      BitBoard<64> unknown = ~known_on & ~known_off & ThreeBoard<N, W>::bounds();

      const BinaryCount<32> on_count_xz = BinaryCount<32>::vertical(known_on.state.x) + BinaryCount<32>::vertical(known_on.state.z);
      const BinaryCount<32> unknown_count_xz = BinaryCount<32>::vertical(unknown.state.x) + BinaryCount<32>::vertical(unknown.state.z);

      uint32_t vulnerable_column_xz =
          (on_count_xz.bit0 & ~on_count_xz.bit1 & ~on_count_xz.overflow &
           ~unknown_count_xz.bit0 & unknown_count_xz.bit1 & ~unknown_count_xz.overflow)
        | (~on_count_xz.bit0 & ~on_count_xz.bit1 & ~on_count_xz.overflow &
           unknown_count_xz.bit0 & unknown_count_xz.bit1 & ~unknown_count_xz.overflow);

      result.state.x |= vulnerable_column_xz;
      result.state.z |= vulnerable_column_xz;

      const BinaryCount<32> on_count_yw = BinaryCount<32>::vertical(known_on.state.y) + BinaryCount<32>::vertical(known_on.state.w);
      const BinaryCount<32> unknown_count_yw = BinaryCount<32>::vertical(unknown.state.y) + BinaryCount<32>::vertical(unknown.state.w);

      uint32_t vulnerable_column_yw =
          (on_count_yw.bit0 & ~on_count_yw.bit1 & ~on_count_yw.overflow &
           ~unknown_count_yw.bit0 & unknown_count_yw.bit1 & ~unknown_count_yw.overflow)
        | (~on_count_yw.bit0 & ~on_count_yw.bit1 & ~on_count_yw.overflow &
           unknown_count_yw.bit0 & unknown_count_yw.bit1 & ~unknown_count_yw.overflow);

      result.state.y |= vulnerable_column_yw;
      result.state.w |= vulnerable_column_yw;
    }
  }

  result &= ~known_on & ~known_off & ThreeBoard<N, W>::bounds();

  return result;
}

template <unsigned N, unsigned W>
template <unsigned UnknownTarget>
_DI_ BitBoard<W> ThreeBoard<N, W>::semivulnerable_like() const {
  static_assert(UnknownTarget < 8, "semivulnerable_like expects a target < 8");
  BitBoard<W> result;
  const BitBoard<W> bounds = ThreeBoard<N, W>::bounds();

  if constexpr (W == 32) {
    unsigned off_pop = popcount<32>(known_off.state);
    unsigned unknown_pop = N - off_pop;
    if (known_on.state == 0 && unknown_pop == UnknownTarget) {
      result.state = ~(board_row_t<W>)0;
    }

    BitBoard<32> unknown = ~known_on & ~known_off & bounds;
    const BinaryCountSaturating3<32> on_count = BinaryCountSaturating3<32>::vertical(known_on.state);
    const BinaryCountSaturating3<32> unknown_count = BinaryCountSaturating3<32>::vertical(unknown.state);
    const uint32_t on_zero = on_count.template eq_target<0>();
    const uint32_t unknown_eq = unknown_count.template eq_target<UnknownTarget>();
    result.state |= on_zero & unknown_eq;
  } else {
    unsigned off_pop_xy = popcount<32>(known_off.state.x) + popcount<32>(known_off.state.y);
    unsigned unknown_pop_xy = N - off_pop_xy;
    if ((known_on.state.x | known_on.state.y) == 0 && unknown_pop_xy == UnknownTarget) {
      result.state.x = ~(board_row_t<W>)0;
      result.state.y = ~(board_row_t<W>)0;
    }

    unsigned off_pop_zw = popcount<32>(known_off.state.z) + popcount<32>(known_off.state.w);
    unsigned unknown_pop_zw = N - off_pop_zw;
    if ((known_on.state.z | known_on.state.w) == 0 && unknown_pop_zw == UnknownTarget) {
      result.state.z = ~(board_row_t<W>)0;
      result.state.w = ~(board_row_t<W>)0;
    }

    BitBoard<64> unknown = ~known_on & ~known_off & bounds;

    const BinaryCountSaturating3<32> on_count_xz =
        BinaryCountSaturating3<32>::vertical(known_on.state.x) + BinaryCountSaturating3<32>::vertical(known_on.state.z);
    const BinaryCountSaturating3<32> unknown_count_xz =
        BinaryCountSaturating3<32>::vertical(unknown.state.x) + BinaryCountSaturating3<32>::vertical(unknown.state.z);
    const uint32_t on_zero_xz = on_count_xz.template eq_target<0>();
    const uint32_t unknown_eq_xz = unknown_count_xz.template eq_target<UnknownTarget>();
    const uint32_t column_xz = on_zero_xz & unknown_eq_xz;
    result.state.x |= column_xz;
    result.state.z |= column_xz;

    const BinaryCountSaturating3<32> on_count_yw =
        BinaryCountSaturating3<32>::vertical(known_on.state.y) + BinaryCountSaturating3<32>::vertical(known_on.state.w);
    const BinaryCountSaturating3<32> unknown_count_yw =
        BinaryCountSaturating3<32>::vertical(unknown.state.y) + BinaryCountSaturating3<32>::vertical(unknown.state.w);
    const uint32_t on_zero_yw = on_count_yw.template eq_target<0>();
    const uint32_t unknown_eq_yw = unknown_count_yw.template eq_target<UnknownTarget>();
    const uint32_t column_yw = on_zero_yw & unknown_eq_yw;
    result.state.y |= column_yw;
    result.state.w |= column_yw;
  }

  result &= ~known_on & ~known_off & bounds;
  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::semivulnerable() const {
  return semivulnerable_like<4>();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W> ThreeBoard<N, W>::quasivulnerable() const {
  return semivulnerable_like<5>();
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W>
ThreeBoard<N, W>::eliminate_line_inner(cuda::std::pair<unsigned, unsigned> p,
                                       cuda::std::pair<unsigned, unsigned> q,
                                       cuda::std::pair<unsigned, unsigned> delta) {
  BitBoard<W> result;

  unsigned p_quo = p.second / delta.second;
  unsigned p_rem = p.second % delta.second;

  if constexpr (W == 32) {
    unsigned row = threadIdx.x & 31;
    if (row % delta.second == p_rem) {
      int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
      if(col >= 0 && col < 32) result.state |= 1 << col;
    }
    if (p.second == row || q.second == row) {
      result.state = 0;
    }
  } else {
    {
      unsigned row = 2 * (threadIdx.x & 31);
      if (row % delta.second == p_rem) {
        int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
        if(col >= 0 && col < 32) result.state.x |= 1 << col;
        else if(col >= 32 && col < 64) result.state.y |= 1 << (col-32);
      }
      if (p.second == row || q.second == row) {
        result.state.x = 0;
        result.state.y = 0;
      }
    }

    {
      unsigned row = 2 * (threadIdx.x & 31) + 1;
      if (row % delta.second == p_rem) {
        int col = p.first + ((int)(row / delta.second) - p_quo) * delta.first;
        if(col >= 0 && col < 32) result.state.z |= 1 << col;
        else if(col >= 32 && col < 64) result.state.w |= 1 << (col-32);
      }
      if (p.second == row || q.second == row) {
        result.state.z = 0;
        result.state.w = 0;
      }
    }
  }

  return result;
}

template <unsigned N, unsigned W>
_DI_ BitBoard<W>
ThreeBoard<N, W>::eliminate_line(cuda::std::pair<unsigned, unsigned> p,
                                 cuda::std::pair<unsigned, unsigned> q) {
  if constexpr (W == 32) {
    constexpr unsigned cell_count = N * N;
    unsigned p_idx = p.second * N + p.first;
    unsigned q_idx = q.second * N + q.first;
    const uint32_t *entry = g_line_table_32 + (static_cast<size_t>(p_idx) * cell_count + q_idx) * LINE_ROWS;
    const unsigned lane = threadIdx.x & 31;
    const uint32_t row = __ldg(entry + lane);
    return BitBoard<32>(row);
  } else {
    if (p.second > q.second)
      cuda::std::swap(p, q);

    cuda::std::pair<int, unsigned> delta = {(int)q.first - p.first, q.second - p.second};

    // Recall div_gcd_table[x][y] = x / gcd(x, y)
    const unsigned first_div = div_gcd_table[std::abs(delta.first)][delta.second];
    const unsigned second_div = div_gcd_table[delta.second][std::abs(delta.first)];
    delta.first = (delta.first < 0 ? -1 : 1) * first_div;
    delta.second = second_div;

    switch(delta.second) {
    case 1: return eliminate_line_inner(p, q, {delta.first, 1});
    case 2: return eliminate_line_inner(p, q, {delta.first, 2});
    case 4: return eliminate_line_inner(p, q, {delta.first, 4});
    default: return eliminate_line_inner(p, q, delta);
    }
  }
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> qs = known_on & ThreeBoard<N, W>::relevant_endpoint(p);
  if constexpr (W == 32) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned p_idx = p.second * N + p.first;
    const uint32_t *base = g_line_table_32 + (static_cast<size_t>(p_idx) * N * N) * LINE_ROWS;

    cuda::std::pair<int, int> q;
    while (qs.some_on_if_any(q)) {
      qs.erase(q);
      const unsigned q_idx = static_cast<unsigned>(q.second) * N + static_cast<unsigned>(q.first);
      const uint32_t row = __ldg(base + q_idx * LINE_ROWS + lane);
      known_off |= BitBoard<32>(row);
      if (__any_sync(0xffffffff, row & known_on.state)) {
        return;
      }
    }
  } else {
    cuda::std::pair<int, int> q;
    while (qs.some_on_if_any(q)) {
      known_off |= eliminate_line(p, q);
      if (!consistent())
        return;
      qs.erase(q);
    }
    known_off &= bounds();
  }
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_all_lines(BitBoard<W> ps) {
  cuda::std::pair<int, int> p;
  while (ps.some_on_if_any(p)) {
    BitBoard<W> qs = known_on & ~ps & ThreeBoard<N, W>::relevant_endpoint(p);

    if constexpr (W == 32) {
      const unsigned lane = threadIdx.x & 31;
      const unsigned p_idx = static_cast<unsigned>(p.second) * N + static_cast<unsigned>(p.first);
      const uint32_t *base = g_line_table_32 + (static_cast<size_t>(p_idx) * N * N) * LINE_ROWS;

      cuda::std::pair<int, int> q;
      while (qs.some_on_if_any(q)) {
        qs.erase(q);
        const unsigned q_idx = static_cast<unsigned>(q.second) * N + static_cast<unsigned>(q.first);
        const uint32_t row = __ldg(base + q_idx * LINE_ROWS + lane);
        known_off |= BitBoard<32>(row);
        if (__any_sync(0xffffffff, row & known_on.state)) {
          return;
        }
      }
    } else {
      cuda::std::pair<int, int> q;
      while (qs.some_on_if_any(q)) {
        known_off |= eliminate_line(p, q);
        if (!consistent())
          return;
        qs.erase(q);
      }
    }
    ps.erase(p);
    known_off &= bounds();
  }
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_one_hop(cuda::std::pair<unsigned, unsigned> p) {
  BitBoard<W> to_eliminate = known_on.mirror_around(p);
  to_eliminate.erase_row(p.second);
  known_off |= to_eliminate;
  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void
ThreeBoard<N, W>::eliminate_one_hop(BitBoard<W> ps) {
  cuda::std::pair<int, int> p;
  while (ps.some_on_if_any(p)) {
    BitBoard<W> to_eliminate = known_on.mirror_around(p);
    to_eliminate.erase_row(p.second);
    known_off |= to_eliminate;
    ps.erase(p);
  }

  known_off &= bounds();
}

template <unsigned N, unsigned W>
_DI_ void ThreeBoard<N, W>::propagate() {
  ThreeBoard<N, W> prev;

  BitBoard<W> done_ons = known_on;

  do {
    do {
      prev = *this;
      *this = force_orthogonal();

      if(!consistent())
        return;

    } while(*this != prev);

    prev = *this;

    eliminate_all_lines(known_on & ~done_ons);

    if(!consistent())
        return;

    done_ons = known_on;
  } while (*this != prev);
}

_DI_ unsigned row_unknown_score(unsigned on_pop, unsigned unknown) {
  constexpr unsigned kMaxUnknown = 8;
  constexpr unsigned kOn1Penalty = 6;
  constexpr unsigned kTable[2][kMaxUnknown + 1] = {
    // known_on = 0 (default: choose-2 unknown pairs)
    {0, 0, 1, 3, 6, 10, 15, 21, 28},
    // known_on = 1
    {0, 5, 9, 14, 20, 40, 48, 56, 64},
  };

  if (on_pop <= 1 && unknown <= kMaxUnknown) {
    return kTable[on_pop][unknown];
  }

  if (on_pop == 0) {
    return unknown * (unknown - 1) / 2;
  }
  if (on_pop == 1) {
    return unknown * kOn1Penalty;
  }
  return unknown;
}

template <unsigned N, unsigned W>
_DI_ cuda::std::pair<unsigned, unsigned>
ThreeBoard<N, W>::most_constrained_row() const {
  unsigned row;
  unsigned unknown;

  if constexpr (W == 32) {
    BitBoard<W> known = known_on | known_off;
    unknown = N - popcount<32>(known.state);
    unsigned on_pop = popcount<32>(known_on.state);
    unknown = row_unknown_score(on_pop, unknown);

    if ((threadIdx.x & 31) >= N || unknown == 0)
      unknown = std::numeric_limits<unsigned>::max();

    row = (threadIdx.x & 31);
  } else {
    BitBoard<W> known = known_on | known_off;
    unsigned unknown_xy = N - popcount<32>(known.state.x) - popcount<32>(known.state.y);
    unsigned unknown_zw = N - popcount<32>(known.state.z) - popcount<32>(known.state.w);

    unsigned on_pop_xy = popcount<32>(known_on.state.x) + popcount<32>(known_on.state.y);
    unsigned on_pop_zw = popcount<32>(known_on.state.z) + popcount<32>(known_on.state.w);
    unknown_xy = row_unknown_score(on_pop_xy, unknown_xy);
    unknown_zw = row_unknown_score(on_pop_zw, unknown_zw);

    if ((threadIdx.x & 31) * 2 >= N || unknown_xy == 0)
      unknown_xy = std::numeric_limits<unsigned>::max();
    if ((threadIdx.x & 31) * 2 + 1 >= N || unknown_zw == 0)
      unknown_zw = std::numeric_limits<unsigned>::max();

    if (unknown_xy < unknown_zw) {
      row = (threadIdx.x & 31) * 2;
      unknown = unknown_xy;
    } else {
      row = (threadIdx.x & 31) * 2 + 1;
      unknown = unknown_zw;
    }
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    unsigned other_row = __shfl_down_sync(0xffffffff, row, offset);
    unsigned other_unknown = __shfl_down_sync(0xffffffff, unknown, offset);

    if (other_unknown < unknown) {
      row = other_row;
      unknown = other_unknown;
    }
  }

  row = __shfl_sync(0xffffffff, row, 0);
  unknown = __shfl_sync(0xffffffff, unknown, 0);

  return {row, unknown};
}
