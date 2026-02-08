#pragma once

#ifdef __CUDACC__

template <unsigned W>
struct BinaryCount {
  board_row_t<W> bit0;
  board_row_t<W> bit1;
  board_row_t<W> overflow;

  _DI_ BinaryCount operator+(const BinaryCount other) const {
    const board_row_t<W> out0 = bit0 ^ other.bit0;
    const board_row_t<W> carry0 = bit0 & other.bit0;

    const board_row_t<W> out1_temp = bit1 ^ other.bit1;
    const board_row_t<W> out1 = out1_temp ^ carry0;
    const board_row_t<W> carry1 = (bit1 & other.bit1) | (carry0 & out1_temp);
    const board_row_t<W> out_overflow = carry1 | overflow | other.overflow;

    return {out0, out1, out_overflow};
  }

  _DI_ void operator+=(const BinaryCount other) { *this = *this + other; }

  static _DI_ BinaryCount vertical(const board_row_t<W> value) {
    BinaryCount result = {value, 0, 0};

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      BinaryCount other;
      other.bit0 = __shfl_xor_sync(0xffffffff, result.bit0, offset);
      other.bit1 = __shfl_xor_sync(0xffffffff, result.bit1, offset);
      other.overflow = __shfl_xor_sync(0xffffffff, result.overflow, offset);
      result += other;
    }

    return result;
  }

  static _DI_ BinaryCount horizontal(const board_row_t<W> value) {
    static_assert(W == 32, "BinaryCount::horizontal currently supports W=32 only");
    BinaryCount result{};

    const unsigned pop = popcount<W>(value);
    const bool bit0 = (pop & 1u) != 0u;
    const bool bit1 = (pop & 2u) != 0u;
    const bool overflow = pop >= 4u;

    const unsigned mask = __activemask();
    result.bit0 = __ballot_sync(mask, bit0);
    result.bit1 = __ballot_sync(mask, bit1);
    result.overflow = __ballot_sync(mask, overflow);
    return result;
  }
};

template <unsigned W>
struct BinaryCountSaturating {
  board_row_t<W> bit0;
  board_row_t<W> bit1;

  _DI_ BinaryCountSaturating operator+(const BinaryCountSaturating other) const {
    const board_row_t<W> new0 = maj3(bit1, other.bit1, other.bit0) | (bit0 ^ other.bit0);
    const board_row_t<W> new1 = bit1 | other.bit1 | (bit0 & other.bit0);
    return {new0, new1};
  }

  _DI_ void operator+=(const BinaryCountSaturating other) { *this = *this + other; }

  template <unsigned Target>
  _DI_ board_row_t<W> eq_target() const {
    static_assert(Target < 4, "eq_target supports targets 0-3");
    board_row_t<W> mask = ~board_row_t<W>(0);
    if constexpr (Target & 1) {
      mask &= bit0;
    } else {
      mask &= ~bit0;
    }
    if constexpr (Target & 2) {
      mask &= bit1;
    } else {
      mask &= ~bit1;
    }
    return mask;
  }

  static _DI_ BinaryCountSaturating vertical(const board_row_t<W> value) {
    BinaryCountSaturating result = {value, 0};

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      BinaryCountSaturating other;
      other.bit0 = __shfl_xor_sync(0xffffffff, result.bit0, offset);
      other.bit1 = __shfl_xor_sync(0xffffffff, result.bit1, offset);
      result += other;
    }

    return result;
  }

  static _DI_ BinaryCountSaturating horizontal(const board_row_t<W> value) {
    static_assert(W == 32, "BinaryCountSaturating::horizontal currently supports W=32 only");
    BinaryCountSaturating result{};
    unsigned pop = popcount<W>(value);
    result.bit0 = __ballot_sync(0xffffffff, pop == 1 || pop > 2);
    result.bit1 = __ballot_sync(0xffffffff, pop >= 2);
    return result;
  }

  static _DI_ BinaryCountSaturating horizontal_interleave(const board_row_t<W> even_value,
                                                          const board_row_t<W> odd_value) {
    static_assert(W == 64, "BinaryCountSaturating::horizontal_interleave currently supports W=64 only");
    BinaryCountSaturating result{};

    const unsigned even_pop = popcount<64>(even_value);
    const unsigned odd_pop = popcount<64>(odd_value);

    const uint32_t bit0_even = __ballot_sync(0xffffffff, even_pop == 1 || even_pop > 2);
    const uint32_t bit0_odd = __ballot_sync(0xffffffff, odd_pop == 1 || odd_pop > 2);
    const uint32_t bit1_even = __ballot_sync(0xffffffff, even_pop >= 2);
    const uint32_t bit1_odd = __ballot_sync(0xffffffff, odd_pop >= 2);

    result.bit0 = interleave32(bit0_even, bit0_odd);
    result.bit1 = interleave32(bit1_even, bit1_odd);
    return result;
  }
};

template <unsigned W>
struct BinaryCountSaturating3 {
  board_row_t<W> bit0;
  board_row_t<W> bit1;
  board_row_t<W> bit2;

  _DI_ BinaryCountSaturating3 operator+(const BinaryCountSaturating3 other) const {
    const board_row_t<W> sum0 = bit0 ^ other.bit0;
    const board_row_t<W> carry0 = bit0 & other.bit0;

    if constexpr (W == 32) {
      const board_row_t<W> sum1 = xor3(bit1, other.bit1, carry0);
      const board_row_t<W> carry1 = maj3(bit1, other.bit1, carry0);

      const board_row_t<W> sum2 = xor3(bit2, other.bit2, carry1);
      const board_row_t<W> carry2 = maj3(bit2, other.bit2, carry1);

      const board_row_t<W> overflow = carry2;
      return {sum0 | overflow, sum1 | overflow, sum2 | overflow};
    } else {
      const board_row_t<W> sum1 = bit1 ^ other.bit1 ^ carry0;
      const board_row_t<W> carry1 = (bit1 & other.bit1) | (carry0 & (bit1 ^ other.bit1));

      const board_row_t<W> sum2 = bit2 ^ other.bit2 ^ carry1;
      const board_row_t<W> carry2 = (bit2 & other.bit2) | (carry1 & (bit2 ^ other.bit2));

      const board_row_t<W> overflow = carry2;
      return {sum0 | overflow, sum1 | overflow, sum2 | overflow};
    }
  }

  _DI_ void operator+=(const BinaryCountSaturating3 other) { *this = *this + other; }

  template <unsigned Target>
  _DI_ board_row_t<W> eq_target() const {
    static_assert(Target < 8, "eq_target supports targets 0-7");
    board_row_t<W> mask = ~board_row_t<W>(0);
    if constexpr (Target & 1) {
      mask &= bit0;
    } else {
      mask &= ~bit0;
    }
    if constexpr (Target & 2) {
      mask &= bit1;
    } else {
      mask &= ~bit1;
    }
    if constexpr (Target & 4) {
      mask &= bit2;
    } else {
      mask &= ~bit2;
    }
    return mask;
  }

  static _DI_ BinaryCountSaturating3 vertical(const board_row_t<W> value) {
    BinaryCountSaturating3 result = {value, 0, 0};

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      BinaryCountSaturating3 other;
      other.bit0 = __shfl_xor_sync(0xffffffff, result.bit0, offset);
      other.bit1 = __shfl_xor_sync(0xffffffff, result.bit1, offset);
      other.bit2 = __shfl_xor_sync(0xffffffff, result.bit2, offset);
      result += other;
    }

    return result;
  }

  static _DI_ BinaryCountSaturating3 horizontal(const board_row_t<W> value) {
    static_assert(W == 32, "BinaryCountSaturating3::horizontal currently supports W=32 only");
    BinaryCountSaturating3 result{};

    unsigned pop = popcount<W>(value);
    bool bit0 = (pop & 1u) != 0u;
    bool bit1 = (pop & 2u) != 0u;
    bool bit2 = (pop & 4u) != 0u;
    if (pop >= 7u) {
      bit0 = true;
      bit1 = true;
      bit2 = true;
    }

    const unsigned mask = __activemask();
    result.bit0 = __ballot_sync(mask, bit0);
    result.bit1 = __ballot_sync(mask, bit1);
    result.bit2 = __ballot_sync(mask, bit2);
    return result;
  }

  static _DI_ BinaryCountSaturating3 horizontal_interleave(const board_row_t<W> even_value,
                                                           const board_row_t<W> odd_value) {
    static_assert(W == 64, "BinaryCountSaturating3::horizontal_interleave currently supports W=64 only");
    BinaryCountSaturating3 result{};

    unsigned even_pop = popcount<64>(even_value);
    unsigned odd_pop = popcount<64>(odd_value);

    bool even_bit0 = (even_pop & 1u) != 0u;
    bool even_bit1 = (even_pop & 2u) != 0u;
    bool even_bit2 = (even_pop & 4u) != 0u;
    if (even_pop >= 7u) {
      even_bit0 = true;
      even_bit1 = true;
      even_bit2 = true;
    }

    bool odd_bit0 = (odd_pop & 1u) != 0u;
    bool odd_bit1 = (odd_pop & 2u) != 0u;
    bool odd_bit2 = (odd_pop & 4u) != 0u;
    if (odd_pop >= 7u) {
      odd_bit0 = true;
      odd_bit1 = true;
      odd_bit2 = true;
    }

    const uint32_t bit0_even = __ballot_sync(0xffffffff, even_bit0);
    const uint32_t bit0_odd = __ballot_sync(0xffffffff, odd_bit0);
    const uint32_t bit1_even = __ballot_sync(0xffffffff, even_bit1);
    const uint32_t bit1_odd = __ballot_sync(0xffffffff, odd_bit1);
    const uint32_t bit2_even = __ballot_sync(0xffffffff, even_bit2);
    const uint32_t bit2_odd = __ballot_sync(0xffffffff, odd_bit2);

    result.bit0 = interleave32(bit0_even, bit0_odd);
    result.bit1 = interleave32(bit1_even, bit1_odd);
    result.bit2 = interleave32(bit2_even, bit2_odd);
    return result;
  }
};

#endif
