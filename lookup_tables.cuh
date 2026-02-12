#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>

#include <cstdint>
#include <numeric>
#include <utility>

__constant__ unsigned char div_gcd_table[64][64];

__device__ uint64_t relevant_endpoint_table[64];
__device__ uint64_t relevant_endpoint_table_64[256];

inline void init_lookup_tables_host() {
  unsigned char host_div_gcd_table[64][64];

  for (unsigned i = 1; i < 64; i++) {
    for (unsigned j = 1; j < 64; j++) {
      host_div_gcd_table[i][j] = i / std::gcd(i, j);
    }
  }

  cudaMemcpyToSymbol(div_gcd_table, host_div_gcd_table, sizeof(host_div_gcd_table));
}

inline bool relevant_endpoint(unsigned n, std::pair<unsigned, unsigned> q) {
  if (q.first == 0 || q.second == 0)
    return false;

  unsigned factor = std::gcd(q.first, q.second);

  if (factor > 1)
    return true;

  if (q.first * 2 >= n || q.second * 2 >= n)
    return false;

  return true;
}

inline void init_relevant_endpoint_host(unsigned n) {
  uint64_t host_relevant_endpoint_table[64] = {0};

  const unsigned limit = n > 32 ? 32 : n;

  for (unsigned i = 0; i < limit; i++) {
    for (unsigned j = 0; j < limit; j++) {
      bool relevant = relevant_endpoint(n, {i, j});
      if (relevant) {
        host_relevant_endpoint_table[32 + j] |= 1ULL << (32 + i);
        host_relevant_endpoint_table[32 + j] |= 1ULL << (32 - i);
        host_relevant_endpoint_table[32 - j] |= 1ULL << (32 + i);
        host_relevant_endpoint_table[32 - j] |= 1ULL << (32 - i);
      }
    }
  }

  cudaMemcpyToSymbol(
      relevant_endpoint_table, host_relevant_endpoint_table, sizeof(host_relevant_endpoint_table));
}

// 128x128 grid from (-64, -64) to (63, 63)
// Layout A[0], A[1]
//        A[2], A[3]
//        A[4], A[5]
// Etc
// So (0, 0) is stored in A[129], awkwardly.
inline void init_relevant_endpoint_host_64(unsigned n) {
  uint64_t host_relevant_endpoint_table_64[256] = {0};

  const unsigned limit = n > 64 ? 64 : n;

  for (unsigned i = 1; i < limit; i++) {
    for (unsigned j = 1; j < limit; j++) {
      bool relevant = relevant_endpoint(n, {i, j});
      if (relevant) {
        host_relevant_endpoint_table_64[(64 + j) * 2 + 0] |= 1ULL << (64 - i);
        host_relevant_endpoint_table_64[(64 + j) * 2 + 1] |= 1ULL << i;
        host_relevant_endpoint_table_64[(64 - j) * 2 + 0] |= 1ULL << (64 - i);
        host_relevant_endpoint_table_64[(64 - j) * 2 + 1] |= 1ULL << i;
      }
    }
  }

  cudaMemcpyToSymbol(relevant_endpoint_table_64,
                     host_relevant_endpoint_table_64,
                     sizeof(host_relevant_endpoint_table_64));
}

#endif
