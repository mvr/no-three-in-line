#include <array>

#include "board.cu"

template <unsigned N>
__global__ void soft_branch_kernel(uint64_t *knownOn, uint64_t *knownOff) {
  ThreeBoard<N> board;
  board.knownOn = BitBoard::load(knownOn);
  board.knownOff = BitBoard::load(knownOff);
  board.eliminate_all_lines();
  board.propagate();
  board.soft_branch_all();
  board.knownOn.save(knownOn);
  board.knownOff.save(knownOff);
}

template <unsigned N>
std::pair<std::array<uint64_t, 64>, std::array<uint64_t, 64>>
soft_branch(const std::array<uint64_t, 64> &inputKnownOn,
            const std::array<uint64_t, 64> &inputKnownOff) {
  uint64_t *d_knownOn;
  uint64_t *d_knownOff;

  cudaMalloc((void**) &d_knownOn, 64 * sizeof(uint64_t));
  cudaMemcpy(d_knownOn, inputKnownOn.data(), 64 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_knownOff, 64 * sizeof(uint64_t));
  cudaMemcpy(d_knownOff, inputKnownOff.data(), 64 * sizeof(uint64_t), cudaMemcpyHostToDevice);

  soft_branch_kernel<N><<<1, 32>>>(d_knownOn, d_knownOff);

  std::array<uint64_t, 64> resultOn, resultOff;
  cudaMemcpy(resultOn.data(), d_knownOn, 64*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(resultOff.data(), d_knownOff, 64*sizeof(uint64_t), cudaMemcpyDeviceToHost);

  cudaFree(d_knownOn);
  cudaFree(d_knownOff);

  return {resultOn, resultOff};
}

// Explicitly instantiate the template, or it doesn't get compiled at all.
template
std::pair<std::array<uint64_t, 64>, std::array<uint64_t, 64>>
soft_branch<10>(const std::array<uint64_t, 64> &inputKnownOn,
                const std::array<uint64_t, 64> &inputKnownOff);
