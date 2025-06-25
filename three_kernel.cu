#include <array>
#include <vector>
#include <numeric>

#include "board.cu"
#include "three_board.cu"

#include "three_kernel.hpp"

#include "params.hpp"

template <unsigned W>
DeviceMemory<W>::DeviceMemory(unsigned batch_size) : max_batch_size(batch_size) {
  cudaMalloc((void**) &d_problems, max_batch_size * sizeof(Problem<W>));
  cudaMalloc((void**) &d_outcomes, max_batch_size * sizeof(Outcome<W>));
}

template <unsigned W>
DeviceMemory<W>::~DeviceMemory() {
  cudaFree(d_problems);
  cudaFree(d_outcomes);
}

template <unsigned N, unsigned W>
__global__ void work_kernel(Problem<W> *problems, Outcome<W> *outcomes) {
  Problem<W> &problem = problems[blockIdx.x];
  BitBoard<W> seed = BitBoard<W>::load(problem.seed.data());

  ThreeBoard<N, W> board;
  board.knownOn = BitBoard<W>::load(problem.knownOn.data()) | seed;
  board.knownOff = BitBoard<W>::load(problem.knownOff.data());

  board.eliminate_all_lines(seed);
  board.propagate();
  board.soft_branch_all();

  Outcome<W> &outcome = outcomes[blockIdx.x];
  board.knownOn.save(outcome.knownOn.data());
  board.knownOff.save(outcome.knownOff.data());

  bool consistent = board.consistent();

  if(threadIdx.x == 0) {
    outcome.consistent = consistent;
  }

  if(consistent) {
    unsigned unknown_pop = board.unknown_pop();
    auto [row, _] = board.most_constrained_row();

    if(threadIdx.x == 0) {
      outcome.unknownPop = unknown_pop;
      outcome.solved = outcome.unknownPop == 0;
      outcome.axis = Axis::Horizontal;
      outcome.ix = row;
    }
  }
}


template <unsigned N, unsigned W>
std::vector<Outcome<W>>
launch_work_kernel(unsigned batch_size,
                   std::vector<Problem<W>> problems,
                   DeviceMemory<W> &device_mem) {
  cudaMemcpy(device_mem.d_problems, problems.data(), batch_size * sizeof(Problem<W>), cudaMemcpyHostToDevice);

  work_kernel<N, W><<<batch_size, 32>>>(device_mem.d_problems, device_mem.d_outcomes);

  device_mem.outcomes_buffer.resize(batch_size);
  cudaMemcpy(device_mem.outcomes_buffer.data(), device_mem.d_outcomes, batch_size * sizeof(Outcome<W>), cudaMemcpyDeviceToHost);

  return std::move(device_mem.outcomes_buffer);
}

// Explicitly instantiate the template to the N in params.hpp, or it doesn't get compiled at all.
template DeviceMemory<32>::DeviceMemory(unsigned batch_size);
template DeviceMemory<32>::~DeviceMemory();
template std::vector<Outcome<32>>
launch_work_kernel<N, 32>(unsigned batch_size, std::vector<Problem<32>> problems, DeviceMemory<32> &device_mem);

// template std::vector<Outcome<64>>
// launch_work_kernel<N, 64>(unsigned batch_size, std::vector<Problem<64>> problems);

void init_lookup_tables_host() {
  unsigned char host_div_gcd_table[64][64];

  for (unsigned i = 1; i < 64; i++) {
    for (unsigned j = 1; j < 64; j++) {
      host_div_gcd_table[i][j] = i / std::gcd(i, j);
    }
  }

  cudaMemcpyToSymbol(div_gcd_table, host_div_gcd_table, sizeof(host_div_gcd_table));
}

bool relevant_endpoint(std::pair<unsigned, unsigned> q) {
  if (q.first == 0 || q.second == 0)
    return false;

  unsigned factor = std::gcd(q.first, q.second);

  if (factor > 1)
    // There is a point between that needs checking
    return true;

  if(q.first * 2 >= N || q.second*2 >= N)
    // There is no way a third point can fit in the square
    return false;

  return true;
}

// TODO: implement W==64 version
void init_relevant_endpoint_host() {
  uint64_t host_relevant_endpoint_table[64] = {0};

  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < N; j++) {
     bool relevant = relevant_endpoint({i, j});
     if(relevant) {
       host_relevant_endpoint_table[32+j] |= (1ULL << (32+i));
       host_relevant_endpoint_table[32+j] |= (1ULL << (32-i));
       host_relevant_endpoint_table[32-j] |= (1ULL << (32+i));
       host_relevant_endpoint_table[32-j] |= (1ULL << (32-i));
     }
    }
  }

  cudaMemcpyToSymbol(relevant_endpoint_table, host_relevant_endpoint_table, sizeof(host_relevant_endpoint_table));
}
