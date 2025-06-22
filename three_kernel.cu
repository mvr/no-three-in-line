#include <array>
#include <vector>

#include "board.cu"
#include "three_board.cu"

#include "three_kernel.hpp"

#include "params.hpp"

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

  unsigned unknown_pop = board.unknown_pop();
  bool consistent = board.consistent();
  auto [axis, row] = board.most_constrained();

  if(threadIdx.x == 0) {
    outcome.unknownPop = unknown_pop;
    outcome.solved = outcome.unknownPop == 0;
    outcome.consistent = consistent;
    outcome.axis = axis;
    outcome.ix = row;
  }
}

template <unsigned N, unsigned W>
std::vector<Outcome<W>>
launch_work_kernel(unsigned batch_size,
                   std::vector<Problem<W>> problems) {
  Problem<W> *d_problems;
  cudaMalloc((void**) &d_problems, batch_size * sizeof(Problem<W>));
  cudaMemcpy(d_problems, problems.data(), batch_size * sizeof(Problem<W>), cudaMemcpyHostToDevice);

  Outcome<W> *d_outcomes;
  cudaMalloc((void**) &d_outcomes, batch_size * sizeof(Outcome<W>));

  work_kernel<N, W><<<batch_size, 32>>>(d_problems, d_outcomes);

  std::vector<Outcome<W>> outcomes;
  outcomes.resize(batch_size);
  cudaMemcpy(outcomes.data(), d_outcomes, batch_size * sizeof(Outcome<W>), cudaMemcpyDeviceToHost);

  cudaFree(d_problems);
  cudaFree(d_outcomes);

  return outcomes;
}

// Explicitly instantiate the template to the N in params.hpp, or it doesn't get compiled at all.
template std::vector<Outcome<32>>
launch_work_kernel<N, 32>(unsigned batch_size, std::vector<Problem<32>> problems);

template std::vector<Outcome<64>>
launch_work_kernel<N, 64>(unsigned batch_size, std::vector<Problem<64>> problems);
