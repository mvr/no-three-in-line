#include <array>
#include <vector>

#include "board.cu"
#include "three_board.cu"

#include "three_kernel.hpp"

#include "params.hpp"

template <unsigned N>
__global__ void work_kernel(Problem *problems, Outcome *outcomes) {
  Problem &problem = problems[blockIdx.x];
  BitBoard seed = BitBoard::load(problem.seed.data());

  ThreeBoard<N> board;
  board.knownOn = BitBoard::load(problem.knownOn.data()) | seed;
  board.knownOff = BitBoard::load(problem.knownOff.data());

  board.eliminate_all_lines(seed);
  board.propagate();
  board.soft_branch_all();

  Outcome &outcome = outcomes[blockIdx.x];
  board.knownOn.save(outcome.knownOn.data());
  board.knownOff.save(outcome.knownOff.data());

  unsigned unknown_pop = board.unknown_pop();
  bool consistent = board.consistent();
  auto [row, _] = board.most_constrained_row();

  if(threadIdx.x == 0) {
    outcome.unknownPop = unknown_pop;
    outcome.solved = outcome.unknownPop == 0;
    outcome.consistent = consistent;
    outcome.ix = row;
  }
}

template <unsigned N>
std::vector<Outcome>
launch_work_kernel(unsigned batch_size,
                   std::vector<Problem> problems
                   ) {
  Problem *d_problems;
  cudaMalloc((void**) &d_problems, batch_size * sizeof(Problem));
  cudaMemcpy(d_problems, problems.data(), batch_size * sizeof(Problem), cudaMemcpyHostToDevice);

  Outcome *d_outcomes;
  cudaMalloc((void**) &d_outcomes, batch_size * sizeof(Outcome));

  work_kernel<N><<<batch_size, 32>>>(d_problems, d_outcomes);

  std::vector<Outcome> outcomes;
  outcomes.resize(batch_size);
  cudaMemcpy(outcomes.data(), d_outcomes, batch_size * sizeof(Outcome), cudaMemcpyDeviceToHost);

  cudaFree(d_problems);
  cudaFree(d_outcomes);

  return outcomes;

}

// Explicitly instantiate the template to the N in params.hpp, or it doesn't get compiled at all.
template std::vector<Outcome>
launch_work_kernel<N>(unsigned batch_size, std::vector<Problem> problems);
