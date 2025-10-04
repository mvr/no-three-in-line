#include <array>
#include <bit>
#include <memory>

#include "board.cu"
#include "three_board_c4.cu"

#include "three_kernel_c4.hpp"
#include "parsing.hpp"
#include "params.hpp"
#include "queue_generic.hpp"

struct DeviceProblemC4 {
  BitBoard<32> known_on;
  BitBoard<32> known_off;
};

struct ProblemC4 {
  board_array_t<32> known_on;
  board_array_t<32> known_off;
};

template <unsigned N>
struct SolutionBufferC4 {
  board_array_t<ThreeBoardC4<N>::FULL_W> solutions[SOLUTION_BUFFER_CAPACITY];
  unsigned size;
};

template <unsigned N>
struct C4QueueTraits {
  using Element = DeviceProblemC4;
  using Slot = ProblemC4;
  using Queue = queue::Queue<C4QueueTraits<N>>;

  static constexpr uint32_t element_capacity = STACK_CAPACITY;
  static constexpr uint32_t staging_capacity = MAX_BATCH_SIZE * 2;
  static constexpr uint32_t dispatch_capacity = MAX_BATCH_SIZE;
  static constexpr uint32_t free_capacity = STACK_CAPACITY;
  static constexpr uint32_t heap_log2_warps_per_block = 5;

  __device__ static void store_element(Queue queue,
                                       uint32_t slot,
                                       const Element &element) {
    element.known_on.save(queue.elements[slot].known_on.data());
    element.known_off.save(queue.elements[slot].known_off.data());
  }

  __device__ static Element load_element(Queue queue, uint32_t slot) {
    Element result{};
    result.known_on = BitBoard<32>::load(queue.elements[slot].known_on.data());
    result.known_off = BitBoard<32>::load(queue.elements[slot].known_off.data());
    return result;
  }

  __device__ static uint32_t compute_priority(const Element &element) {
    ThreeBoardC4<N> snapshot(element.known_on, element.known_off);
    return snapshot.priority();
  }
};

template <unsigned N>
__global__ void seed_queue_kernel(typename C4QueueTraits<N>::Queue queue) {
  ThreeBoardC4<N> board;
  resolve_outcome_row(board, N / 2, queue);
}

template <unsigned N>
__device__ bool solution_buffer_push(SolutionBufferC4<N> *buffer, const ThreeBoardC4<N> &board) {
  unsigned pos;
  if ((threadIdx.x & 31) == 0) {
    pos = atomicAdd(&buffer->size, 1);
  }
  pos = __shfl_sync(0xffffffff, pos, 0);

  if (pos >= SOLUTION_BUFFER_CAPACITY) {
    return false;
  }

  ThreeBoardC4<N> expanded_board = board;
  auto full = expanded_board.expand_to_full();
  full.known_on.save(buffer->solutions[pos].data());

  return true;
}

template <unsigned N>
__device__ void resolve_outcome_row(const ThreeBoardC4<N> board,
                                    unsigned ix,
                                    typename C4QueueTraits<N>::Queue queue) {
  using Traits = C4QueueTraits<N>;
  ThreeBoardC4<N> tried_board = board;

  board_row_t<32> row_known_on, row_known_off;
  row_known_on  = board.known_on.row(ix);
  row_known_off = board.known_off.row(ix);

  board_row_t<32> col_known_on, col_known_off;
  col_known_on  = board.known_on.column(ix);
  col_known_off = board.known_off.column(ix);

  uint64_t full_on  = (uint64_t(row_known_on)  << N) | col_known_on;
  uint64_t full_off = (uint64_t(row_known_off) << N) | col_known_off;

  uint64_t remaining = ~full_on & ~full_off & (((board_row_t<64>)1 << (2*N)) - 1);

  remaining &= ~(board_row_t<64>(1) << ix);

  if (full_on == 0) {
    unsigned keep = find_last_set<64>(remaining);
    remaining &= ~(board_row_t<64>(1) << keep);
  }

  while (remaining != 0) {
    unsigned bit = find_first_set<64>(remaining);
    cuda::std::pair<unsigned, unsigned> cell;

    if(bit >= N)
      cell = {bit - N, ix};
    else
      cell = {ix, bit};

    ThreeBoardC4<N> sub_board = tried_board;
    sub_board.known_on.set(cell);
    sub_board.eliminate_all_lines(cell);
    sub_board.propagate();

    if (sub_board.consistent()) {
      DeviceProblemC4 element = {sub_board.known_on, sub_board.known_off};
      queue::enqueue<Traits>(queue, element);
    }

    tried_board.known_off.set(cell);
    remaining &= (remaining - 1);
  }

}

template <unsigned N>
__device__ void resolve_outcome_cell(const ThreeBoardC4<N> &board,
                                     cuda::std::pair<unsigned, unsigned> cell,
                                     typename C4QueueTraits<N>::Queue queue) {
  using Traits = C4QueueTraits<N>;
  {
    ThreeBoardC4<N> sub_board = board;
    BitBoard<32> seed;
    seed.set(static_cast<int>(cell.first), static_cast<int>(cell.second));
    sub_board.known_on |= seed;
    sub_board.propagate(seed);
    if (sub_board.consistent()) {
      DeviceProblemC4 element = {sub_board.known_on, sub_board.known_off};
      queue::enqueue<Traits>(queue, element);
    }
  }
  {
    ThreeBoardC4<N> sub_board = board;
    sub_board.known_off.set(static_cast<int>(cell.first), static_cast<int>(cell.second));
    sub_board.propagate();
    if (sub_board.consistent()) {
      DeviceProblemC4 element = {sub_board.known_on, sub_board.known_off};
      queue::enqueue<Traits>(queue, element);
    }
  }
}

template <unsigned N>
// __launch_bounds__(32 * WARPS_PER_BLOCK, 12)
__global__ void work_kernel(typename C4QueueTraits<N>::Queue queue,
                            SolutionBufferC4<N> *solution_buffer,
                            unsigned dispatch_base,
                            unsigned batch_size) {
  using Traits = C4QueueTraits<N>;
  const unsigned warp_idx = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / 32);

  if (warp_idx >= batch_size)
    return;

  const unsigned lane = threadIdx.x & 31;
  const uint32_t mask = 0xffffffffu;

  const uint32_t dispatch_index = dispatch_base + warp_idx;

  uint32_t element_id = queue::kInvalidIndex;
  if (lane == 0) {
    element_id = queue::load_dispatch_entry<Traits>(queue, dispatch_index);
  }
  element_id = __shfl_sync(mask, element_id, 0);

  if (element_id == queue::kInvalidIndex) {
    return;
  }

  auto element = Traits::load_element(queue, element_id);
  ThreeBoardC4<N> board;
  board.known_on = element.known_on;
  board.known_off = element.known_off;

  bool released = false;
  auto release_element = [&]() {
    if (!released) {
      queue::complete<Traits>(queue, element_id);
      released = true;
    }
  };

  board.propagate();
  if (!board.consistent()) {
    release_element();
    return;
  }

  BitBoard<32> unknown = ~(board.known_on | board.known_off) & ThreeBoardC4<N>::bounds();

  if (unknown.empty()) {
    solution_buffer_push(solution_buffer, board);
    release_element();
    return;
  }


  cuda::std::pair<unsigned, unsigned> cell;

  BitBoard<32> vulnerable = board.vulnerable();
  if(!vulnerable.empty()) {
    cell = vulnerable.first_origin_on<N>();
  } else {
    auto [row, unknown_count] = board.most_constrained_row();
    board_row_t<32> row_unknown = unknown.row(row);

    if (row_unknown != 0) {
      unsigned col = find_first_set<32>(row_unknown);
      cell = {col, row};
    } else {
      board_row_t<32> col_unknown = unknown.column(row);
      unsigned col = find_first_set<32>(col_unknown);
      cell = {row, col};
    }
  }

  resolve_outcome_cell(board, cell, queue);
  release_element();
}

template <unsigned N>
int solve_with_device_stack_c4() {
  using Traits = C4QueueTraits<N>;
  init_lookup_tables_host();
  init_relevant_endpoint_host(ThreeBoardC4<N>::FULL_N);
  init_relevant_endpoint_host_64(ThreeBoardC4<N>::FULL_N);

  queue::Queue<Traits> queue{};
  [[maybe_unused]] auto queue_cleanup = queue::make_workspace_owner<Traits>(queue);
  queue::DeviceQueueCounters host_counters{};

  if (queue::allocate_workspace<Traits>(queue) != cudaSuccess) {
    return -1;
  }
  if (queue::zero_workspace_async(queue) != cudaSuccess) {
    return -1;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return -1;
  }

  auto solution_deleter = [](SolutionBufferC4<N> *ptr) {
    if (ptr) cudaFree(ptr);
  };
  std::unique_ptr<SolutionBufferC4<N>, decltype(solution_deleter)> d_solution_buffer(nullptr, solution_deleter);
  SolutionBufferC4<N> *raw_solution = nullptr;
  if (cudaMalloc(reinterpret_cast<void **>(&raw_solution), sizeof(SolutionBufferC4<N>)) != cudaSuccess) {
    return -1;
  }
  d_solution_buffer.reset(raw_solution);
  if (cudaMemset(d_solution_buffer.get(), 0, sizeof(SolutionBufferC4<N>)) != cudaSuccess) {
    return -1;
  }

  seed_queue_kernel<N><<<1, 32>>>(queue);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return -1;
  }

  while (true) {
    queue::maintenance_kernel<Traits><<<1, 32u << Traits::heap_log2_warps_per_block>>>(queue, MAX_BATCH_SIZE);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      return -1;
    }

    cudaMemcpy(&host_counters, queue.counters, sizeof(queue::DeviceQueueCounters), cudaMemcpyDeviceToHost);

    const uint64_t dispatch_available = queue::counter_diff(host_counters,
                                                            queue::Counter::DispatchWrite,
                                                            queue::Counter::DispatchRead);

    if (dispatch_available == 0) {
      if (!queue::queue_has_work(host_counters)) {
        break;
      }
      continue;
    }

    unsigned batch_size = static_cast<unsigned>(std::min<uint64_t>(dispatch_available, MAX_BATCH_SIZE));
    if (batch_size == 0) {
      continue;
    }

    const unsigned dispatch_start = static_cast<unsigned>(queue::counter_value(host_counters, queue::Counter::DispatchRead));

    unsigned blocks = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (blocks == 0) {
      blocks = 1;
    }

    work_kernel<N><<<blocks, WARPS_PER_BLOCK * 32>>>(queue, d_solution_buffer.get(), dispatch_start, batch_size);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      return -1;
    }

    const uint64_t new_dispatch_read = queue::counter_value(host_counters, queue::Counter::DispatchRead) + batch_size;
    cudaMemcpy(&queue.counters->values[static_cast<unsigned>(queue::Counter::DispatchRead)],
               &new_dispatch_read, sizeof(uint64_t), cudaMemcpyHostToDevice);

    unsigned solution_count = 0;
    cudaMemcpy(&solution_count, &d_solution_buffer->size, sizeof(unsigned), cudaMemcpyDeviceToHost);

    if (solution_count > 0) {
      std::vector<board_array_t<ThreeBoardC4<N>::FULL_W>> host_solutions(solution_count);
      for (unsigned i = 0; i < solution_count; ++i) {
        cudaMemcpy(&host_solutions[i], &d_solution_buffer->solutions[i],
                   sizeof(board_array_t<ThreeBoardC4<N>::FULL_W>), cudaMemcpyDeviceToHost);
        std::cout << to_rle<ThreeBoardC4<N>::FULL_N, ThreeBoardC4<N>::FULL_W>(host_solutions[i]) << std::endl;
        exit(0);
      }
      cudaMemset(&d_solution_buffer->size, 0, sizeof(unsigned));
    }
  }

  return 0;
}

template int solve_with_device_stack_c4<N>();
