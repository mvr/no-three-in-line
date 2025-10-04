#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include <cpads/sorting/bitonic.hpp>

namespace queue {

namespace heap_detail {

using HeapVector = hh::vec<uint64_t, 2>;

template <size_t Log2WarpsPerBlock, bool ReplaceWithZeros = false>
__device__ inline HeapVector load_heap_element(int n, uint4 *heap) {
  constexpr uint64_t heap_stride = 32u << Log2WarpsPerBlock;
  uint4 yv = heap[(n - 1) * heap_stride + threadIdx.x];
  HeapVector y;
  y[0] = yv.x | (static_cast<uint64_t>(yv.y) << 32);
  y[1] = yv.z | (static_cast<uint64_t>(yv.w) << 32);
  if constexpr (ReplaceWithZeros) {
    yv.x = 0;
    yv.y = 0;
    yv.z = 0;
    yv.w = 0;
    heap[(n - 1) * heap_stride + threadIdx.x] = yv;
  }
  return y;
}

template <size_t Log2WarpsPerBlock>
__device__ inline void store_heap_element(int n, uint4 *heap, HeapVector &y) {
  constexpr uint64_t heap_stride = 32u << Log2WarpsPerBlock;
  uint4 yv;
  yv.x = static_cast<uint32_t>(y[0]);
  yv.y = static_cast<uint32_t>(y[0] >> 32);
  yv.z = static_cast<uint32_t>(y[1]);
  yv.w = static_cast<uint32_t>(y[1] >> 32);
  heap[(n - 1) * heap_stride + threadIdx.x] = yv;
}

template <size_t Log2WarpsPerBlock>
__device__ inline void heap_parallel_insert(HeapVector &x,
                                            int n,
                                            uint4 *heap,
                                            uint64_t *smem) {
  __syncthreads();
  hh::block_bitonic_sort<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(x, smem);

  int child_idx = n;
  while (child_idx > 1) {
    int parent_idx = child_idx >> 1;
    auto y = load_heap_element<Log2WarpsPerBlock>(parent_idx, heap);

#pragma unroll
    for (int i = 0; i < 2; i++) {
      hh::compare_and_swap(y[i], x[i]);
    }

    hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(y, smem, true);
    store_heap_element<Log2WarpsPerBlock>(child_idx, heap, y);
    child_idx = parent_idx;
  }

  hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(x, smem, true);
  store_heap_element<Log2WarpsPerBlock>(1, heap, x);
  __syncthreads();
}

template <size_t Log2WarpsPerBlock>
__device__ inline void heap_parallel_delete(int n, uint4 *heap, uint64_t *smem) {
  constexpr int last_thread = (32 << Log2WarpsPerBlock) - 1;

  __syncthreads();
  auto x = load_heap_element<Log2WarpsPerBlock, true>(n, heap);
  __syncthreads();
  int parent_idx = 1;
  while (2 * parent_idx < n) {
    int lidx = 2 * parent_idx;
    int ridx = lidx + 1;

    auto xl = load_heap_element<Log2WarpsPerBlock>(lidx, heap);
    auto xr = load_heap_element<Log2WarpsPerBlock>(ridx, heap);

    __syncthreads();
    if (threadIdx.x == last_thread) {
      smem[0] = xl[1];
      smem[1] = xr[1];
    }
    __syncthreads();
    bool left_was_bigger = (smem[0] >= smem[1]);
    __syncthreads();

    hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xr, smem);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      hh::compare_and_swap(xl[i], xr[i]);
    }
    hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xr, smem);
    hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xl, smem, true);
    store_heap_element<Log2WarpsPerBlock>(lidx + left_was_bigger, heap, xl);

#pragma unroll
    for (int i = 0; i < 2; i++) {
      hh::compare_and_swap(x[i], xr[i]);
    }
    hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(xr, smem, true);
    hh::block_bitonic_merge<Log2WarpsPerBlock, Log2WarpsPerBlock + 5>(x, smem, true);
    store_heap_element<Log2WarpsPerBlock>(parent_idx, heap, xr);
    parent_idx = ridx - left_was_bigger;
  }

  store_heap_element<Log2WarpsPerBlock>(parent_idx, heap, x);
  __syncthreads();
}

}  // namespace heap_detail

constexpr uint32_t kInvalidIndex = 0xffffffffu;

enum class Counter : uint32_t {
  StagingRead = 0,
  StagingWrite,
  HeapNodes,
  FreeRead,
  FreeWrite,
  DispatchRead,
  DispatchWrite,
  ProblemReserve,
  ActiveProblems,
  Count
};

struct DeviceQueueCounters {
  uint64_t values[static_cast<unsigned>(Counter::Count)];
};

namespace detail {

template <typename Traits>
constexpr uint32_t heap_threads_per_node() {
  return 32u << Traits::heap_log2_warps_per_block;
}

template <typename Traits>
constexpr uint32_t heap_vector_width() {
  return 2u * heap_threads_per_node<Traits>();
}

template <typename Traits>
constexpr uint32_t heap_node_stride() {
  return heap_threads_per_node<Traits>();
}

template <typename Traits>
constexpr uint32_t heap_max_nodes() {
  return Traits::problem_capacity / heap_vector_width<Traits>();
}

template <typename Traits>
constexpr uint32_t entries_per_lane() {
  return heap_vector_width<Traits>() / heap_threads_per_node<Traits>();
}

}  // namespace detail

template <typename Traits>
struct Queue {
  using ProblemSlot = typename Traits::ProblemSlot;

  static_assert(Traits::problem_capacity > 0, "problem_capacity must be positive");
  static_assert((Traits::staging_capacity & (Traits::staging_capacity - 1u)) == 0,
                "staging ring must be power of two");
  static_assert((Traits::dispatch_capacity & (Traits::dispatch_capacity - 1u)) == 0,
                "dispatch ring must be power of two");
  static_assert((Traits::free_capacity & (Traits::free_capacity - 1u)) == 0,
                "free list ring must be power of two");
  static_assert(detail::heap_max_nodes<Traits>() >= 2, "heap needs space for root and child");
  static_assert(Traits::problem_capacity % detail::heap_vector_width<Traits>() == 0,
                "heap vectors must tile the problem pool");

  ProblemSlot *problem_pool = nullptr;
  uint64_t *staging_entries = nullptr;
  uint32_t *dispatch_ids = nullptr;
  uint32_t *free_ids = nullptr;
  uint4 *heap_nodes = nullptr;
  DeviceQueueCounters *counters = nullptr;
};

template <typename Traits>
constexpr uint32_t free_list_mask() {
  return Traits::free_capacity - 1u;
}

template <typename Traits>
constexpr uint32_t staging_mask() {
  return Traits::staging_capacity - 1u;
}

template <typename Traits>
constexpr uint32_t dispatch_mask() {
  return Traits::dispatch_capacity - 1u;
}

template <typename Traits>
constexpr uint32_t heap_threads_per_node() {
  return detail::heap_threads_per_node<Traits>();
}

template <typename Traits>
constexpr uint32_t heap_vector_width() {
  return detail::heap_vector_width<Traits>();
}

template <typename Traits>
constexpr uint32_t heap_node_stride() {
  return detail::heap_node_stride<Traits>();
}

template <typename Traits>
constexpr uint32_t heap_max_nodes() {
  return detail::heap_max_nodes<Traits>();
}

template <typename Traits>
constexpr size_t problem_pool_bytes() {
  return static_cast<size_t>(Traits::problem_capacity) * sizeof(typename Traits::ProblemSlot);
}

template <typename Traits>
constexpr size_t staging_ring_bytes() {
  return static_cast<size_t>(Traits::staging_capacity) * sizeof(uint64_t);
}

template <typename Traits>
constexpr size_t dispatch_ring_bytes() {
  return static_cast<size_t>(Traits::dispatch_capacity) * sizeof(uint32_t);
}

template <typename Traits>
constexpr size_t free_list_bytes() {
  return static_cast<size_t>(Traits::free_capacity) * sizeof(uint32_t);
}

template <typename Traits>
constexpr size_t heap_bytes() {
  return static_cast<size_t>(heap_max_nodes<Traits>()) * heap_node_stride<Traits>() * sizeof(uint4);
}

template <typename Traits>
constexpr size_t counter_bytes() {
  return sizeof(DeviceQueueCounters);
}

template <typename Traits>
inline cudaError_t allocate_workspace(Queue<Traits> &workspace) {
  workspace = Queue<Traits>{};
  auto allocate = [&](auto *&ptr, size_t bytes) -> cudaError_t {
    if (bytes == 0) {
      ptr = nullptr;
      return cudaSuccess;
    }
    auto err = cudaMalloc(reinterpret_cast<void **>(&ptr), bytes);
    if (err != cudaSuccess) {
      ptr = nullptr;
    }
    return err;
  };

  if (auto err = allocate(workspace.problem_pool, problem_pool_bytes<Traits>()); err != cudaSuccess) {
    free_workspace(workspace);
    return err;
  }
  if (auto err = allocate(workspace.staging_entries, staging_ring_bytes<Traits>()); err != cudaSuccess) {
    free_workspace(workspace);
    return err;
  }
  if (auto err = allocate(workspace.dispatch_ids, dispatch_ring_bytes<Traits>()); err != cudaSuccess) {
    free_workspace(workspace);
    return err;
  }
  if (auto err = allocate(workspace.free_ids, free_list_bytes<Traits>()); err != cudaSuccess) {
    free_workspace(workspace);
    return err;
  }
  if (auto err = allocate(workspace.heap_nodes, heap_bytes<Traits>()); err != cudaSuccess) {
    free_workspace(workspace);
    return err;
  }
  if (auto err = allocate(workspace.counters, counter_bytes<Traits>()); err != cudaSuccess) {
    free_workspace(workspace);
    return err;
  }

  return cudaSuccess;
}

template <typename Traits>
inline void free_workspace(Queue<Traits> &workspace) {
  auto release = [&](auto *&ptr) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  };

  release(workspace.problem_pool);
  release(workspace.staging_entries);
  release(workspace.dispatch_ids);
  release(workspace.free_ids);
  release(workspace.heap_nodes);
  release(workspace.counters);
}

template <typename Traits>
class WorkspaceOwner {
 public:
  explicit WorkspaceOwner(Queue<Traits> &workspace)
      : workspace_(&workspace), owns_(true) {}

  WorkspaceOwner(const WorkspaceOwner &) = delete;
  WorkspaceOwner &operator=(const WorkspaceOwner &) = delete;

  WorkspaceOwner(WorkspaceOwner &&other) noexcept
      : workspace_(other.workspace_), owns_(other.owns_) {
    other.workspace_ = nullptr;
    other.owns_ = false;
  }

  WorkspaceOwner &operator=(WorkspaceOwner &&other) noexcept {
    if (this != &other) {
      release();
      workspace_ = other.workspace_;
      owns_ = other.owns_;
      other.workspace_ = nullptr;
      other.owns_ = false;
    }
    return *this;
  }

  ~WorkspaceOwner() { release(); }

 private:
  void release() {
    if (owns_ && workspace_) {
      free_workspace(*workspace_);
    }
    workspace_ = nullptr;
    owns_ = false;
  }

  Queue<Traits> *workspace_;
  bool owns_;
};

template <typename Traits>
WorkspaceOwner<Traits> make_workspace_owner(Queue<Traits> &workspace) {
  return WorkspaceOwner<Traits>(workspace);
}

inline __host__ __device__ uint64_t pack_problem_key(uint32_t priority, uint32_t index) {
  return (uint64_t(priority) << 32) | uint64_t(index);
}

inline __host__ __device__ uint32_t unpack_problem_index(uint64_t key) {
  return static_cast<uint32_t>(key);
}

inline __host__ __device__ uint32_t unpack_problem_priority(uint64_t key) {
  return static_cast<uint32_t>(key >> 32);
}

template <typename Traits>
inline cudaError_t zero_workspace_async(const Queue<Traits> &queue, cudaStream_t stream = nullptr) {
  auto try_memset = [&](void *ptr, size_t bytes) {
    if (!ptr || bytes == 0) {
      return cudaSuccess;
    }
    return cudaMemsetAsync(ptr, 0, bytes, stream);
  };

  if (auto err = try_memset(queue.problem_pool, problem_pool_bytes<Traits>()); err != cudaSuccess) {
    return err;
  }
  if (auto err = try_memset(queue.staging_entries, staging_ring_bytes<Traits>()); err != cudaSuccess) {
    return err;
  }
  if (auto err = try_memset(queue.dispatch_ids, dispatch_ring_bytes<Traits>()); err != cudaSuccess) {
    return err;
  }
  if (auto err = try_memset(queue.free_ids, free_list_bytes<Traits>()); err != cudaSuccess) {
    return err;
  }
  if (auto err = try_memset(queue.heap_nodes, heap_bytes<Traits>()); err != cudaSuccess) {
    return err;
  }
  if (auto err = try_memset(queue.counters, counter_bytes<Traits>()); err != cudaSuccess) {
    return err;
  }

  if (queue.free_ids && Traits::problem_capacity > 0) {
    std::vector<uint32_t> host_free(Traits::problem_capacity);
    std::iota(host_free.begin(), host_free.end(), 0u);
    const size_t bytes = host_free.size() * sizeof(uint32_t);
    if (auto err = cudaMemcpyAsync(queue.free_ids, host_free.data(), bytes, cudaMemcpyHostToDevice, stream);
        err != cudaSuccess) {
      return err;
    }
  }

  if (queue.counters) {
    DeviceQueueCounters init{};
    init.values[static_cast<unsigned>(Counter::FreeRead)] = 0ull;
    init.values[static_cast<unsigned>(Counter::FreeWrite)] = Traits::problem_capacity;
    if (auto err = cudaMemcpyAsync(queue.counters, &init, sizeof(init), cudaMemcpyHostToDevice, stream);
        err != cudaSuccess) {
      return err;
    }
  }

  return cudaSuccess;
}

inline __host__ __device__ uint64_t counter_value(const DeviceQueueCounters &counters, Counter which) {
  return counters.values[static_cast<unsigned>(which)];
}

inline __host__ __device__ uint64_t counter_diff(const DeviceQueueCounters &counters,
                                                 Counter hi,
                                                 Counter lo) {
  return counter_value(counters, hi) - counter_value(counters, lo);
}

inline __host__ __device__ bool queue_has_work(const DeviceQueueCounters &counters) {
  const uint64_t active = counter_value(counters, Counter::ActiveProblems);
  const uint64_t staging = counter_diff(counters, Counter::StagingWrite, Counter::StagingRead);
  const uint64_t heap = counter_value(counters, Counter::HeapNodes);
  return active != 0 || staging != 0 || heap != 0;
}

template <typename Traits>
__device__ inline unsigned long long *counter_ptr(Queue<Traits> queue, Counter which) {
  return reinterpret_cast<unsigned long long *>(
      &queue.counters->values[static_cast<unsigned>(which)]);
}

template <typename Traits>
__device__ inline const unsigned long long *counter_ptr_const(const Queue<Traits> &queue,
                                                              Counter which) {
  return reinterpret_cast<const unsigned long long *>(
      &queue.counters->values[static_cast<unsigned>(which)]);
}

template <typename Traits>
__device__ uint32_t reserve_problem_index(Queue<Traits> queue) {
  constexpr uint32_t invalid = kInvalidIndex;
  if (!queue.counters) {
    return invalid;
  }

  auto *free_read_ptr = counter_ptr(queue, Counter::FreeRead);
  const auto *free_write_ptr = counter_ptr_const(queue, Counter::FreeWrite);
  auto *reserve_counter = counter_ptr(queue, Counter::ProblemReserve);

  const uint32_t capacity = Traits::problem_capacity;

  unsigned lane = threadIdx.x & 31u;
  uint32_t reservation = invalid;

  if (lane == 0) {
    while (true) {
      unsigned long long free_read = *free_read_ptr;
      unsigned long long free_write = *free_write_ptr;
      if (free_read < free_write && queue.free_ids) {
        if (atomicCAS(free_read_ptr, free_read, free_read + 1ull) == free_read) {
          const uint32_t pos = static_cast<uint32_t>(free_read) & free_list_mask<Traits>();
          reservation = queue.free_ids[pos];
          break;
        }
        continue;
      }

      unsigned long long slot = atomicAdd(reserve_counter, 1ull);
      if (slot < capacity) {
        reservation = static_cast<uint32_t>(slot);
      }
      break;
    }
  }

  reservation = __shfl_sync(0xffffffffu, reservation, 0);
  return reservation;
}

template <typename Traits>
__device__ void stage_problem(Queue<Traits> queue,
                              const typename Traits::Problem &problem) {
  constexpr uint32_t invalid = kInvalidIndex;

  if (!queue.problem_pool || !queue.counters || !queue.staging_entries) {
    return;
  }

  const uint32_t slot = reserve_problem_index<Traits>(queue);
  if (slot == invalid) {
    return;
  }

  Traits::store_problem(queue, slot, problem);
  const uint32_t priority = Traits::compute_priority(problem);

  const unsigned lane = threadIdx.x & 31u;
  if (lane == 0) {
    auto *staging_write = counter_ptr(queue, Counter::StagingWrite);
    const uint64_t write_index = atomicAdd(staging_write, 1ull);
    const uint32_t pos = static_cast<uint32_t>(write_index) & staging_mask<Traits>();
    queue.staging_entries[pos] = pack_problem_key(priority, slot);

    auto *active_counter = counter_ptr(queue, Counter::ActiveProblems);
    atomicAdd(active_counter, 1ull);
  }
}

template <typename Traits>
__device__ void enqueue_problem(Queue<Traits> queue,
                                const typename Traits::Problem &problem) {
  stage_problem<Traits>(queue, problem);
}

template <typename Traits>
__device__ void problem_complete(Queue<Traits> queue, uint32_t problem_id) {
  if (!queue.counters) {
    return;
  }

  const unsigned lane = threadIdx.x & 31u;
  if (lane == 0) {
    auto *active_counter = counter_ptr(queue, Counter::ActiveProblems);
    atomicAdd(active_counter, 0xffffffffffffffffull);

    if (queue.free_ids) {
      auto *free_write = counter_ptr(queue, Counter::FreeWrite);
      const uint64_t write_index = atomicAdd(free_write, 1ull);
      const uint32_t pos = static_cast<uint32_t>(write_index) & free_list_mask<Traits>();
      queue.free_ids[pos] = problem_id;
    }
  }
}

__device__ inline uint64_t lane_entry_base() {
  return static_cast<uint64_t>(threadIdx.x) * 2ull;
}

template <typename Traits>
__device__ void load_staging_vector(const Queue<Traits> &queue,
                                    uint64_t start_index,
                                    heap_detail::HeapVector &out) {
  const uint32_t mask = staging_mask<Traits>();
  const uint64_t lane_base = lane_entry_base();
  out[0] = queue.staging_entries[(start_index + lane_base + 0) & mask];
  out[1] = queue.staging_entries[(start_index + lane_base + 1) & mask];
}

template <typename Traits>
__device__ void store_dispatch_vector(const Queue<Traits> &queue,
                                      uint64_t start_index,
                                      const heap_detail::HeapVector &values) {
  const uint32_t mask = dispatch_mask<Traits>();
  const uint64_t lane_base = lane_entry_base();
  queue.dispatch_ids[(start_index + lane_base + 0) & mask] = unpack_problem_index(values[0]);
  queue.dispatch_ids[(start_index + lane_base + 1) & mask] = unpack_problem_index(values[1]);
}

template <typename Traits>
__device__ uint32_t load_dispatch_entry(const Queue<Traits> &queue, uint64_t index) {
  const uint32_t mask = dispatch_mask<Traits>();
  return queue.dispatch_ids[index & mask];
}

template <typename Traits>
__device__ uint32_t staging_problem_id(const Queue<Traits> &queue, uint64_t index) {
  return unpack_problem_index(queue.staging_entries[index & staging_mask<Traits>()]);
}

template <typename Traits>
__device__ void store_dispatch_entry(const Queue<Traits> &queue,
                                     uint64_t index,
                                     uint32_t problem_id) {
  queue.dispatch_ids[index & dispatch_mask<Traits>()] = problem_id;
}

template <typename Traits>
__device__ void set_counter(Queue<Traits> queue, Counter which, uint64_t value) {
  queue.counters->values[static_cast<unsigned>(which)] = value;
}

template <typename Traits>
__global__ void maintenance_kernel(Queue<Traits> queue,
                                   unsigned dispatch_quota) {
  if (!queue.counters || !queue.heap_nodes || !queue.staging_entries || !queue.dispatch_ids) {
    return;
  }

  __shared__ uint64_t scratch[heap_vector_width<Traits>()];

  uint64_t staging_read = queue.counters->values[static_cast<unsigned>(Counter::StagingRead)];
  const uint64_t staging_write = queue.counters->values[static_cast<unsigned>(Counter::StagingWrite)];
  uint64_t heap_nodes = queue.counters->values[static_cast<unsigned>(Counter::HeapNodes)];
  const uint64_t heap_capacity = heap_max_nodes<Traits>();
  const uint64_t dispatch_read = queue.counters->values[static_cast<unsigned>(Counter::DispatchRead)];
  uint64_t dispatch_write = queue.counters->values[static_cast<unsigned>(Counter::DispatchWrite)];

  uint64_t staging_available = staging_write - staging_read;
  uint64_t vectors_available = staging_available / heap_vector_width<Traits>();
  uint64_t vector_slots = (heap_nodes < heap_capacity) ? (heap_capacity - heap_nodes) : 0ull;
  uint64_t vectors_to_insert = vectors_available < vector_slots ? vectors_available : vector_slots;

  uint64_t read_ptr = staging_read;
  for (uint64_t vec = 0; vec < vectors_to_insert; ++vec) {
    heap_detail::HeapVector values;
    load_staging_vector(queue, read_ptr, values);

    heap_detail::heap_parallel_insert<Traits::heap_log2_warps_per_block>(values,
                                                                         static_cast<int>(heap_nodes + 1),
                                                                         queue.heap_nodes,
                                                                         scratch);

    heap_nodes += 1;
    read_ptr += heap_vector_width<Traits>();
  }
  staging_read = read_ptr;

  uint64_t dispatch_capacity_entries = Traits::dispatch_capacity - (dispatch_write - dispatch_read);
  uint64_t quota_entries = dispatch_quota;
  uint64_t vectors_quota = quota_entries / heap_vector_width<Traits>();
  uint64_t capacity_vectors = dispatch_capacity_entries / heap_vector_width<Traits>();

  uint64_t vectors_to_remove = heap_nodes;
  if (vectors_to_remove > vectors_quota) vectors_to_remove = vectors_quota;
  if (vectors_to_remove > capacity_vectors) vectors_to_remove = capacity_vectors;

  uint64_t write_ptr = dispatch_write;
  for (uint64_t vec = 0; vec < vectors_to_remove; ++vec) {
    auto top = heap_detail::load_heap_element<Traits::heap_log2_warps_per_block>(1, queue.heap_nodes);
    store_dispatch_vector(queue, write_ptr, top);

    heap_detail::heap_parallel_delete<Traits::heap_log2_warps_per_block>(static_cast<int>(heap_nodes),
                                                                        queue.heap_nodes,
                                                                        scratch);
    heap_nodes -= 1;
    write_ptr += heap_vector_width<Traits>();
  }

  uint64_t entries_from_heap = vectors_to_remove * heap_vector_width<Traits>();
  uint64_t quota_remaining = (quota_entries > entries_from_heap) ? (quota_entries - entries_from_heap) : 0ull;
  if (dispatch_capacity_entries > entries_from_heap) {
    dispatch_capacity_entries -= entries_from_heap;
  } else {
    dispatch_capacity_entries = 0;
  }

  uint64_t staging_remaining = staging_write - staging_read;
  uint64_t direct_capacity = dispatch_capacity_entries;
  if (direct_capacity > quota_remaining) {
    direct_capacity = quota_remaining;
  }
  uint64_t direct_count = staging_remaining < direct_capacity ? staging_remaining : direct_capacity;

  if (direct_count > 0) {
    const uint64_t src_base = staging_read;
    const uint64_t dst_base = write_ptr;
    for (uint64_t offset = threadIdx.x; offset < direct_count; offset += blockDim.x) {
      const uint32_t idx = staging_problem_id(queue, src_base + offset);
      store_dispatch_entry(queue, dst_base + offset, idx);
    }
    staging_read += direct_count;
    write_ptr += direct_count;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    set_counter(queue, Counter::StagingRead, staging_read);
    set_counter(queue, Counter::HeapNodes, heap_nodes);
    set_counter(queue, Counter::DispatchWrite, write_ptr);
  }
}

}  // namespace queue
