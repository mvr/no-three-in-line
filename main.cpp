#include "three_kernel.hpp"

#include "params.hpp"

int main() {
  if (N > 32) {
    return solve_with_device_stack<N, 64>();
  } else {
    return solve_with_device_stack<N, 32>();
  }
}

// Last pre-multistream commit:
// https://gitlab.com/apgoucher/silk/-/commit/f4005091b4093f403e62570a44d135347d1f012f
