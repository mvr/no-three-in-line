#include <iostream>
#include <tuple>

#include "parsing.hpp"
#include "three_kernel.hpp"

int main() {
  auto [knownOn, knownOff] = parse_rle_history("A4.A$2.A.A$5.A.A$2A7.A$2.2A!");
  std::tie(knownOn, knownOff) = soft_branch<10>(knownOn, knownOff);
  std::cout << "x = 10, y = 10, rule = LifeHistory" << std::endl;
  std::cout << to_rle_history(knownOn, knownOff) << std::endl;
}
