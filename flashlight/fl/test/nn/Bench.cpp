#include <arrayfire.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

int main() {
  af::dim4 N{128};
  const auto a_base = af::randu(N);
  const auto b = fl::Variable(af::randu(N), false);
  const auto depth = 1000;
  auto run_grad = [&]() -> float {
    const auto a = fl::Variable(a_base, true);
    auto c = a * b;
    auto d = c + b;
    for (auto k = 0; k < depth; ++k) {
      d = c + b;
    }
    fl::sum(d, {}).backward();
    return a.grad().scalar<float>();
  };

  std::cerr << run_grad() << "\n";
  const auto iters = 1000;
  for (auto i = 0; i < 1000; ++i) {
    run_grad();
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < iters; ++i) {
    run_grad();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << iters / diff.count() << " iters/sec\n";
  return 0;
}
