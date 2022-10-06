/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>
#include <numeric>

#include "flashlight/fl/common/Timer.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/jit/JitTensor.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

using namespace fl;

#define TIME_BACKEND(FUNC, TENSOR_BACKEND)                                   \
  fl::setDefaultTensorType<TENSOR_BACKEND>();                                \
  std::cout << "Timing " << #FUNC << " with " << #TENSOR_BACKEND << " ...  " \
            << std::flush;                                                   \
  std::cout << std::setprecision(5) << FUNC() * 1000.0 << " msec" << std::endl;

// NOTE order matters -- need some gap between 2 runs that both use OneDNN, else
// numbers vary a lot
#define TIME(FUNC)                            \
  TIME_BACKEND(FUNC, JitTensor<OneDnnTensor>) \
  TIME_BACKEND(FUNC, ArrayFireTensor)         \
  TIME_BACKEND(FUNC, OneDnnTensor)

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 10; ++i) {
    fn();
  }
  fl::sync();

  int iters = 20;
  fl::sync();
  auto start = fl::Timer::start();
  for (int i = 0; i < iters; i++) {
    fn();
  }
  fl::sync();
  return fl::Timer::stop(start) / iters;
}

std::vector<Tensor> getInputs(const Shape& shape, dtype type, unsigned count) {
  std::vector<Tensor> inputs;
  for (auto i = 0u; i < count; i++) {
    std::vector<int> data;
    std::iota(data.begin(), data.end(), i);
    inputs.push_back(Tensor::fromVector(shape, data, type));
    // don't benchmark input creation
    fl::eval(inputs.back());
  }
  return inputs;
}

double addMul() {
  const Shape shape({1024, 1024});
  auto inputs = getInputs(shape, dtype::f32, 200);
  auto fn = [&]() {
    auto res = fl::full(shape, 1);
    for (unsigned i = 0; i + 1 < inputs.size(); i += 2) {
      res = res + inputs[i];
      res = res * inputs[i + 1];
    }
    fl::eval(res);
  };
  return timeit(fn);
}

int main() {
  fl::init();
  TIME(addMul);
  return 0;
}
