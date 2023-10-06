/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "flashlight/fl/common/Timer.h"
#include "flashlight/fl/distributed/distributed.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace fl;

int main() {
  fl::init();
  distributedInit(
      DistributedInit::MPI,
      -1,
      -1,
      {{DistributedConstants::kMaxDevicePerNode, "8"}});

  auto wRank = getWorldRank();
  auto wSize = getWorldSize();

  if (wRank == 0) {
    std::cout << "Running allreduce on " << wSize << " machines" << std::endl;
  }

  const int kNumIters = 10000;
  std::vector<int64_t> sizes = {1, 2, 5};
  int64_t multiplier = 10;
  int64_t maxSize = 5000000, curMaxSize = 0;
  std::vector<double> times(kNumIters);
  while (true) {
    for (auto& size : sizes) {
      for (size_t i = 0; i < kNumIters; ++i) {
        Tensor in = fl::rand({size});
        fl::eval(in);
        fl::sync();
        auto start = fl::Timer::start();
        allReduce(in);
        fl::sync();
        times[i] = fl::Timer::stop(start);
      }
      auto timesAf = Tensor::fromVector({kNumIters}, times);
      if (wRank == 0) {
        std::cout << "Size: " << size
                  << " ; avg: " << fl::mean(timesAf).asScalar<double>() * 1000
                  << "ms ; p50: "
                  << fl::median(timesAf).asScalar<double>() * 1000 << "ms"
                  << std::endl;
      }
      curMaxSize = std::max(curMaxSize, size);
      size *= multiplier;
    }
    if (curMaxSize >= maxSize) {
      break;
    }
  }
  return 0;
}
