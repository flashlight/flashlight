/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/benchmark/Utils.h"

#include <iomanip>

#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/text/String.h"

namespace fl {
namespace app {
namespace benchmark {

void init() {
  fl::DynamicBenchmark::setBenchmarkMode(true);
  fl::OptimMode::get().setOptimLevel(fl::OptimLevel::DEFAULT);
}

void printInfo(
    std::string&& name,
    bool fp16,
    const fl::app::benchmark::ModelBenchmarker& benchmarker,
    int numUnits,
    bool verbose) {
  if (fl::getWorldRank() != 0) {
    return;
  }
  name += fp16 ? " + AMP" : "";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n----- " + name + " -----" << std::endl;

  std::cout << "Throughput: "
            << (numUnits * fl::getWorldSize() / benchmarker.getBatchTime());
  std::cout << std::endl;

  if (verbose) {
    std::cout << "\nBatch Time(ms): " << benchmarker.getBatchTime() * 1000;
    std::cout << "\nModel Forward Time(ms): "
              << benchmarker.getForwardTime() * 1000;
    std::cout << "\nCriterion Forward Time(ms): "
              << benchmarker.getCriterionTime() * 1000;
    std::cout << "\nBackward Time(ms): "
              << benchmarker.getBackwardTime() * 1000;
    std::cout << "\nOptimization Time(ms): "
              << benchmarker.getOptimizationTime() * 1000;
    std::cout << std::endl;

    fl::detail::getMemMgrInfo("Memory Manager Stats", /* device id = */ 0);
  }
}

} // namespace benchmark
} // namespace app
} // namespace fl
