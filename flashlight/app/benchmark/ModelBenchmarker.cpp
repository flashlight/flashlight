/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/benchmark/ModelBenchmarker.h"

#include "flashlight/pkg/runtime/common/DistributedUtils.h"
#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace benchmark {

constexpr size_t kWarmupUpdates = 50;
constexpr size_t kRunUpdates = 100;

ModelBenchmarker::ModelBenchmarker(
    std::shared_ptr<fl::Module>& model,
    const Criterion& criterion,
    const int worldSize)
    : model_(model), criterion_(criterion) {
  createOptimizer();
  createReducer(worldSize);
}

void ModelBenchmarker::runBenchmark(const std::vector<fl::Variable>& input) {
  model_->train();

  // Warmup
  for (int i = 0; i < kWarmupUpdates; i++) {
    optimizer_->zeroGrad();
    auto output = model_->forward(input);
    auto loss = criterion_(output);
    loss.backward();
    if (reducer_) {
      reducer_->finalize();
    }
    optimizer_->step();
  }
  fl::sync();

  // Benchmark
  for (int i = 0; i < kRunUpdates; i++) {
    batchTimerMeter_.resume();
    optimizer_->zeroGrad();

    // 1. model forward
    fwdTimeMeter_.resume();
    auto output = model_->forward(input);
    fl::sync();
    fwdTimeMeter_.stopAndIncUnit();

    // 2. criterion forward
    critFwdTimeMeter_.resume();
    auto loss = criterion_(output);
    fl::sync();
    critFwdTimeMeter_.stopAndIncUnit();

    // 3. backward
    bwdTimeMeter_.resume();
    loss.backward();
    if (reducer_) {
      reducer_->finalize();
    }
    fl::sync();
    bwdTimeMeter_.stopAndIncUnit();

    // 4. optimize
    optimTimeMeter_.resume();
    optimizer_->step();
    fl::sync();
    optimTimeMeter_.stopAndIncUnit();

    batchTimerMeter_.stopAndIncUnit();
  }

  syncMeters();
}

double ModelBenchmarker::getBatchTime() const {
  return batchTimerMeter_.value();
}

double ModelBenchmarker::getForwardTime() const {
  return fwdTimeMeter_.value();
}

double ModelBenchmarker::getCriterionTime() const {
  return critFwdTimeMeter_.value();
}

double ModelBenchmarker::getBackwardTime() const {
  return bwdTimeMeter_.value();
}

double ModelBenchmarker::getOptimizationTime() const {
  return optimTimeMeter_.value();
}

void ModelBenchmarker::syncMeters() {
  fl::pkg::runtime::syncMeter(batchTimerMeter_);
  fl::pkg::runtime::syncMeter(fwdTimeMeter_);
  fl::pkg::runtime::syncMeter(critFwdTimeMeter_);
  fl::pkg::runtime::syncMeter(bwdTimeMeter_);
  fl::pkg::runtime::syncMeter(optimTimeMeter_);
}

void ModelBenchmarker::createOptimizer() {
  auto parameters = model_->params();
  optimizer_ = std::make_shared<fl::SGDOptimizer>(
      parameters,
      0.1, // lr
      0.9, // momentum
      0.1 // weight_decay
  );
}

void ModelBenchmarker::createReducer(const int worldSize) {
  if (worldSize <= 1) {
    return;
  }

  reducer_ =
      std::make_shared<fl::CoalescingReducer>(1.0 / worldSize, true, true);
  fl::distributeModuleGrads(model_, reducer_);
}

} // namespace benchmark
} // namespace app
} // namespace fl
