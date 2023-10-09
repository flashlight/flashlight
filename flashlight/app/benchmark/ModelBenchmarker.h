/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace benchmark {

using Criterion = std::function<fl::Variable(const std::vector<fl::Variable>&)>;

class ModelBenchmarker {
 public:
  ModelBenchmarker(
      std::shared_ptr<fl::Module>& model,
      const Criterion& criterion,
      const int worldSize = 1);

  void runBenchmark(const std::vector<fl::Variable>& input);

  // Return time splits in seconds
  double getBatchTime() const;
  double getForwardTime() const;
  double getCriterionTime() const;
  double getBackwardTime() const;
  double getOptimizationTime() const;

 private:
  std::shared_ptr<fl::Module> model_;
  Criterion criterion_;
  std::shared_ptr<fl::Reducer> reducer_;

  std::shared_ptr<fl::SGDOptimizer> optimizer_;

  fl::TimeMeter batchTimerMeter_{true};
  fl::TimeMeter fwdTimeMeter_{true};
  fl::TimeMeter critFwdTimeMeter_{true};
  fl::TimeMeter bwdTimeMeter_{true};
  fl::TimeMeter optimTimeMeter_{true};

  void syncMeters();

  // TODO: support optimizer selection
  void createOptimizer();
  // TODO: support reducer selection
  void createReducer(const int worldSize);
};

} // namespace benchmark
} // namespace app
} // namespace fl
