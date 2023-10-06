/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/DistributedUtils.h"

#include "flashlight/fl/distributed/DistributedApi.h"

#include <stdexcept>

namespace fl {

void distributeModuleGrads(
    std::shared_ptr<const Module> module,
    std::shared_ptr<Reducer> reducer) {
  for (auto& param : module->params()) {
    param.registerGradHook([reducer](Variable& grad) { reducer->add(grad); });
  }
}

void allReduceParameters(std::shared_ptr<const Module> module) {
  if (!module) {
    throw std::invalid_argument("null module passed to allReduceParameters");
  }
  double scale = 1.0 / getWorldSize();
  for (auto& param : module->params()) {
    allReduce(param, scale);
  }
}

void allReduceGradients(
    std::shared_ptr<const Module> module,
    double scale /*= 1.0 */) {
  if (!module) {
    throw std::invalid_argument("null module passed to allReduceGradients");
  }
  for (auto& param : module->params()) {
    allReduce(param.grad(), scale);
  };
}

} // namespace fl
