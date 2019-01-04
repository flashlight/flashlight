/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/nn/modules/DistributedUtils.h"

#include "flashlight/distributed/distributed.h"

namespace fl {

void distributeModuleGrads(std::shared_ptr<const Module> module, double scale) {
  for (auto& param : module->params()) {
    param.registerGradHook([scale](Variable& grad) { allReduce(grad, scale); });
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
