/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "DistributedUtils.h"

#include <flashlight/common/Exception.h>
#include <flashlight/distributed/distributed.h>

namespace fl {

void distributeModuleGrads(std::shared_ptr<const Module> module, double scale) {
  for (auto& param : module->params()) {
    param.registerGradHook([scale](Variable* grad) {
      AFML_ASSERT(grad, "Calling all reduce on a null variable.", AF_ERR_ARG);
      allReduce(*grad, scale);
    });
  }
}

void allReduceParameters(std::shared_ptr<const Module> module) {
  AFML_ASSERT(module, "Module cannot be null.", AF_ERR_ARG);
  double scale = 1.0 / getWorldSize();
  for (auto& param : module->params()) {
    allReduce(param, scale);
  }
}

void allReduceGradients(
    std::shared_ptr<const Module> module,
    double scale /*= 1.0 */) {
  AFML_ASSERT(module, "Module cannot be null.", AF_ERR_ARG);
  for (auto& param : module->params()) {
    allReduce(param.grad(), scale);
  };
}

} // namespace fl
