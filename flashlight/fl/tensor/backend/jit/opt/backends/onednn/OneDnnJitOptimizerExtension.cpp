/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/OneDnnJitOptimizerExtension.h"

#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/OneDnnOpFusion.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"

namespace fl {

std::vector<std::unique_ptr<Pass>> OneDnnJitOptimizerExtension::passes() {
  std::vector<std::unique_ptr<Pass>> passes;
  passes.emplace_back(std::make_unique<OneDnnOpFusion>());
  return passes;
}

bool OneDnnJitOptimizerExtension::isDataTypeSupported(
    const fl::dtype& dtype) const {
  return OneDnnBackend::getInstance().isDataTypeSupported(dtype);
}

} // namespace fl
