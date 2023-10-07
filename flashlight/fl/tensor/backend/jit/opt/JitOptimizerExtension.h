/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorExtension.h"
#include "flashlight/fl/tensor/backend/jit/opt/Pass.h"

namespace fl {

/**
 * Tensor Extension to enable backend-specific JIT graph optimization.
 */
class JitOptimizerExtension : public TensorExtension<JitOptimizerExtension> {
 public:
  static constexpr auto extensionType = TensorExtensionType::JitOptimizer;

  /**
   * Get backend-specific optimization passes.
   */
  virtual std::vector<std::unique_ptr<Pass>> passes() = 0;
};

} // namespace fl
