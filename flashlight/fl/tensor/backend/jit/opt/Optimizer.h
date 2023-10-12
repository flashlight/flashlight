/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"
#include "flashlight/fl/tensor/backend/jit/opt/Pass.h"

namespace fl {

/**
 * A JIT tree optimizer.
 */
class Optimizer {
  std::vector<std::unique_ptr<Pass>> passes_;
  // backend used for optional JIT optimizer extension
  TensorBackend& backend_;

 public:
  explicit Optimizer(TensorBackend& backend);

  /**
   * Apply in-place optimization to nodes within the tree.
   *
   * @param[in] node the root node of the JIT tree to be optimized
   * @return root to the updated tree
   */
  NodePtr optimize(NodePtr node);
};

} // namespace fl
