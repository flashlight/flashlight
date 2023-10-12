/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * A transformation pass over a JIT tree.
 */
class Pass {
 public:
  Pass() = default;
  virtual ~Pass() = default;

  /**
   * Apply in-place transformation to nodes within the tree.
   *
   * @param[in] node the root node of the JIT tree to be transformed
   * @return root to the updated tree
   */
  virtual NodePtr apply(NodePtr node) = 0;
};

} // namespace fl
