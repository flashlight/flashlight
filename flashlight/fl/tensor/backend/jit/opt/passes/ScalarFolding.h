/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/jit/opt/Pass.h"

namespace fl {

/**
 * An optimization pass that recursively merges scalars inside a JIT tree,
 * e.g., 1 + 2 becomes 3.
 */
class ScalarFolding : public Pass {
 public:
  NodePtr apply(NodePtr node) override;
};

} // namespace fl
