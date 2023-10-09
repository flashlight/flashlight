/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * A node that represents an evaluated tensor
 */
class ValueNode : public NodeTrait<ValueNode> {
  // intentionally kept private to control allocation
  ValueNode(Tensor&& value);

 public:
  static constexpr NodeType nodeType = NodeType::Value;

  static ValueNode* create(Tensor&& value);
  const Tensor& value() const;
};

} // namespace fl
