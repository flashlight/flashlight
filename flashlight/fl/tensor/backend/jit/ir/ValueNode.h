/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

class ValueNode;
using ValueNodePtr = std::shared_ptr<ValueNode>;

/**
 * A node that represents an evaluated tensor
 */
class ValueNode : public NodeTrait<ValueNode> {
  // help control allocation while allowing `std::make_shared`
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Value;
  ValueNode(Tensor&& value, PrivateHelper);

  static ValueNodePtr create(Tensor&& value);
  const Tensor& value() const;
};

} // namespace fl
