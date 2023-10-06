/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

ValueNode::ValueNode(Tensor&& value) : NodeTrait({}, value.shape()) {
  setResult(std::move(value));
}

ValueNode* ValueNode::create(Tensor&& value) {
  return new ValueNode(std::move(value));
}

const Tensor& ValueNode::value() const {
  return getResult().value(); // guaranteed to be present (by construction)
}

} // namespace fl
