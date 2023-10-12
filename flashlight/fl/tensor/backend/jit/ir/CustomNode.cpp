/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"

#include <utility>

namespace fl {

CustomNode::CustomNode(
    std::string&& name,
    std::vector<NodePtr>&& inputs,
    const Shape& shape,
    EvalFunc&& evalFunc,
    PrivateHelper)
    : NodeTrait(std::move(inputs), shape),
      name_(name),
      evalFunc_(std::move(evalFunc)) {}

CustomNodePtr CustomNode::create(
    std::string&& name,
    std::vector<NodePtr>&& inputs,
    const Shape& shape,
    EvalFunc&& evalFunc) {
  return std::make_shared<CustomNode>(
      std::move(name), std::move(inputs), shape, std::move(evalFunc),
      PrivateHelper{});
}

const std::string& CustomNode::name() const {
  return name_;
}

const CustomNode::EvalFunc& CustomNode::evalFunc() const {
  return evalFunc_;
}

} // namespace fl
