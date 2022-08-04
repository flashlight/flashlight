/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"

#include <utility>

namespace fl {

CustomNode::CustomNode(
    std::string&& name,
    std::vector<Node*>&& inputs,
    EvalFunc&& evalFunc)
    : NodeTrait(std::move(inputs)),
      name_(name),
      evalFunc_(std::move(evalFunc)) {}

CustomNode* CustomNode::create(
    std::string&& name,
    std::vector<Node*>&& inputs,
    EvalFunc&& evalFunc) {
  return new CustomNode(
      std::move(name), std::move(inputs), std::move(evalFunc));
}

const std::string& CustomNode::name() const {
  return name_;
}

const CustomNode::EvalFunc& CustomNode::evalFunc() const {
  return evalFunc_;
}

} // namespace fl
