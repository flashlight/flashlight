/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include <functional>
#include <vector>

namespace fl {

/**
 * A node that holds customized evaluation logic for things like
 * backend-specific graph rewrite or as a fallback to capture maximal graph.
 */
class CustomNode : public NodeTrait<CustomNode> {
 public:
  using EvalFunc = std::function<Tensor(const std::vector<const Tensor*>&)>;

 private:
  const std::string name_;
  const EvalFunc evalFunc_;

  // intentionally kept private to control allocation
  CustomNode(
      std::string&& name,
      std::vector<Node*>&& inputs,
      const Shape& shape,
      EvalFunc&& evalFunc);

 public:
  static constexpr NodeType nodeType = NodeType::Custom;

  static CustomNode* create(
      std::string&& debugName,
      std::vector<Node*>&& inputs,
      const Shape& shape,
      EvalFunc&& evalFunc);

  const std::string& name() const;
  const EvalFunc& evalFunc() const;
};

} // namespace fl
