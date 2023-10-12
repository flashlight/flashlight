/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

#include <functional>
#include <vector>
#include <memory>

namespace fl {

class CustomNode;
using CustomNodePtr = std::shared_ptr<CustomNode>;

/**
 * A node that holds customized evaluation logic for things like
 * backend-specific graph rewrite or as a fallback to capture maximal graph.
 */
class CustomNode : public NodeTrait<CustomNode> {
 public:
  using EvalFunc = std::function<Tensor(const std::vector<const Tensor*>&)>;

  // help control allocation while allowing `std::make_shared`
  struct PrivateHelper{};

 private:
  const std::string name_;
  const EvalFunc evalFunc_;

 public:
  static constexpr NodeType nodeType = NodeType::Custom;
  CustomNode(
      std::string&& name,
      std::vector<NodePtr>&& inputs,
      const Shape& shape,
      EvalFunc&& evalFunc,
      PrivateHelper);

  static CustomNodePtr create(
      std::string&& debugName,
      std::vector<NodePtr>&& inputs,
      const Shape& shape,
      EvalFunc&& evalFunc);

  const std::string& name() const;
  const EvalFunc& evalFunc() const;
};

} // namespace fl
