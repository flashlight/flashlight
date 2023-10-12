/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

class IndexNode;
using IndexNodePtr = std::shared_ptr<IndexNode>;

/**
 * A node that represents indexing operation.
 */
class IndexNode : public NodeTrait<IndexNode> {
  const std::vector<Index> indices_;

  // helps indexing into inputs
  static constexpr unsigned indexedNodeIdx = 0;

  // help control allocation while allowing `std::make_shared`
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Index;
  IndexNode(NodePtr indexedNode, const std::vector<Index>& indices, PrivateHelper);

  static IndexNodePtr create(NodePtr indexedNode, const std::vector<Index>& indices);

  NodePtr indexedNode() const;
  const std::vector<Index>& indices() const;
};

} // namespace fl
