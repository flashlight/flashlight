/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * A node that represents indexing operation.
 */
class IndexNode : public NodeTrait<IndexNode> {
  const std::vector<Index> indices_;

  // helps indexing into inputs
  static constexpr unsigned indexedNodeIdx = 0;

  // intentionally kept private to control allocation
  IndexNode(Node* indexedNode, const std::vector<Index>& indices);

 public:
  static constexpr NodeType nodeType = NodeType::Index;

  static IndexNode* create(Node* indexedNode, const std::vector<Index>& indices);

  Node* indexedNode() const;
  const std::vector<Index>& indices() const;
};

} // namespace fl
