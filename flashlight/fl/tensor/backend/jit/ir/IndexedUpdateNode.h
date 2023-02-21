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
 * A node that represents an indexed update in SSA form.
 * let output = IndexedUpdateNode(indexedNode, indexings, updateDataNode),
 *     indexedTensor = indexedNode.value(),
 *     updateDataTensor = updateDataNode.value(),
 * we have
 *   output == indexedTensor __after__
 *      indexedTensor(indexings[0])(indexings[1])... = updateDataTensor
 *
 * But note that since the graph represents pure computation (SSA, no
 * assignment), this node conceptually creates a new tensor that holds the
 * result of the indexed update, thus the name.
 */
class IndexedUpdateNode : public NodeTrait<IndexedUpdateNode> {
  const std::vector<std::vector<Index>> indexings_;

  // helps indexing into inputs
  static constexpr unsigned indexedNodeIdx = 0;
  static constexpr unsigned updateDataNodeIdx = 1;

  // intentionally kept private to control allocation
  IndexedUpdateNode(
      Node* indexedNode,
      const std::vector<std::vector<Index>>& indexings,
      Node* updateDataNode);

 public:
  static constexpr NodeType nodeType = NodeType::IndexedUpdate;

  static IndexedUpdateNode* create(
      Node* indexedNode,
      const std::vector<std::vector<Index>>& indexings,
      Node* updateDataNode);

  Node* indexedNode() const;
  const std::vector<std::vector<Index>>& indexings() const;
  Node* updateDataNode() const;
};

} // namespace fl
