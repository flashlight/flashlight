/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/IndexedUpdateNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

namespace fl {

namespace {

std::vector<Node*> computeInputs(
    Node* indexedNode,
    Node* updateDataNode,
    const std::vector<std::vector<Index>>& indexings) {
  std::vector<Node*> inputs{indexedNode, updateDataNode};
  for (const auto& indexing : indexings) {
    for (const auto& idx : indexing) {
      switch (idx.type()) {
        case detail::IndexType::Tensor: {
          const auto& tensorIdx = idx.get<Tensor>();
          const auto tensorIdxNode = toJitTensorBase(tensorIdx).node();
          inputs.push_back(tensorIdxNode);
        }
        default:
          continue;
      }
    }
  }
  return inputs;
}

} // namespace

IndexedUpdateNode::IndexedUpdateNode(
    Node* indexedNode,
    const std::vector<std::vector<Index>>& indexings,
    Node* updateDataNode)
    : NodeTrait(
          computeInputs(indexedNode, updateDataNode, indexings),
          Shape(indexedNode->shape())),
      indexings_(indexings) {}

IndexedUpdateNode* IndexedUpdateNode::create(
    Node* indexedNode,
    const std::vector<std::vector<Index>>& indexings,
    Node* updateDataNode) {
  return new IndexedUpdateNode(indexedNode, indexings, updateDataNode);
}

Node* IndexedUpdateNode::indexedNode() const {
  return getInput(indexedNodeIdx);
}

const std::vector<std::vector<Index>>& IndexedUpdateNode::indexings() const {
  return indexings_;
}

Node* IndexedUpdateNode::updateDataNode() const {
  return getInput(updateDataNodeIdx);
}

} // namespace fl
