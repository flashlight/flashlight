/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"

#include <stdexcept>

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"

namespace fl {

namespace {

Dim ceilDiv(Dim numerator, Dim denominator) {
  return (numerator + denominator - 1) / denominator;
}

// TODO refactor with common index logic in OneDnnTensor & ArrayFireTensor;
// maybe put things like this into a tensorBackendUtil file.
Shape inferIndexedShape(const Shape& shape, const std::vector<Index>& indices) {
  std::vector<Dim> indexedDims;
  // TODO check indexing & shape validity, for now we let backend catch it
  for (unsigned i = 0; i < indices.size(); i++) {
    const auto& idx = indices[i];
    switch (idx.type()) {
      case detail::IndexType::Tensor: {
        // TODO behavior of having more indices with the tensor index is a bit
        // subtle, also uncommon. Rather than handling special cases, we don't
        // support it at the moment.
        if (indices.size() != 1) {
          throw std::runtime_error(
              "[inferIndexedShape]: Tensor must appear be the only index");
        }
        const auto& tensorIdx = idx.get<Tensor>();
        // use tensor index's shape as base shape, and then append currently
        // indexed-result shape onto the base shape (ignore the first dimension
        // since it's reduced by tensor index)
        //
        // In FL, the semantics is a bit different from numpy's:
        // 1. if FL tensor index has more than 1 dimension, it's flattened first
        // 2. if FL tensor index has same shape as indexed tensor, even if the
        //    index isn't b8 type, the result shape is the flattened tensor
        //    shape.
        // TODO consider updating FL semantics
        if (shape == tensorIdx.shape()) {
          if (tensorIdx.type() == dtype::b8) {
            throw std::runtime_error(
                "[inferIndexedShape]: Tensor index of type b8 has dynamic sthape");
          }
          return {Dim(tensorIdx.elements())};
        }
        indexedDims.push_back(tensorIdx.elements());
        indexedDims.insert( // ignore first dim, which is reduced
            indexedDims.end(),
            shape.get().begin() + 1,
            shape.get().end());
        return Shape(indexedDims);
      }
      case detail::IndexType::Span: {
        indexedDims.push_back(shape[i]);
        break;
      }
      case detail::IndexType::Range: {
        const auto& rangeIdx = idx.get<range>();
        const auto start = rangeIdx.start();
        const auto end = rangeIdx.end().value_or(shape.dim(i));
        const auto stride = rangeIdx.stride();
        indexedDims.push_back(ceilDiv(end - start, stride));
        break;
      }
      case detail::IndexType::Literal:
        continue; // dimension is reduced
      default:
        throw std::invalid_argument("[inferIndexedShape]: unknown index type.");
    }
  }
  indexedDims.insert( // collect the remaining unindexed dimensions
      indexedDims.end(),
      shape.get().begin() + indices.size(),
      shape.get().end());
  return Shape(indexedDims);
}

std::vector<NodePtr> getIndexNodeInputs(
    NodePtr indexedNode,
    const std::vector<Index>& indices) {
  std::vector<NodePtr> inputs{indexedNode};
  for (const auto& idx : indices) {
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
  return inputs;
}

} // namespace

IndexNodePtr IndexNode::create(
    NodePtr indexedNode,
    const std::vector<Index>& indices) {
  return std::make_shared<IndexNode>(indexedNode, indices, PrivateHelper{});
}

IndexNode::IndexNode(NodePtr indexedNode, const std::vector<Index>& indices, PrivateHelper)
    : NodeTrait(
          getIndexNodeInputs(indexedNode, indices),
          inferIndexedShape(indexedNode->shape(), indices)),
      indices_(indices) {}

NodePtr IndexNode::indexedNode() const {
  return getInput(indexedNodeIdx);
}

const std::vector<Index>& IndexNode::indices() const {
  return indices_;
}

} // namespace fl
