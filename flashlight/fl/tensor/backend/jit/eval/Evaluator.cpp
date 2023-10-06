/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"

#include <queue>

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

namespace {

// Build a map from each node in the tree to its current refcount.
std::unordered_map<Node*, unsigned> getNodeToRefCountInTree(Node* root) {
  std::unordered_map<Node*, unsigned> nodeToRefCount;
  std::queue<Node*> worklist({root}); // nodes to be visited
  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop();
    if (nodeToRefCount.find(node) == nodeToRefCount.end()) {
      nodeToRefCount.emplace(node, node->getRefCount());
      for (const auto& input : node->inputs()) {
        worklist.push(input);
      }
    }
  }
  return nodeToRefCount;
}

} // namespace

Evaluator::Evaluator(TensorBackend& backend) : backend_(backend) {}

void Evaluator::evalBinaryNode(BinaryNode& node) {
  const auto& lhs = node.lhs()->getResult().value();
  const auto& rhs = node.rhs()->getResult().value();
  node.setResult(evalBinaryOp(node.op(), lhs, rhs));
}

void Evaluator::evalCustomNode(CustomNode& node) {
  std::vector<const Tensor*> inputTensors;
  for (auto& inputNode : node.inputs()) {
    inputTensors.push_back(&inputNode->getResult().value());
  }
  node.setResult(node.evalFunc()(inputTensors));
}

void Evaluator::evalIndexNode(IndexNode& node) {
  const auto& indexedTensor = node.indexedNode()->getResult().value();
  node.setResult(indexedTensor(unwrapTensorInIndices(node.indices())));
}

void Evaluator::evalIndexedUpdateNode(IndexedUpdateNode& node) {
  // TODO no need to copy if indexedNode has only 1 user here
  auto indexedTensor = node.indexedNode()->getResult().value().copy();
  const auto firstUnwrappedIndices =
      unwrapTensorInIndices(node.indexings().front());
  const auto& updateDataTensor = node.updateDataNode()->getResult().value();
  // if we do X = Y, it's a copy instead of update, thus the special case here
  if (node.indexings().size() == 1) {
    indexedTensor(firstUnwrappedIndices) = updateDataTensor;
  } else {
    auto currIndexResult = indexedTensor(firstUnwrappedIndices);
    for (unsigned i = 1; i < node.indexings().size() - 1; i++) {
      const auto unwrappedIndices =
          unwrapTensorInIndices(node.indexings().at(i));
      currIndexResult = currIndexResult(unwrappedIndices);
    }
    const auto lastUnwrappedIndices =
        unwrapTensorInIndices(node.indexings().back());
    currIndexResult(lastUnwrappedIndices) = updateDataTensor;
  }
  node.setResult(std::move(indexedTensor));
}

std::vector<Index> Evaluator::unwrapTensorInIndices(
    const std::vector<Index>& indices) {
  std::vector<Index> unwrappedIndices;
  for (const auto& index : indices) {
    if (index.type() == detail::IndexType::Tensor) {
      const auto tensorIndexNode = toJitTensorBase(index.get<Tensor>()).node();
      unwrappedIndices.push_back(tensorIndexNode->getResult().value());
    } else {
      unwrappedIndices.push_back(index);
    }
  }
  return unwrappedIndices;
}

void Evaluator::evalScalarNode(ScalarNode& node) {
  node.setResult(evalScalar(node));
}

Tensor
Evaluator::evalBinaryOp(BinaryOp op, const Tensor& lhs, const Tensor& rhs) {
  switch (op) {
    case BinaryOp::Add:
      return backend_.add(lhs, rhs);
    case BinaryOp::Sub:
      return backend_.sub(lhs, rhs);
    case BinaryOp::Mul:
      return backend_.mul(lhs, rhs);
    case BinaryOp::Div:
      return backend_.div(lhs, rhs);
    case BinaryOp::Eq:
      return backend_.eq(lhs, rhs);
    case BinaryOp::Neq:
      return backend_.neq(lhs, rhs);
    case BinaryOp::Gt:
      return backend_.greaterThan(lhs, rhs);
    case BinaryOp::Gte:
      return backend_.greaterThanEqual(lhs, rhs);
    case BinaryOp::Lt:
      return backend_.lessThan(lhs, rhs);
    case BinaryOp::Lte:
      return backend_.lessThanEqual(lhs, rhs);
    case BinaryOp::Min:
      return backend_.minimum(lhs, rhs);
    case BinaryOp::Max:
      return backend_.maximum(lhs, rhs);
    case BinaryOp::Pow:
      return backend_.power(lhs, rhs);
    case BinaryOp::Mod:
      return backend_.mod(lhs, rhs);
    case BinaryOp::And:
      return backend_.logicalAnd(lhs, rhs);
    case BinaryOp::Or:
      return backend_.logicalOr(lhs, rhs);
    case BinaryOp::Shl:
      return backend_.lShift(lhs, rhs);
    case BinaryOp::Shr:
      return backend_.rShift(lhs, rhs);
    case BinaryOp::BitAnd:
      return backend_.bitwiseAnd(lhs, rhs);
    case BinaryOp::BitOr:
      return backend_.bitwiseOr(lhs, rhs);
    case BinaryOp::BitXor:
      return backend_.bitwiseXor(lhs, rhs);
  }
  throw std::runtime_error(
      "[Evaluator::evalBinaryOp] Unknown binary operation type");
}

Tensor Evaluator::evalScalar(ScalarNode& node) {
  const Shape& shape = node.shape();
  const auto dtype = node.dataType();
  switch (dtype) {
    case dtype::f16:
    case dtype::f32:
    case dtype::f64:
      return backend_.full(shape, node.scalar<double>(), dtype);
    case dtype::b8:
    case dtype::s16:
    case dtype::s32:
    case dtype::s64:
    case dtype::u8:
    case dtype::u16:
    case dtype::u32:
      return backend_.full(shape, node.scalar<long long>(), dtype);
    case dtype::u64:
      return backend_.full(shape, node.scalar<unsigned long long>(), dtype);
  }
  throw std::runtime_error("Unknown dtype");
}

void Evaluator::evalNodeDispatch(Node* node) {
  switch (node->type()) {
    case NodeType::Binary:
      return evalBinaryNode(node->impl<BinaryNode>());
    case NodeType::Custom:
      return evalCustomNode(node->impl<CustomNode>());
    case NodeType::Index:
      return evalIndexNode(node->impl<IndexNode>());
    case NodeType::IndexedUpdate:
      return evalIndexedUpdateNode(node->impl<IndexedUpdateNode>());
    case NodeType::Scalar:
      return evalScalarNode(node->impl<ScalarNode>());
    case NodeType::Value:
      return; // already has a result
  }
  throw std::runtime_error("[Evaluator::evalNodeDispatch] Unknown node type");
}

void Evaluator::evalNode(Node* node) {
  if (!node->getResult().has_value()) {
    for (const auto& input : node->inputs()) {
      evalNode(input);
    }
    evalNodeDispatch(node);
    for (const auto& input : node->inputs()) {
      auto& count = nodeToResultUseCount_.at(input);
      count--;
      if (count == 0 && !input->isValue()) {
        // This helps reduce memory footprint during evaluation, allowing the
        // result tensor memory to be reused. This has a non-trivial performance
        // impact on graph with high intermediate tensor memory usage.
        input->unsetResult();
      }
    }
  }
}

void Evaluator::eval(Node* node) {
  nodeToResultUseCount_ = getNodeToRefCountInTree(node);
  evalNode(node);
  nodeToResultUseCount_.clear();
}

} // namespace fl
