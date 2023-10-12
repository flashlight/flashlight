/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"

#include <chrono>
#include <functional>
#include <queue>

#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

namespace fl {

namespace {

// Build a map from each node in the tree to the # of its current uses
// (including external use).
std::unordered_map<NodePtr, unsigned> getNodeToUseCountInTree(NodePtr root) {
  auto getUseCount = [](NodePtr node) {
    return node->uses().size() + node->externalUses().size();
  };
  std::unordered_map<NodePtr, unsigned> nodeToUseCount{
      {root, getUseCount(root)}};
  std::queue<NodePtr> worklist({root}); // nodes to be visited
  while (!worklist.empty()) {
    NodePtr node = worklist.front();
    worklist.pop();
    for (const auto& input : node->inputs()) {
      if (nodeToUseCount.find(input) == nodeToUseCount.end()) {
        worklist.push(input);
        nodeToUseCount.emplace(input, getUseCount(input));
      }
    }
  }
  return nodeToUseCount;
}

} // namespace

Evaluator::Evaluator(TensorBackend& backend) : backend_(backend) {}

void Evaluator::profile(std::function<void()> func, NodePtr nodePtr) {
  if (!profilerEnabled_) {
    func();
    return;
  }
  const auto start = std::chrono::high_resolution_clock::now();
  func();
  const auto end = std::chrono::high_resolution_clock::now();
  const auto durNs =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start)
          .count();
  nodeToTotTimeMs_.insert({nodePtr, durNs * 1000});
}

void Evaluator::evalBinaryNode(BinaryNodePtr node) {
  std::function<void()> func = [this, node] {
    const auto& lhs = node->lhs()->getResult().value();
    const auto& rhs = node->rhs()->getResult().value();
    node->setResult(evalBinaryOp(node->op(), lhs, rhs));
  };
  profile(func, node);
}

void Evaluator::evalCustomNode(CustomNodePtr node) {
  std::vector<const Tensor*> inputTensors;
  for (auto& inputNode : node->inputs()) {
    inputTensors.push_back(&inputNode->getResult().value());
  }
  std::function<void()> func = [node, inputTensors = std::move(inputTensors)] {
    node->setResult(node->evalFunc()(inputTensors));
  };
  profile(func, node);
}

void Evaluator::evalIndexNode(IndexNodePtr node) {
  std::function<void()> func = [this, node]() {
    const auto& indexedTensor = node->indexedNode()->getResult().value();
    node->setResult(indexedTensor(unwrapTensorInIndices(node->indices())));
  };
  profile(func, node);
}

void Evaluator::evalIndexedUpdateNode(IndexedUpdateNodePtr node) {
  // TODO no need to copy if indexedNode has only 1 user here
  std::function<void()> func = [this, node]() {
    auto indexedTensor = node->indexedNode()->getResult().value().copy();
    const auto firstUnwrappedIndices =
        unwrapTensorInIndices(node->indexings().front());
    const auto& updateDataTensor = node->updateDataNode()->getResult().value();
    // if we do X = Y, it's a copy instead of update, thus the special case here
    if (node->indexings().size() == 1) {
      indexedTensor(firstUnwrappedIndices) = updateDataTensor;
    } else {
      auto currIndexResult = indexedTensor(firstUnwrappedIndices);
      for (unsigned i = 1; i < node->indexings().size() - 1; i++) {
        const auto unwrappedIndices =
            unwrapTensorInIndices(node->indexings().at(i));
        currIndexResult = currIndexResult(unwrappedIndices);
      }
      const auto lastUnwrappedIndices =
          unwrapTensorInIndices(node->indexings().back());
      currIndexResult(lastUnwrappedIndices) = updateDataTensor;
    }
    node->setResult(std::move(indexedTensor));
  };
  profile(func, node);
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

void Evaluator::evalScalarNode(ScalarNodePtr node) {
  std::function<void()> func = [&]() { node->setResult(evalScalar(node)); };
  profile(func, node);
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

Tensor Evaluator::evalScalar(ScalarNodePtr node) {
  const Shape& shape = node->shape();
  const auto dtype = node->dataType();
  switch (dtype) {
    case dtype::f16:
    case dtype::f32:
    case dtype::f64:
      return backend_.full(shape, node->scalar<double>(), dtype);
    case dtype::b8:
    case dtype::s16:
    case dtype::s32:
    case dtype::s64:
    case dtype::u8:
    case dtype::u16:
    case dtype::u32:
      return backend_.full(shape, node->scalar<long long>(), dtype);
    case dtype::u64:
      return backend_.full(shape, node->scalar<unsigned long long>(), dtype);
  }
  throw std::runtime_error("Unknown dtype");
}

void Evaluator::evalNodeDispatch(NodePtr node) {
  switch (node->type()) {
    case NodeType::Binary:
      return evalBinaryNode(Node::cast<BinaryNodePtr>(node));
    case NodeType::Custom:
      return evalCustomNode(Node::cast<CustomNodePtr>(node));
    case NodeType::Index:
      return evalIndexNode(Node::cast<IndexNodePtr>(node));
    case NodeType::IndexedUpdate:
      return evalIndexedUpdateNode(Node::cast<IndexedUpdateNodePtr>(node));
    case NodeType::Scalar:
      return evalScalarNode(Node::cast<ScalarNodePtr>(node));
    case NodeType::Value:
      return; // already has a result
  }
  throw std::runtime_error("[Evaluator::evalNodeDispatch] Unknown node type");
}

void Evaluator::evalNode(NodePtr node) {
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

void Evaluator::eval(NodePtr node) {
  nodeToResultUseCount_ = getNodeToUseCountInTree(node);
  evalNode(node);
  for (const auto& callback : postEvalCallbacks_) {
    callback(node, nodeToTotTimeMs_);
  }
  nodeToTotTimeMs_.clear();
  nodeToResultUseCount_.clear();
}

void Evaluator::setProfilerState(bool active) {
  this->profilerEnabled_ = active;
}

bool Evaluator::getProfilerState() {
  return this->profilerEnabled_;
}

const std::unordered_map<NodePtr, float>& Evaluator::getProfilerStats() {
  return nodeToTotTimeMs_;
}

void Evaluator::clearProfilerStats() {
  nodeToTotTimeMs_.clear();
}

Evaluator::PostEvalCallbackHandle Evaluator::addPostEvalCallback(
    PostEvalCallback callback) {
  return postEvalCallbacks_.insert(postEvalCallbacks_.end(), callback);
}

void Evaluator::removePostEvalCallback(PostEvalCallbackHandle handle) {
  postEvalCallbacks_.erase(handle);
}

} // namespace fl
