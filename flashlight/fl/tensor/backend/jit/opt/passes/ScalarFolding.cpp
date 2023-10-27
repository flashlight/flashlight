/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/passes/ScalarFolding.h"

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

#include <cmath>
#include <optional>

namespace fl {

namespace {

template <typename T>
std::optional<T> foldScalars(const T lhs, const T rhs, const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return lhs + rhs;
    case BinaryOp::Sub:
      return lhs - rhs;
    case BinaryOp::Mul:
      return lhs * rhs;
    case BinaryOp::Div:
      return lhs / rhs;
    case BinaryOp::Eq:
      return lhs == rhs;
    case BinaryOp::Neq:
      return lhs != rhs;
    case BinaryOp::Gt:
      return lhs >= rhs;
    case BinaryOp::Gte:
      return lhs >= rhs;
    case BinaryOp::Lt:
      return lhs < rhs;
    case BinaryOp::Lte:
      return lhs <= rhs;
    case BinaryOp::Min:
      return std::min(lhs, rhs);
    case BinaryOp::Max:
      return std::max(lhs, rhs);
    case BinaryOp::Pow:
      return std::pow(lhs, rhs);
    // TODO the following support integral scalars, specialize template to
    // support them if it becomes profitable.
    case BinaryOp::And:
    case BinaryOp::Or:
    case BinaryOp::Mod:
    case BinaryOp::Shl:
    case BinaryOp::Shr:
    case BinaryOp::BitAnd:
    case BinaryOp::BitOr:
    case BinaryOp::BitXor:
      return std::nullopt;
  }
  throw std::runtime_error("[foldScalars] Unknown binary operation type");
}

template <typename T>
std::optional<ScalarNodePtr> foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op,
    const dtype type) {
  T lhsVal = lhs.scalar<T>();
  T rhsVal = rhs.scalar<T>();
  std::optional<T> resVal = foldScalars(lhsVal, rhsVal, op);
  if (resVal.has_value()) {
    return ScalarNode::create(Shape(lhs.shape()), type, resVal.value());
  }
  return std::nullopt;
}

std::optional<ScalarNodePtr> foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op) {
  // TODO shape doesn't matter for scalars, support broadcast and add test
  if (lhs.shape() != rhs.shape() || lhs.dataType() != rhs.dataType()) {
    return std::nullopt;
  }
  const auto type = lhs.dataType();
  switch (type) {
    case dtype::f16:
      return std::nullopt;
    case dtype::f32:
      return foldScalarNodes<float>(lhs, rhs, op, type);
    case dtype::f64:
      return foldScalarNodes<double>(lhs, rhs, op, type);
    case dtype::b8:
      return foldScalarNodes<char>(lhs, rhs, op, type);
    case dtype::s16:
      return foldScalarNodes<short>(lhs, rhs, op, type);
    case dtype::s32:
      return foldScalarNodes<int>(lhs, rhs, op, type);
    case dtype::s64:
      return foldScalarNodes<long long>(lhs, rhs, op, type);
    case dtype::u8:
      return foldScalarNodes<unsigned char>(lhs, rhs, op, type);
    case dtype::u16:
      return foldScalarNodes<unsigned short>(lhs, rhs, op, type);
    case dtype::u32:
      return foldScalarNodes<unsigned int>(lhs, rhs, op, type);
    case dtype::u64:
      return foldScalarNodes<unsigned long long>(lhs, rhs, op, type);
  }
  throw std::runtime_error("[foldScalarNodes] Unknown data type");
}

NodePtr foldScalarsInBinaryNode(BinaryNodePtr node) {
  const auto binop = node->op();
  const auto lhs = node->lhs();
  const auto rhs = node->rhs();
  if (lhs->isScalar() && rhs->isScalar()) {
    const auto& lhsScalar = lhs->impl<ScalarNode>();
    const auto& rhsScalar = rhs->impl<ScalarNode>();
    const auto optFoldedScalarNode =
        foldScalarNodes(lhsScalar, rhsScalar, binop);
    if (optFoldedScalarNode.has_value()) {
      const auto foldedScalarNode = optFoldedScalarNode.value();
      node->replaceAllUsesWith(foldedScalarNode);
      return foldedScalarNode;
    }
  }
  return node;
}

NodePtr foldScalars(NodePtr node) {
  for (const auto& input : node->inputs()) {
    foldScalars(input);
  }
  switch (node->type()) {
    case NodeType::Binary:
      return foldScalarsInBinaryNode(Node::cast<BinaryNodePtr>(node));
    case NodeType::Custom:
    case NodeType::Index:
    case NodeType::IndexedUpdate:
    case NodeType::Scalar:
    case NodeType::Value:
      return node;
  }
  throw std::runtime_error("[foldScalars] Unknown node type");
}

} // namespace

NodePtr ScalarFolding::apply(NodePtr node) {
  return foldScalars(node);
}

} // namespace fl
