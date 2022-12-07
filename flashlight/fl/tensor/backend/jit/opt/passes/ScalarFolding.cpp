/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/opt/passes/ScalarFolding.h"

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

namespace {

template <typename T>
T foldScalars(const T lhs, const T rhs, const BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return lhs + rhs;
    case BinaryOp::Sub:
      return lhs - rhs;
    case BinaryOp::Mul:
      return lhs * rhs;
    case BinaryOp::Div:
      return lhs / rhs;
  }
  throw std::runtime_error("[foldScalars] Unknown binary operation type");
}

template <typename T>
ScalarNode* foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op,
    const dtype type) {
  T lhsVal = lhs.scalar<T>();
  T rhsVal = rhs.scalar<T>();
  T resVal = foldScalars(lhsVal, rhsVal, op);
  return ScalarNode::create(Shape(lhs.shape()), type, resVal);
}

ScalarNode* foldScalarNodes(
    const ScalarNode& lhs,
    const ScalarNode& rhs,
    const BinaryOp op,
    const dtype type) {
  switch (type) {
    case dtype::f16:
      throw std::runtime_error("[foldScalarNodes] unexpected f16");
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

Node* foldScalarsInBinaryNode(BinaryNode* node) {
  const auto binop = node->op();
  const auto lhs = node->lhs();
  const auto rhs = node->rhs();
  if (lhs->isScalar() && rhs->isScalar()) {
    const auto& lhsScalar = lhs->impl<ScalarNode>();
    const auto& rhsScalar = rhs->impl<ScalarNode>();
    const auto& shape = lhsScalar.shape();
    const auto dtype = lhsScalar.dataType();
    if (shape == rhsScalar.shape() && dtype == rhsScalar.dataType() &&
        dtype != dtype::f16) {
      auto foldedScalar = foldScalarNodes(lhsScalar, rhsScalar, binop, dtype);
      node->replaceAllUsesWith(foldedScalar);
      return foldedScalar;
    }
  }
  return node;
}

Node* foldScalars(Node* node) {
  for (const auto& input : node->inputs()) {
    foldScalars(input);
  }
  switch (node->type()) {
    case NodeType::Binary:
      return foldScalarsInBinaryNode(&node->impl<BinaryNode>());
    case NodeType::Custom:
    case NodeType::Index:
    case NodeType::Scalar:
    case NodeType::Value:
      return node;
  }
  throw std::runtime_error("[foldScalars] Unknown node type");
}

} // namespace

Node* ScalarFolding::apply(Node* node) {
  return foldScalars(node);
}

} // namespace fl
