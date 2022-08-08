/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * Types of binary operations.
 */
enum class BinaryOp { Add, Sub, Mul, Div };

/**
 * A node that represents binary operations.
 */
class BinaryNode : public NodeTrait<BinaryNode> {
  const BinaryOp op_;

  // helps indexing into inputs
  static constexpr unsigned kLhsIdx = 0;
  static constexpr unsigned kRhsIdx = 1;

  // intentionally kept private to control allocation
  BinaryNode(Node* lhs, Node* rhs, BinaryOp op, const Shape& shape);

 public:
  static constexpr NodeType nodeType = NodeType::Binary;

  static BinaryNode* create(Node* lhs, Node* rhs, BinaryOp op);

  BinaryOp op() const;
  Node* lhs() const;
  Node* rhs() const;
};

} // namespace fl
