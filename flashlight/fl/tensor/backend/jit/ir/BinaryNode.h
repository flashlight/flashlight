/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/tensor/backend/jit/ir/Node.h"

namespace fl {

/**
 * Types of binary operations.
 */
enum class BinaryOp {
  Add,
  Sub,
  Mul,
  Div,
  Eq,
  Neq,
  Gt,
  Gte,
  Lt,
  Lte,
  Max,
  Min,
  Pow,
  Mod,
  And,
  Or,
  Shl,
  Shr,
  BitAnd,
  BitOr,
  BitXor,
};

class BinaryNode;
using BinaryNodePtr = std::shared_ptr<BinaryNode>;

/**
 * A node that represents binary operations.
 */
class BinaryNode : public NodeTrait<BinaryNode> {
  const BinaryOp op_;

  // helps indexing into inputs
  static constexpr unsigned kLhsIdx = 0;
  static constexpr unsigned kRhsIdx = 1;

  // help control allocation while allowing `std::make_shared`
  struct PrivateHelper{};

 public:
  static constexpr NodeType nodeType = NodeType::Binary;
  BinaryNode(NodePtr lhs, NodePtr rhs, BinaryOp op, const Shape& shape, PrivateHelper);

  static BinaryNodePtr create(NodePtr lhs, NodePtr rhs, BinaryOp op);

  BinaryOp op() const;
  NodePtr lhs() const;
  NodePtr rhs() const;
};

} // namespace fl
