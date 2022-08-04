/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"

namespace fl {

BinaryNode::BinaryNode(Node* lhs, Node* rhs, BinaryOp op)
    : NodeTrait({lhs, rhs}), op_(op) {}

BinaryNode* BinaryNode::create(Node* lhs, Node* rhs, BinaryOp op) {
  return new BinaryNode(lhs, rhs, op);
}

BinaryOp BinaryNode::op() const {
  return op_;
}

Node* BinaryNode::lhs() const {
  return getInput(kLhsIdx);
}

Node* BinaryNode::rhs() const {
  return getInput(kRhsIdx);
}

} // namespace fl
