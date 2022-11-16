/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include <sstream>
#include <stdexcept>

namespace fl {

namespace {

/**
 * Rule:
 *    LHS: (r1, ..., rn)
 *    RHS: (l1, ..., ln)
 *  where ri == li, or 1 ∈ (ri, li)
 *  output shape: (max(r1, l1), ..., max(rn, ln))
 * TODO allow different # of dimensions.
 */
std::optional<Shape> getBinopOutputShape(const Shape& lhs, const Shape& rhs) {
  if (lhs.ndim() != rhs.ndim()) {
    return std::nullopt;
  }
  // check and accumulate output dimensions
  auto ndim = lhs.ndim();
  std::vector<Dim> dstDims;
  for (auto i = 0; i < ndim; ++i) {
    auto lhsDim = lhs.dim(i);
    auto rhsDim = rhs.dim(i);
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1) {
      return std::nullopt;
    }
    dstDims.push_back(std::max(lhsDim, rhsDim));
  }
  return Shape(dstDims);
}

} // namespace

BinaryNode::BinaryNode(Node* lhs, Node* rhs, BinaryOp op, const Shape& shape)
    : NodeTrait({lhs, rhs}, shape), op_(op) {}

BinaryNode* BinaryNode::create(Node* lhs, Node* rhs, BinaryOp op) {
  const auto outputShapeOpt = getBinopOutputShape(lhs->shape(), rhs->shape());
  if (!outputShapeOpt.has_value()) {
    std::ostringstream oss;
    oss << "[BinaryNode::create] Invalid shapes: " << lhs->shape() << " and "
        << rhs->shape();
    throw std::runtime_error(oss.str());
  }
  return new BinaryNode(lhs, rhs, op, outputShapeOpt.value());
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
