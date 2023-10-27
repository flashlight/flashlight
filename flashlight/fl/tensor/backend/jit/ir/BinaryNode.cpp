/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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
 *    RHS: (l1, ..., lm)
 *  where
 *    n < m (WLoG)
 *    ∀ i <= n, ri == li or 1 ∈ (ri, li)
 *  output shape: (max(r1, l1), ..., max(rn, ln), ..., lm)
 */
std::optional<Shape> getBinopOutputShape(const Shape& lhs, const Shape& rhs) {
  // check and accumulate output dimensions
  const auto lhsRank = lhs.ndim();
  const auto rhsRank = rhs.ndim();
  const auto maxRank = std::max(lhsRank, rhsRank);
  std::vector<Dim> dstDims;
  for (auto i = 0; i < maxRank; ++i) {
    // if one side ran out, fill it with 1, so it'll broadcast to the other side
    const auto lhsDim = i < lhsRank ? lhs.dim(i) : 1;
    const auto rhsDim = i < rhsRank ? rhs.dim(i) : 1;
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1) {
      return std::nullopt;
    }
    dstDims.push_back(std::max(lhsDim, rhsDim));
  }
  return Shape(dstDims);
}

} // namespace

BinaryNode::BinaryNode(NodePtr lhs, NodePtr rhs, BinaryOp op, const Shape& shape, PrivateHelper)
    : NodeTrait({lhs, rhs}, shape), op_(op) {}

BinaryNodePtr BinaryNode::create(NodePtr lhs, NodePtr rhs, BinaryOp op) {
  const auto outputShapeOpt = getBinopOutputShape(lhs->shape(), rhs->shape());
  if (!outputShapeOpt.has_value()) {
    std::ostringstream oss;
    oss << "[BinaryNode::create] Invalid shapes: " << lhs->shape() << " and "
        << rhs->shape();
    throw std::invalid_argument(oss.str());
  }
  return std::make_shared<BinaryNode>(lhs, rhs, op, outputShapeOpt.value(), PrivateHelper{});
}

BinaryOp BinaryNode::op() const {
  return op_;
}

NodePtr BinaryNode::lhs() const {
  return getInput(kLhsIdx);
}

NodePtr BinaryNode::rhs() const {
  return getInput(kRhsIdx);
}

} // namespace fl
