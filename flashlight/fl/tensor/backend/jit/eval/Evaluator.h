/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

/**
 * A JIT tree evaluator. It dispatches to another Tensor Backend for carrying
 * out the computation represented by the JIT tree.
 */
class Evaluator {
  // backend used for dispatching Tensor ops.
  TensorBackend& backend_;
  // track (conservatively) how many more times the a node's result will be used
  std::unordered_map<Node*, unsigned> nodeToResultUseCount_{};

  void evalNode(Node* node);
  void evalNodeDispatch(Node* node);

  // evaluate and set result without checking for existing result
  // ASSUME inputs have been evaluated
  void evalBinaryNode(BinaryNode& node);
  void evalCustomNode(CustomNode& node);
  void evalIndexNode(IndexNode& node);
  void evalScalarNode(ScalarNode& node);

  // helpers that evaluates without setting results
  Tensor evalBinaryOp(BinaryOp op, const Tensor& lhs, const Tensor& rhs);
  Tensor evalScalar(ScalarNode& node);

 public:
  /**
   * Creates a JIT graph Evaluator that dispatches to the given backend.
   */
  explicit Evaluator(TensorBackend& backend);

  // no copy/move
  Evaluator(const Evaluator&) = delete;
  Evaluator(Evaluator&&) = delete;
  Evaluator& operator=(const Evaluator&) = delete;
  Evaluator& operator=(const Evaluator&&) = delete;

  /**
   * Execute the entire computation tree rooted at `node`.
   * 1. no op if result already set
   * 2. set result for all intermediate/final tensors evaluated
   */
  void eval(Node* node);
};

} // namespace fl
