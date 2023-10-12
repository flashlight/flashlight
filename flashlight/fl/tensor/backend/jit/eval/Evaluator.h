/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

namespace fl {

/**
 * A JIT tree evaluator. It dispatches to another Tensor Backend for carrying
 * out the computation represented by the JIT tree.
 */
class Evaluator {
 public:
  // takes in the evaluated node and execution stats (empty if profiling is
  // disabled)
  using PostEvalCallback =
      std::function<void(NodePtr, std::unordered_map<NodePtr, float>)>;
  using PostEvalCallbackList = std::list<PostEvalCallback>;
  using PostEvalCallbackHandle = PostEvalCallbackList::iterator;

 private:
  // backend used for dispatching Tensor ops.
  TensorBackend& backend_;
  // track (conservatively) how many more times the a node's result will be used
  std::unordered_map<NodePtr, unsigned> nodeToResultUseCount_{};
  // track time spent on executing a node alone (not its inputs)
  std::unordered_map<NodePtr, float> nodeToTotTimeMs_{};
  bool profilerEnabled_{false};
  PostEvalCallbackList postEvalCallbacks_;

  void evalNode(NodePtr node);
  void evalNodeDispatch(NodePtr node);
  // profile execution time of `func` and associate it with `nodePtr`
  void profile(std::function<void()> func, NodePtr nodePtr);

  // evaluate and set result without checking for existing result
  // ASSUME inputs have been evaluated
  void evalBinaryNode(BinaryNodePtr node);
  void evalCustomNode(CustomNodePtr node);
  void evalIndexNode(IndexNodePtr node);
  void evalIndexedUpdateNode(IndexedUpdateNodePtr node);
  // JitTensor in indices becomes the backing tensor
  std::vector<Index> unwrapTensorInIndices(const std::vector<Index>& indices);
  void evalScalarNode(ScalarNodePtr node);

  // helpers that evaluates without setting results
  Tensor evalBinaryOp(BinaryOp op, const Tensor& lhs, const Tensor& rhs);
  Tensor evalScalar(ScalarNodePtr node);

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
   * 2. set result for intermediate nodes if they have external uses
   */
  void eval(NodePtr node);

  /**
   * TODO document
   */
  void setProfilerState(bool active);
  bool getProfilerState();
  const std::unordered_map<NodePtr, float>& getProfilerStats();
  void clearProfilerStats();

  PostEvalCallbackHandle addPostEvalCallback(PostEvalCallback callback);
  void removePostEvalCallback(PostEvalCallbackHandle handle);
};

} // namespace fl
