/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"
#include "flashlight/fl/tensor/backend/jit/opt/Optimizer.h"

namespace fl {

/**
 * A JIT Tensor backend, mainly responsible for building up the JIT graph for
 * some specific backend.
 */
class JitBackend : public TensorBackend {
  TensorBackend& wrappedBackend_;
  std::function<Tensor(NodePtr)> jitTensorCreator_;
  Evaluator evaluator_;
  Optimizer optimizer_;

  template <typename T>
  Tensor fullWithType(const Shape& shape, T value, dtype type);
  Tensor
  createBinopJitTensor(const Tensor& lhs, const Tensor& rhs, BinaryOp op);

  template <typename T>
  Tensor createScalarTensor(unsigned ndim, T val);

 public:
  JitBackend(
      TensorBackend& wrappedBackend,
      std::function<Tensor(NodePtr)> jitTensorCreator);
  ~JitBackend() override = default;
  TensorBackendType backendType() const override;

  // No copy or move construction or assignment
  JitBackend(JitBackend&&) = delete;
  JitBackend(const JitBackend&) = delete;
  JitBackend& operator=(JitBackend&&) = delete;
  JitBackend& operator=(const JitBackend&) = delete;

  Evaluator& evaluator();
  Optimizer& optimizer();
  TensorBackend& wrappedBackend();

  /* -------------------------- Compute Functions -------------------------- */
  void eval(const Tensor& tensor) override;
  bool supportsDataType(const fl::dtype& dtype) const override;
  // Memory management
  void getMemMgrInfo(const char* msg, const int deviceId, std::ostream* ostream)
      override;
  void setMemMgrLogStream(std::ostream* stream) override;
  void setMemMgrLoggingEnabled(const bool enabled) override;
  void setMemMgrFlushInterval(const size_t interval) override;

  /* -------------------------- Rand Functions -------------------------- */
  void setSeed(const int seed) override;
  Tensor randn(const Shape& shape, dtype type) override;
  Tensor rand(const Shape& shape, dtype type) override;

  /* --------------------------- Tensor Operators --------------------------- */
  /******************** Tensor Creation Functions ********************/
#define FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(TYPE)   \
  Tensor fromScalar(TYPE value, const dtype type) override; \
  Tensor full(const Shape& dims, TYPE value, const dtype type) override;
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const double&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const float&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const int&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const unsigned&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const char&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const unsigned char&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const long&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const unsigned long&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const long long&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const unsigned long long&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const bool&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const short&);
  FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB(const unsigned short&);
#undef FL_JIT_BACKEND_CREATE_FUN_LITERAL_DECL_STUB

  Tensor identity(const Dim dim, const dtype type) override;
  Tensor arange(const Shape& shape, const Dim seqDim, const dtype type)
      override;
  Tensor iota(const Shape& dims, const Shape& tileDims, const dtype type)
      override;

  /************************ Shaping and Indexing *************************/
  Tensor reshape(const Tensor& tensor, const Shape& shape) override;
  Tensor transpose(const Tensor& tensor, const Shape& axes /* = {} */) override;
  Tensor tile(const Tensor& tensor, const Shape& shape) override;
  Tensor concatenate(const std::vector<Tensor>& tensors, const unsigned axis)
      override;
  Tensor nonzero(const Tensor& tensor) override;
  Tensor pad(
      const Tensor& input,
      const std::vector<std::pair<int, int>>& padWidths,
      const PadType type) override;

  /************************** Unary Operators ***************************/
  Tensor exp(const Tensor& tensor) override;
  Tensor log(const Tensor& tensor) override;
  Tensor negative(const Tensor& tensor) override;
  Tensor logicalNot(const Tensor& tensor) override;
  Tensor log1p(const Tensor& tensor) override;
  Tensor sin(const Tensor& tensor) override;
  Tensor cos(const Tensor& tensor) override;
  Tensor sqrt(const Tensor& tensor) override;
  Tensor tanh(const Tensor& tensor) override;
  Tensor floor(const Tensor& tensor) override;
  Tensor ceil(const Tensor& tensor) override;
  Tensor rint(const Tensor& tensor) override;
  Tensor absolute(const Tensor& tensor) override;
  Tensor sigmoid(const Tensor& tensor) override;
  Tensor erf(const Tensor& tensor) override;
  Tensor flip(const Tensor& tensor, const unsigned dim) override;
  Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high)
      override;
  Tensor roll(const Tensor& tensor, const int shift, const unsigned axis)
      override;
  Tensor isnan(const Tensor& tensor) override;
  Tensor isinf(const Tensor& tensor) override;
  Tensor sign(const Tensor& tensor) override;
  Tensor tril(const Tensor& tensor) override;
  Tensor triu(const Tensor& tensor) override;
  Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y)
      override;
  void topk(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned k,
      const Dim axis,
      const SortMode sortMode) override;
  Tensor sort(const Tensor& input, const Dim axis, const SortMode sortMode)
      override;
  void sort(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const Dim axis,
      const SortMode sortMode) override;
  Tensor argsort(const Tensor& input, const Dim axis, const SortMode sortMode)
      override;

  /************************** Binary Operators ***************************/
#define FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, TYPE) \
  Tensor FUNC(const Tensor& a, TYPE rhs) override;          \
  Tensor FUNC(TYPE lhs, const Tensor& a) override;

#define FL_JIT_BACKEND_BINARY_OP_LITERALS_DECL_STUB(FUNC)                   \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const bool&);               \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const int&);                \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const unsigned&);           \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const char&);               \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const unsigned char&);      \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const long&);               \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const unsigned long&);      \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const long long&);          \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const unsigned long long&); \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const double&);             \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const float&);              \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const short&);              \
  FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB(FUNC, const unsigned short&);

#define FL_JIT_BACKEND_BINARY_OP_DECL_STUB(FUNC)              \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) override; \
  FL_JIT_BACKEND_BINARY_OP_LITERALS_DECL_STUB(FUNC);

  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(add);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(sub);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(mul);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(div);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(eq);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(neq);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(lessThan);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(lessThanEqual);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(greaterThan);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(greaterThanEqual);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(logicalOr);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(logicalAnd);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(mod);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(bitwiseAnd);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(bitwiseOr);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(bitwiseXor);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(lShift);
  FL_JIT_BACKEND_BINARY_OP_DECL_STUB(rShift);
#undef FL_JIT_BACKEND_BINARY_OP_DECL_STUB
#undef FL_JIT_BACKEND_BINARY_OP_TYPE_DECL_STUB
#undef FL_JIT_BACKEND_BINARY_OP_LITERALS_DECL_STUB

  Tensor minimum(const Tensor& lhs, const Tensor& rhs) override;
  Tensor maximum(const Tensor& lhs, const Tensor& rhs) override;
  Tensor power(const Tensor& lhs, const Tensor& rhs) override;

  /******************************* BLAS ********************************/
  Tensor matmul(
      const Tensor& lhs,
      const Tensor& rhs,
      MatrixProperty lhsProp,
      MatrixProperty rhsProp) override;

  /************************** Reductions ***************************/
  Tensor amin(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor amax(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  void min(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      const bool keepDims) override;
  void max(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      const bool keepDims) override;
  Tensor sum(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor cumsum(const Tensor& input, const unsigned axis) override;
  Tensor argmax(const Tensor& input, const unsigned axis, const bool keepDims)
      override;
  Tensor argmin(const Tensor& input, const unsigned axis, const bool keepDims)
      override;
  Tensor mean(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor median(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor var(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool bias,
      const bool keepDims) override;
  Tensor std(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor norm(
      const Tensor& input,
      const std::vector<int>& axes,
      double p,
      const bool keepDims) override;
  Tensor countNonzero(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor any(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor all(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
