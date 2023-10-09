/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

/**
 * A stub Tensor backend implementation to make it easy to get started with the
 * Flashlight Tensor API.
 *
 * This stub can be copied, renamed, and implemented as needed.
 */
class StubBackend : public TensorBackend {
 public:
  StubBackend();

  static StubBackend& getInstance();
  ~StubBackend() override = default;
  TensorBackendType backendType() const override;

  // No copy or move construction or assignment
  StubBackend(StubBackend&&) = delete;
  StubBackend(const StubBackend&) = delete;
  StubBackend& operator=(StubBackend&&) = delete;
  StubBackend& operator=(const StubBackend&) = delete;

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
#define FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(TYPE)       \
  Tensor fromScalar(TYPE value, const dtype type) override; \
  Tensor full(const Shape& dims, TYPE value, const dtype type) override;
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const double&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const float&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const int&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const char&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned char&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const long&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const long long&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long long&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const bool&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const short&);
  FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned short&);
#undef FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL

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
#define FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, TYPE) \
  Tensor FUNC(const Tensor& a, TYPE rhs) override;      \
  Tensor FUNC(TYPE lhs, const Tensor& a) override;

#define FL_STUB_BACKEND_BINARY_OP_LITERALS_DECL(FUNC)                   \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const bool&);               \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const int&);                \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned&);           \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const char&);               \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned char&);      \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const long&);               \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned long&);      \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const long long&);          \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned long long&); \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const double&);             \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const float&);              \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const short&);              \
  FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned short&);

#define FL_STUB_BACKEND_BINARY_OP_DECL(FUNC)                  \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) override; \
  FL_STUB_BACKEND_BINARY_OP_LITERALS_DECL(FUNC);

  FL_STUB_BACKEND_BINARY_OP_DECL(add);
  FL_STUB_BACKEND_BINARY_OP_DECL(sub);
  FL_STUB_BACKEND_BINARY_OP_DECL(mul);
  FL_STUB_BACKEND_BINARY_OP_DECL(div);
  FL_STUB_BACKEND_BINARY_OP_DECL(eq);
  FL_STUB_BACKEND_BINARY_OP_DECL(neq);
  FL_STUB_BACKEND_BINARY_OP_DECL(lessThan);
  FL_STUB_BACKEND_BINARY_OP_DECL(lessThanEqual);
  FL_STUB_BACKEND_BINARY_OP_DECL(greaterThan);
  FL_STUB_BACKEND_BINARY_OP_DECL(greaterThanEqual);
  FL_STUB_BACKEND_BINARY_OP_DECL(logicalOr);
  FL_STUB_BACKEND_BINARY_OP_DECL(logicalAnd);
  FL_STUB_BACKEND_BINARY_OP_DECL(mod);
  FL_STUB_BACKEND_BINARY_OP_DECL(bitwiseAnd);
  FL_STUB_BACKEND_BINARY_OP_DECL(bitwiseOr);
  FL_STUB_BACKEND_BINARY_OP_DECL(bitwiseXor);
  FL_STUB_BACKEND_BINARY_OP_DECL(lShift);
  FL_STUB_BACKEND_BINARY_OP_DECL(rShift);
#undef FL_STUB_BACKEND_BINARY_OP_DECL
#undef FL_STUB_BACKEND_BINARY_OP_TYPE_DECL
#undef FL_STUB_BACKEND_BINARY_OP_LITERALS_DECL

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
