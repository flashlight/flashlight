/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>
#include <unordered_map>

#include "flashlight/fl/tensor/TensorBackend.h"

#include <af/array.h>

namespace fl {

/**
 * A tensor backend implementation of the ArrayFire tensor library.
 *
 * Given that ArrayFire has an internal DeviceManager singleton to manage its
 * global state, nothing is stored here as those internals are opaquely handled.
 * This class simply dispatches operations on global tensor functions to their
 * ArrayFire counterparts.
 */
class ArrayFireBackend : public TensorBackend {
  // TODO: consolidate the ArrayFire memory manager here so its global state can
  // be stored/we can reduce the number of singletons.
  std::once_flag memoryInitFlag;

  // These help ensure we are using native device id in public methods.
  std::unordered_map<int, int> nativeIdToId_;
  std::unordered_map<int, int> idToNativeId_;

  // keep track of the individual active stream on each ArrayFire device
  // NOTE using a `shared_ptr` to allow its capture in setActive callback;
  // see constructor for details.
  std::shared_ptr<std::unordered_map<int, std::shared_ptr<const Stream>>>
      afIdToStream_{std::make_shared<
          std::unordered_map<int, std::shared_ptr<const Stream>>>()};

  // Intentionally private. Only one instance should exist/it should be accessed
  // via getInstance().
  ArrayFireBackend();

 public:
  static ArrayFireBackend& getInstance();
  ~ArrayFireBackend() override = default;
  TensorBackendType backendType() const override;

  // No copy or move construction or assignment
  ArrayFireBackend(ArrayFireBackend&&) = delete;
  ArrayFireBackend(const ArrayFireBackend&) = delete;
  ArrayFireBackend& operator=(ArrayFireBackend&&) = delete;
  ArrayFireBackend& operator=(const ArrayFireBackend&) = delete;

  /* -------------------------- Compute Functions -------------------------- */
  void eval(const Tensor& tensor) override;

  /**
   * Return the stream from which the given array was created.
   *
   * @return an immutable reference to the stream from which `arr` was created.
   */
  const Stream& getStreamOfArray(const af::array& arr);
  bool supportsDataType(const fl::dtype& dtype) const override;
  // Memory management
  void getMemMgrInfo(const char* msg, const int nativeDeviceId, std::ostream* ostream)
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
#define AF_BACKEND_CREATE_FUN_LITERAL_DECL(TYPE)            \
  Tensor fromScalar(TYPE value, const dtype type) override; \
  Tensor full(const Shape& dims, TYPE value, const dtype type) override;
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const double&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const float&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const int&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const char&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned char&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const long&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const long long&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long long&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const bool&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const short&);
  AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned short&);
#undef AF_BACKEND_CREATE_FUN_LITERAL_DECL

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
#define FL_AF_BINARY_OP_TYPE_DECL(FUNC, TYPE)      \
  Tensor FUNC(const Tensor& a, TYPE rhs) override; \
  Tensor FUNC(TYPE lhs, const Tensor& a) override;

#define FL_AF_BINARY_OP_LITERALS_DECL(FUNC)                   \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const bool&);               \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const int&);                \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned&);           \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const char&);               \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned char&);      \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const long&);               \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned long&);      \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const long long&);          \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned long long&); \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const double&);             \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const float&);              \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const short&);              \
  FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned short&);

#define FL_AF_BINARY_OP_DECL(FUNC)                            \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) override; \
  FL_AF_BINARY_OP_LITERALS_DECL(FUNC);

  FL_AF_BINARY_OP_DECL(add);
  FL_AF_BINARY_OP_DECL(sub);
  FL_AF_BINARY_OP_DECL(mul);
  FL_AF_BINARY_OP_DECL(div);
  FL_AF_BINARY_OP_DECL(eq);
  FL_AF_BINARY_OP_DECL(neq);
  FL_AF_BINARY_OP_DECL(lessThan);
  FL_AF_BINARY_OP_DECL(lessThanEqual);
  FL_AF_BINARY_OP_DECL(greaterThan);
  FL_AF_BINARY_OP_DECL(greaterThanEqual);
  FL_AF_BINARY_OP_DECL(logicalOr);
  FL_AF_BINARY_OP_DECL(logicalAnd);
  FL_AF_BINARY_OP_DECL(mod);
  FL_AF_BINARY_OP_DECL(bitwiseAnd);
  FL_AF_BINARY_OP_DECL(bitwiseOr);
  FL_AF_BINARY_OP_DECL(bitwiseXor);
  FL_AF_BINARY_OP_DECL(lShift);
  FL_AF_BINARY_OP_DECL(rShift);
#undef FL_AF_BINARY_OP_DECL
#undef FL_AF_BINARY_OP_TYPE_DECL
#undef FL_AF_BINARY_OP_LITERALS_DECL

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
  Tensor amin(const Tensor& input, const std::vector<int>& axes, const bool keepDims)
      override;
  Tensor amax(const Tensor& input, const std::vector<int>& axes, const bool keepDims)
      override;
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
  Tensor sum(const Tensor& input, const std::vector<int>& axes, const bool keepDims)
      override;
  Tensor cumsum(const Tensor& input, const unsigned axis) override;
  Tensor argmax(const Tensor& input, const unsigned axis, const bool keepDims)
      override;
  Tensor argmin(const Tensor& input, const unsigned axis, const bool keepDims)
      override;
  Tensor mean(const Tensor& input, const std::vector<int>& axes, const bool keepDims)
      override;
  Tensor median(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor var(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool bias,
      const bool keepDims) override;
  Tensor std(const Tensor& input, const std::vector<int>& axes, const bool keepDims)
      override;
  Tensor norm(
      const Tensor& input,
      const std::vector<int>& axes,
      double p,
      const bool keepDims) override;
  Tensor countNonzero(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) override;
  Tensor any(const Tensor& input, const std::vector<int>& axes, const bool keepDims)
      override;
  Tensor all(const Tensor& input, const std::vector<int>& axes, const bool keepDims)
      override;

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
