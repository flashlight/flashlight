/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"

#include <memory>
#include <optional>

#include "flashlight/fl/tensor/backend/onednn/OneDnnCPUStream.h"

#if FL_USE_MKL_RNG
  #include <mkl_vsl.h>
#else
  #include <random>
#endif // FL_USE_MKL_RNG

namespace fl {

/**
 * A tensor backend implementation using the OneDNN library.
 */
class OneDnnBackend : public TensorBackend {
  dnnl::engine engine_;
  std::shared_ptr<OneDnnCPUStream> stream_;
#if FL_USE_MKL_RNG
  VSLStreamStatePtr randStream_;
#else
  using RandEngineType = std::mt19937;
  RandEngineType randEngine_;
#endif // FL_USE_MKL_RNG

  // Apply the given OneDNN binary operation to the tensors
  Tensor applyBinop(
      const Tensor& lhs,
      const Tensor& rhs,
      dnnl::algorithm alg,
      std::optional<dnnl::memory::data_type> dstType = std::nullopt);

  // Apply the given OneDNN element-wise operation to the tensor
  Tensor applyEltwiseOp(
      const Tensor& tensor,
      const dnnl::algorithm alg,
      float alpha = 0,
      float beta = 0);

  // Apply the given OneDNN reduction operation to the tensor
  Tensor applyReductionOp(
      const Tensor& tensor,
      const dnnl::algorithm alg,
      const std::vector<int>& axes,
      const bool keepDims);

  Tensor randnCpu(const Shape& shape, dtype type);
  Tensor randCpu(const Shape& shape, dtype type);

  template <typename T, typename V>
  Tensor fullWithType(const Shape& shape, V value, const dtype type);

 public:
  OneDnnBackend();

  static OneDnnBackend& getInstance();
  ~OneDnnBackend() override = default;
  TensorBackendType backendType() const override;

  // No copy or move construction or assignment
  OneDnnBackend(OneDnnBackend&&) = delete;
  OneDnnBackend(const OneDnnBackend&) = delete;
  OneDnnBackend& operator=(OneDnnBackend&&) = delete;
  OneDnnBackend& operator=(const OneDnnBackend&) = delete;

  /**
   * Gets the active OneDNN stream.
   *
   * @return the active OneDNN stream.
   */
  const Stream& stream() const;

  /**
   * Gets the active native OneDNN stream.
   *
   * @return the active native OneDNN stream.
   */
  dnnl::stream& nativeStream() const;

  /**
   * Gets the active OneDNN engine.
   *
   * @return the active OneDNN engine.
   */
  const dnnl::engine& engine() const;

  /**
   * Gets the OneDNN CPU engine.
   *
   * @return the OneDNN CPU engine.
   */
  const dnnl::engine& cpuEngine() const;

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
#define FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(TYPE)     \
  Tensor fromScalar(TYPE value, const dtype type) override; \
  Tensor full(const Shape& dims, TYPE value, const dtype type) override;
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const double&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const float&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const int&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const char&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned char&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const long&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const long long&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long long&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const bool&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const short&);
  FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned short&);
#undef FL_ONEDNN_BACKEND_CREATE_FUN_LITERAL_DECL

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
  Tensor clip(const Tensor& tensor, const double& low, const double& high)
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
#define FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, TYPE) \
  Tensor FUNC(const Tensor& a, TYPE rhs) override;        \
  Tensor FUNC(TYPE lhs, const Tensor& a) override;

#define FL_ONEDNN_BACKEND_BINARY_OP_LITERALS_DECL(FUNC)                   \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const bool&);               \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const int&);                \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned&);           \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const char&);               \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned char&);      \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const long&);               \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned long&);      \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const long long&);          \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned long long&); \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const double&);             \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const float&);              \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const short&);              \
  FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned short&);

#define FL_ONEDNN_BACKEND_BINARY_OP_DECL(FUNC)                \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) override; \
  FL_ONEDNN_BACKEND_BINARY_OP_LITERALS_DECL(FUNC);

  FL_ONEDNN_BACKEND_BINARY_OP_DECL(add);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(sub);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(mul);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(div);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(eq);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(neq);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(lessThan);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(lessThanEqual);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(greaterThan);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(greaterThanEqual);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(logicalOr);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(logicalAnd);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(mod);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(bitwiseAnd);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(bitwiseOr);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(bitwiseXor);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(lShift);
  FL_ONEDNN_BACKEND_BINARY_OP_DECL(rShift);
#undef FL_ONEDNN_BACKEND_BINARY_OP_DECL
#undef FL_ONEDNN_BACKEND_BINARY_OP_TYPE_DECL
#undef FL_ONEDNN_BACKEND_BINARY_OP_LITERALS_DECL

  Tensor minimum(const Tensor& lhs, const Tensor& rhs) override;
  Tensor minimum(const double& lhs, const Tensor& rhs) override;
  Tensor minimum(const Tensor& lhs, const double& rhs) override;
  Tensor maximum(const Tensor& lhs, const Tensor& rhs) override;
  Tensor maximum(const Tensor& lhs, const double& rhs) override;
  Tensor maximum(const double& lhs, const Tensor& rhs) override;
  Tensor power(const Tensor& lhs, const Tensor& rhs) override;
  Tensor power(const Tensor& lhs, const double& rhs) override;

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
