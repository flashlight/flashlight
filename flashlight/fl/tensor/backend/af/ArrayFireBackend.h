/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBackend.h"

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

  // Intentionally private. Only one instance should exist/it should be accessed
  // via getInstance().
  ArrayFireBackend();

 public:
  static ArrayFireBackend& getInstance();

  ~ArrayFireBackend() override = default;

  // No copy or move construction or assignment
  ArrayFireBackend(ArrayFireBackend&&) = delete;
  ArrayFireBackend(const ArrayFireBackend&) = delete;
  ArrayFireBackend& operator=(ArrayFireBackend&&) = delete;
  ArrayFireBackend& operator=(const ArrayFireBackend&) = delete;

  /* -------------------------- Compute Functions -------------------------- */
  void sync() override;
  void sync(int deviceId) override;
  void eval(const Tensor& tensor) override;
  int getDevice() override;
  void setDevice(int deviceId) override;

  /* -------------------------- Rand Functions -------------------------- */
  void setSeed(int seed) override;
  Tensor randn(const Shape& shape, dtype type) override;
  Tensor rand(const Shape& shape, dtype type) override;

  /* --------------------------- Tensor Operators --------------------------- */
  /******************** Tensor Creation Functions ********************/
#define FL_FULL_FUN_BACKEND_DEF(TYPE) \
  Tensor full(const Shape& dims, TYPE value, const dtype type) override;
  FL_FULL_FUN_BACKEND_DEF(const double&);
  FL_FULL_FUN_BACKEND_DEF(const float&);
  FL_FULL_FUN_BACKEND_DEF(const int&);
  FL_FULL_FUN_BACKEND_DEF(const unsigned&);
  FL_FULL_FUN_BACKEND_DEF(const char&);
  FL_FULL_FUN_BACKEND_DEF(const unsigned char&);
  FL_FULL_FUN_BACKEND_DEF(const long&);
  FL_FULL_FUN_BACKEND_DEF(const unsigned long&);
  FL_FULL_FUN_BACKEND_DEF(const long long&);
  FL_FULL_FUN_BACKEND_DEF(const unsigned long long&);
  FL_FULL_FUN_BACKEND_DEF(const bool&);
  FL_FULL_FUN_BACKEND_DEF(const short&);
  FL_FULL_FUN_BACKEND_DEF(const unsigned short&);
#undef FL_FULL_FUN_BACKEND_DEF

  Tensor identity(const Dim dim, const dtype type) override;
  Tensor arange(const Shape& shape, const Dim seqDim, const dtype type)
      override;
  Tensor iota(const Shape& dims, const Shape& tileDims, const dtype type)
      override;

  /************************ Shaping and Indexing *************************/
  Tensor reshape(const Tensor& tensor, const Shape& shape) override;
  Tensor transpose(const Tensor& tensor, const Shape& axes /* = {} */) override;
  Tensor tile(const Tensor& tensor, const Shape& shape) override;
  Tensor concatenate(const std::vector<Tensor>& tensors, unsigned axis)
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
  Tensor absolute(const Tensor& tensor) override;
  Tensor sigmoid(const Tensor& tensor) override;
  Tensor erf(const Tensor& tensor) override;
  Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high)
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
  Tensor amin(const Tensor& input, const std::vector<int>& axes, bool keepDims)
      override;
  double amin(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor amax(const Tensor& input, const std::vector<int>& axes, bool keepDims)
      override;
  double amax(const Tensor& input) override; // TODO: consolidate w/ above
  void min(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      bool keepDims) override;
  void max(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      bool keepDims) override;
  Tensor sum(const Tensor& input, const std::vector<int>& axes, bool keepDims)
      override;
  double sum(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor argmax(const Tensor& input, unsigned axis, bool keepDims) override;
  Tensor argmin(const Tensor& input, unsigned axis, bool keepDims) override;
  Tensor mean(const Tensor& input, const std::vector<int>& axes, bool keepDims)
      override;
  double mean(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor median(
      const Tensor& input,
      const std::vector<int>& axes,
      bool keepDims) override;
  double median(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor var(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool bias,
      bool keepDims) override;
  double var(const Tensor& input, const bool bias)
      override; // TODO: consolidate w/ above
  Tensor std(const Tensor& input, const std::vector<int>& axes, bool keepDims)
      override;
  double norm(const Tensor& input) override;
  Tensor countNonzero(
      const Tensor& input,
      const std::vector<int>& axes,
      bool keepDims) override;
  Tensor any(const Tensor& input, const std::vector<int>& axes, bool keepDims)
      override;
  bool any(const Tensor& input) override;
  Tensor all(const Tensor& input, const std::vector<int>& axes, bool keepDims)
      override;
  bool all(const Tensor& input) override;

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
