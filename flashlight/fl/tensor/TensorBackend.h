/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/TensorExtension.h"

namespace fl {

class Stream;

/**
 * A Tensor backend that can be used to store global state associated with a
 * particular tensor implementation.
 *
 * This abstraction facilitates adherence to the implementation requirements for
 * global operators that operate on tensors (e.g. those functions that are not
 * members of `fl::Tensor`).
 *
 * Flashlight Tensors dispatch to their corresponding backends using
 * fl::Tensor::backend() --> typeToBackend (see below) to grab the correct
 * instance.
 */
class TensorBackend {
 public:
  TensorBackend() = default;
  virtual ~TensorBackend() = default;
  virtual TensorBackendType backendType() const = 0;

  /* -------------------------- Compute Functions -------------------------- */
  virtual void eval(const Tensor& tensor) = 0;
  virtual bool supportsDataType(const fl::dtype& dtype) const = 0;
  // Memory Management
  virtual void
  getMemMgrInfo(const char* msg, const int deviceId, std::ostream* ostream) = 0;
  virtual void setMemMgrLogStream(std::ostream* stream) = 0;
  virtual void setMemMgrLoggingEnabled(const bool enabled) = 0;
  virtual void setMemMgrFlushInterval(const size_t interval) = 0;

  /* -------------------------- Rand Functions -------------------------- */
  virtual void setSeed(const int seed) = 0;
  virtual Tensor randn(const Shape& shape, dtype type) = 0;
  virtual Tensor rand(const Shape& shape, dtype type) = 0;

  /* --------------------------- Tensor Operators ---------------------------
   * For operator documentation and expected behavior, see TensorBase.h.
   */
  /******************** Tensor Creation Functions ********************/
#define FL_CREATE_FUN_LITERAL_BACKEND_DECL(TYPE)               \
  virtual Tensor fromScalar(TYPE value, const dtype type) = 0; \
  virtual Tensor full(const Shape& dims, TYPE value, const dtype type) = 0;
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const double&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const float&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const int&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const char&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned char&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const long&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned long&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const long long&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned long long&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const bool&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const short&);
  FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned short&);
#undef FL_CREATE_FUN_LITERAL_BACKEND_DECL

  virtual Tensor identity(const Dim dim, const dtype type) = 0;
  virtual Tensor
  arange(const Shape& shape, const Dim seqDim, const dtype type) = 0;
  virtual Tensor
  iota(const Shape& dims, const Shape& tileDims, const dtype type) = 0;

  /************************ Shaping and Indexing *************************/
  virtual Tensor reshape(const Tensor& tensor, const Shape& shape) = 0;
  virtual Tensor transpose(
      const Tensor& tensor,
      const Shape& axes /* = {} */) = 0;
  virtual Tensor tile(const Tensor& tensor, const Shape& shape) = 0;
  virtual Tensor concatenate(
      const std::vector<Tensor>& tensors,
      const unsigned axis) = 0;
  virtual Tensor nonzero(const Tensor& tensor) = 0;
  virtual Tensor pad(
      const Tensor& input,
      const std::vector<std::pair<int, int>>& padWidths,
      const PadType type) = 0;

  /************************** Unary Operators ***************************/
  virtual Tensor exp(const Tensor& tensor) = 0;
  virtual Tensor log(const Tensor& tensor) = 0;
  virtual Tensor negative(const Tensor& tensor) = 0;
  virtual Tensor logicalNot(const Tensor& tensor) = 0;
  virtual Tensor log1p(const Tensor& tensor) = 0;
  virtual Tensor sin(const Tensor& tensor) = 0;
  virtual Tensor cos(const Tensor& tensor) = 0;
  virtual Tensor sqrt(const Tensor& tensor) = 0;
  virtual Tensor tanh(const Tensor& tensor) = 0;
  virtual Tensor floor(const Tensor& tensor) = 0;
  virtual Tensor ceil(const Tensor& tensor) = 0;
  virtual Tensor rint(const Tensor& tensor) = 0;
  virtual Tensor absolute(const Tensor& tensor) = 0;
  virtual Tensor sigmoid(const Tensor& tensor) = 0;
  virtual Tensor erf(const Tensor& tensor) = 0;
  virtual Tensor flip(const Tensor& tensor, const unsigned dim) = 0;
  virtual Tensor
  clip(const Tensor& tensor, const Tensor& low, const Tensor& high) = 0;
  virtual Tensor
  clip(const Tensor& tensor, const Tensor& low, const double& high);
  virtual Tensor
  clip(const Tensor& tensor, const double& low, const Tensor& high);
  virtual Tensor
  clip(const Tensor& tensor, const double& low, const double& high);
  virtual Tensor
  roll(const Tensor& tensor, const int shift, const unsigned axis) = 0;
  virtual Tensor isnan(const Tensor& tensor) = 0;
  virtual Tensor isinf(const Tensor& tensor) = 0;
  virtual Tensor sign(const Tensor& tensor) = 0;
  virtual Tensor tril(const Tensor& tensor) = 0;
  virtual Tensor triu(const Tensor& tensor) = 0;
  virtual Tensor
  where(const Tensor& condition, const Tensor& x, const Tensor& y) = 0;
  virtual Tensor
  where(const Tensor& condition, const Tensor& x, const double& y);
  virtual Tensor
  where(const Tensor& condition, const double& x, const Tensor& y);
  virtual void topk(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned k,
      const Dim axis,
      const SortMode sortMode) = 0;
  virtual Tensor
  sort(const Tensor& input, const Dim axis, const SortMode sortMode) = 0;
  virtual void sort(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const Dim axis,
      const SortMode sortMode) = 0;
  virtual Tensor
  argsort(const Tensor& input, const Dim axis, const SortMode sortMode) = 0;

  /************************** Binary Operators ***************************/
#define FL_BINARY_OP_TYPE_DECL(FUNC, TYPE)            \
  virtual Tensor FUNC(const Tensor& a, TYPE rhs) = 0; \
  virtual Tensor FUNC(TYPE lhs, const Tensor& a) = 0;

#define FL_BINARY_OP_LITERALS_DECL(FUNC)                   \
  FL_BINARY_OP_TYPE_DECL(FUNC, const bool&);               \
  FL_BINARY_OP_TYPE_DECL(FUNC, const int&);                \
  FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned&);           \
  FL_BINARY_OP_TYPE_DECL(FUNC, const char&);               \
  FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned char&);      \
  FL_BINARY_OP_TYPE_DECL(FUNC, const long&);               \
  FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned long&);      \
  FL_BINARY_OP_TYPE_DECL(FUNC, const long long&);          \
  FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned long long&); \
  FL_BINARY_OP_TYPE_DECL(FUNC, const double&);             \
  FL_BINARY_OP_TYPE_DECL(FUNC, const float&);              \
  FL_BINARY_OP_TYPE_DECL(FUNC, const short&);              \
  FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned short&);

#define FL_BINARY_OP_DECL(FUNC)                                  \
  virtual Tensor FUNC(const Tensor& lhs, const Tensor& rhs) = 0; \
  FL_BINARY_OP_LITERALS_DECL(FUNC);

  FL_BINARY_OP_DECL(add);
  FL_BINARY_OP_DECL(sub);
  FL_BINARY_OP_DECL(mul);
  FL_BINARY_OP_DECL(div);
  FL_BINARY_OP_DECL(eq);
  FL_BINARY_OP_DECL(neq);
  FL_BINARY_OP_DECL(lessThan);
  FL_BINARY_OP_DECL(lessThanEqual);
  FL_BINARY_OP_DECL(greaterThan);
  FL_BINARY_OP_DECL(greaterThanEqual);
  FL_BINARY_OP_DECL(logicalOr);
  FL_BINARY_OP_DECL(logicalAnd);
  FL_BINARY_OP_DECL(mod);
  FL_BINARY_OP_DECL(bitwiseAnd);
  FL_BINARY_OP_DECL(bitwiseOr);
  FL_BINARY_OP_DECL(bitwiseXor);
  FL_BINARY_OP_DECL(lShift);
  FL_BINARY_OP_DECL(rShift);
#undef FL_BINARY_OP_DECL
#undef FL_BINARY_OP_TYPE_DECL
#undef FL_BINARY_OP_LITERALS_DECL

  virtual Tensor minimum(const Tensor& lhs, const Tensor& rhs) = 0;
  virtual Tensor minimum(const Tensor& lhs, const double& rhs);
  virtual Tensor minimum(const double& lhs, const Tensor& rhs);
  virtual Tensor maximum(const Tensor& lhs, const Tensor& rhs) = 0;
  virtual Tensor maximum(const Tensor& lhs, const double& rhs);
  virtual Tensor maximum(const double& lhs, const Tensor& rhs);
  virtual Tensor power(const Tensor& lhs, const Tensor& rhs) = 0;
  virtual Tensor power(const Tensor& lhs, const double& rhs);
  virtual Tensor power(const double& lhs, const Tensor& rhs);

  /******************************* BLAS ********************************/
  virtual Tensor matmul(
      const Tensor& lhs,
      const Tensor& rhs,
      MatrixProperty lhsProp,
      MatrixProperty rhsProp) = 0;

  /************************** Reductions ***************************/
  virtual Tensor
  amin(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;
  virtual Tensor
  amax(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;
  virtual void min(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      const bool keepDims) = 0;
  virtual void max(
      Tensor& values,
      Tensor& indices,
      const Tensor& input,
      const unsigned axis,
      const bool keepDims) = 0;
  virtual Tensor
  sum(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;
  virtual Tensor cumsum(const Tensor& input, const unsigned axis) = 0;
  virtual Tensor
  argmax(const Tensor& input, const unsigned axis, const bool keepDims) = 0;
  virtual Tensor
  argmin(const Tensor& input, const unsigned axis, const bool keepDims) = 0;
  virtual Tensor
  mean(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;
  virtual Tensor
  median(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;
  virtual Tensor var(
      const Tensor& input,
      const std::vector<int>& axes,
      bool bias,
      const bool keepDims) = 0;
  virtual Tensor
  std(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;
  virtual Tensor norm(
      const Tensor& input,
      const std::vector<int>& axes,
      double p,
      const bool keepDims) = 0;
  virtual Tensor countNonzero(
      const Tensor& input,
      const std::vector<int>& axes,
      const bool keepDims) = 0;
  virtual Tensor
  any(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;
  virtual Tensor
  all(const Tensor& input, const std::vector<int>& axes, const bool keepDims) = 0;

  /************************** Utils ***************************/
  virtual void print(const Tensor& tensor) = 0;

  /**
   * Checks if a datatype is supported by a TensorBackend and its registered
   * extensions.
   *
   * @param[in] dtype the datatype to check
   *
   * @return true if the data type is supported, false otherwise
   */
  virtual bool isDataTypeSupported(const fl::dtype& dtype) const final;

  /********************* Tensor Extensions **********************/
  template <typename T>
  T& getExtension() {
    static_assert(
        std::is_base_of<TensorExtensionBase, T>::value,
        "TensorBackend::getExtension<T>() called with type T "
        "that is not derived from TensorExtensionBase.");

    TensorExtensionType e = T::getExtensionType();

    // If an extension isn't present, instantiate it via its registered
    // creation function - only do this once per extension.
    if (extensions_.find(e) == extensions_.end()) {
      auto& creationFunc =
          detail::TensorExtensionRegistrar::getInstance()
              .getTensorExtensionCreationFunc(this->backendType(), e);
      extensions_.emplace(e, creationFunc());
    }
    return *(static_cast<T*>(extensions_.at(e).get()));
  }

 protected:
  std::unordered_map<TensorExtensionType, std::unique_ptr<TensorExtensionBase>>
      extensions_;
};

/**
 * Convert a Tensor from one backend to another.
 *
 * The resulting tensor will have the same shape, type, and contents.
 *
 * @param[in] in a tensor rvalue reference
 * @return a tensor with backend type specified by the template
 */
template <typename T>
Tensor toTensorType(Tensor&& in) {
  static_assert(
      std::is_base_of<TensorAdapterBase, T>::value,
      "toTensorType: T must be a derived type of TensorAdapterBase");
  // Fast path - backend is the same
  // TODO: make fl::TensorBackendType a static constexpr on the class as well so
  // as to not need to instantiate a backend to check the type
  if (in.backendType() == T().backendType()) {
    return std::move(in);
  }

  // As per impl requirements, Tensor::device() should return a pointer to host
  // memory if the tensor resides on the host.
  return Tensor(std::make_unique<T>(
      in.shape(),
      in.type(),
      // TODO: use the void specialization instead of a reinterpret cast
      reinterpret_cast<void*>(in.device<char>()), // expects contiguous memory
      in.location()));
}

namespace detail {

/**
 * Compare the backends of two tensors.
 *
 * @return true if the backends of both tensors are the same, else false.
 */
bool areBackendsEqual(const Tensor& a, const Tensor& b);

/**
 * Compare the backends of multiple tensors.
 *
 * @return true if all tensors' backends are the same, false otherwise.
 */
template <typename... Args>
bool areBackendsEqual(const Tensor& a, const Tensor& b, const Args&... args) {
  return areBackendsEqual(a, b) && areBackendsEqual(a, args...) &&
      areBackendsEqual(b, args...);
}

/**
 *
 * @return a reference to a tensor backend instance descripting the default
 backend.
 */
TensorBackend& getDefaultBackend();

} // namespace detail
} // namespace fl
