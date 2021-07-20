/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

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

  /* --------------------------- Tensor Operators ---------------------------
   * For operator documentation and expected behavior, see TensorBase.h.
   */
  /******************** Tensor Creation Functions ********************/
#define FL_FULL_FUN_BACKEND_DEF(TYPE) \
  virtual Tensor full(const Shape& dims, TYPE value, const dtype type) = 0;
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

  virtual Tensor identity(const Dim dim, const dtype type) = 0;

  /************************ Shaping and Indexing *************************/
  virtual Tensor reshape(const Tensor& tensor, const Shape& shape) = 0;
  virtual Tensor transpose(
      const Tensor& tensor,
      const Shape& dims /* = {} */) = 0;
  virtual Tensor tile(const Tensor& tensor, const Shape& shape) = 0;
  virtual Tensor concatenate(
      const std::vector<Tensor>& tensors,
      unsigned axis) = 0;
  virtual Tensor nonzero(const Tensor& tensor) = 0;

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
  virtual Tensor absolute(const Tensor& tensor) = 0;
  virtual Tensor
  clip(const Tensor& tensor, const Tensor& low, const Tensor& high) = 0;
  virtual Tensor isnan(const Tensor& tensor) = 0;

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
  FL_BINARY_OP_DECL(bitwiseOr);
  FL_BINARY_OP_DECL(bitwiseXor);
  FL_BINARY_OP_DECL(lShift);
  FL_BINARY_OP_DECL(rShift);
#undef FL_BINARY_OP_DECL
#undef FL_BINARY_OP_TYPE_DECL
#undef FL_BINARY_OP_LITERALS_DECL

  virtual Tensor minimum(const Tensor& lhs, const Tensor& rhs) = 0;
  virtual Tensor maximum(const Tensor& lhs, const Tensor& rhs) = 0;
  virtual Tensor power(const Tensor& lhs, const Tensor& rhs) = 0;

  /************************** Reductions ***************************/
  virtual Tensor amin(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double amin(const Tensor& input) = 0; // TODO: consoildate w/ above
  virtual Tensor amax(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double amax(const Tensor& input) = 0; // TODO: consoildate w/ above
  virtual Tensor sum(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double sum(const Tensor& input) = 0; // TODO: consolidate w/ above
  virtual Tensor mean(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double mean(const Tensor& input) = 0; // TODO: consolidate w/ above
  virtual Tensor
  var(const Tensor& input, const std::vector<int>& axes, bool bias) = 0;
  virtual double var(
      const Tensor& input,
      bool bias) = 0; // TODO: consolidate w/ above
  virtual Tensor std(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double norm(const Tensor& input) = 0;
  virtual Tensor countNonzero(
      const Tensor& input,
      const std::vector<int>& axes) = 0;

  /************************** Utils ***************************/
  virtual void print(const Tensor& tensor) = 0;
};

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
