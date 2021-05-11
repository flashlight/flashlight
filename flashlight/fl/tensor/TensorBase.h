/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// TODO:fl::Tensor {af} remove me when ArrayFire is a particular subimpl
#include <af/array.h>

#include "flashlight/fl/tensor/ShapeBase.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

/**
 * A base tensor interface under which implementations can define tensor
 * operations. These operations may differ across tensor libraries,
 * hardware-specific libraries or compilers and DSLs.
 *
 * TODO:fl::Tensor {doc} more documentation. For now, this class serves as a
 * simple shim between tensor operations expressed in Flashlight and underlying
 * tensor computations in ArrayFire; for now, this API will wrap the ArrayFire
 * API using semantics similar to
 * [numpy](https://numpy.org/doc/stable/reference/) and translating those into
 * ArrayFire semantics. NOTE: this API will break frequently and is not yet
 * stable.
 */
class Tensor {
  // TODO:fl::Tensor {af} remove me when ArrayFire is a particular subimpl
  af::array array_;

 public:
  /**
   * Temporary. Since af::arrays are refcounted, an instance of this class
   * should only be created using arrays that are moved therein. Tensor
   * operations occurring on that array, while adapting functions in Flashlight,
   * should operate on references and should not copy the array else take a
   * performance penalty (via an internal copy if refcount is > 1 in some
   * cases).
   *
   * @param[in] array&& construct a tensor from an ArrayFire array rvalue
   * reference.
   */
  explicit Tensor(af::array&& array);

  // TODO:fl::Tensor {af} remove me when not dependent on AF
  af::array& getArray();
  const af::array& getArray() const;

  /**
   * Get the shape of a tensor.
   *
   * @return the shape of the tensor
   */
  Shape shape() const;

  /**
   * Get the data type of tensor.
   *
   * @return the dtype of the tensor
   */
  dtype type() const;
};

/******************** Tensor Creation Functions ********************/
/**
 * Creates a new tensor with a given shape and filled with a particular value.
 *
 * @param[in] dims the shape of the tensor to create
 * @param[in] val the value with which to fill the tensor
 * @param[in] type the type of the tensor to create. Defaults to a value based
 * on the value type
 * @return a tensor of the specified shape filled with the specified value
 */
template <typename T>
Tensor full(
    const Shape& dims,
    const T& val,
    const dtype type = dtype_traits<T>::ctype);

/**
 * Return a the identity tensor of a given size and type.
 *
 * @param[in] dim the size of the dimension of the matrix (dim x dim)
 * @param[in] type the type of the resulting matrix
 */
Tensor identity(const Dim dim, const dtype type = dtype::f32);

/************************** Binary Operators ***************************/
#define BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, TYPE) \
  Tensor FUNC(TYPE lhs, const Tensor& rhs);         \
  Tensor FUNC(const Tensor& lhs, TYPE rhs);         \
  Tensor operator OP(TYPE lhs, const Tensor& rhs);  \
  Tensor operator OP(const Tensor& lhs, TYPE rhs);
#define BINARY_OP_LITERALS_DECL(OP, FUNC)                           \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const bool&);               \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const int&);                \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned&);           \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const char&);               \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned char&);      \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const long&);               \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned long&);      \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const long long&);          \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned long long&); \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const double&);             \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const float&);              \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const short&);              \
  BINARY_OP_LITERAL_TYPE_DECL(OP, FUNC, const unsigned short&);
#define BINARY_OP_DECL(OP, FUNC)                                    \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs);                \
  inline Tensor operator OP(const Tensor& lhs, const Tensor& rhs) { \
    return FUNC(lhs, rhs);                                          \
  }                                                                 \
  BINARY_OP_LITERALS_DECL(OP, FUNC);
// Definitions. Each declares:
// - a functional operator that takes two tensors (e.g. add)
// - a symbolic operator that takes two tensors (calls the functional one)
// - functional and symbolic operators for all lhs/rhs primitive types
BINARY_OP_DECL(+, add);
BINARY_OP_DECL(-, sub);
BINARY_OP_DECL(*, mul);
BINARY_OP_DECL(/, div);
BINARY_OP_DECL(==, eq);
BINARY_OP_DECL(!=, neq);
BINARY_OP_DECL(<, lessThan);
BINARY_OP_DECL(<=, lessThanEqual);
BINARY_OP_DECL(>, greaterThan);
BINARY_OP_DECL(>=, greaterThanEqual);
BINARY_OP_DECL(||, logicalOr);
BINARY_OP_DECL(%, mod);
BINARY_OP_DECL(|, bitwiseOr);
BINARY_OP_DECL(^, bitwiseXor);
BINARY_OP_DECL(<<, lShift);
BINARY_OP_DECL(>>, rShift);

} // namespace fl
