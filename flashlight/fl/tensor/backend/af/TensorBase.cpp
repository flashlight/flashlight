/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/data.h>

namespace fl {

/*
 * Below this point are ArrayFire-specific implementations. They should be
 * [re]moved to a specific ArrayFire backend implementation (along with other
 * flashlight/fl/tensor assets once migration is complete).
 */

Tensor::Tensor(af::array&& array) : array_(std::move(array)) {}

af::array& Tensor::getArray() {
  return array_;
}

const af::array& Tensor::getArray() const {
  return array_;
}

Shape Tensor::shape() const {
  return detail::afToFlDims(array_.dims());
}

fl::dtype Tensor::type() const {
  return detail::afToFlType(array_.type());
}

/******************** Tensor Creation Functions ********************/
#define AF_FULL_FUN_DEF(TYPE)                                        \
  template <>                                                        \
  Tensor full(const Shape& dims, TYPE value, const dtype type) {     \
    return Tensor(af::constant(                                      \
        value, detail::flToAfDims(dims), detail::flToAfType(type))); \
  }
AF_FULL_FUN_DEF(const double&);
AF_FULL_FUN_DEF(const float&);
AF_FULL_FUN_DEF(const int&);
AF_FULL_FUN_DEF(const unsigned&);
AF_FULL_FUN_DEF(const char&);
AF_FULL_FUN_DEF(const unsigned char&);
AF_FULL_FUN_DEF(const long&);
AF_FULL_FUN_DEF(const unsigned long&);
AF_FULL_FUN_DEF(const long long&);
AF_FULL_FUN_DEF(const unsigned long long&);
AF_FULL_FUN_DEF(const bool&);
AF_FULL_FUN_DEF(const short&);
AF_FULL_FUN_DEF(const unsigned short&);

Tensor identity(const Dim dim, const dtype type) {
  return Tensor(af::identity({dim, dim}, detail::flToAfType(type)));
}

/************************** Binary Operators ***************************/
// For ArrayFire, af::array already implements overloads for all needed
// operators -- use these by default.
#define AF_BINARY_OP_TYPE_DEF(OP, TYPE)           \
  Tensor operator OP(const Tensor& a, TYPE rhs) { \
    return Tensor(a.getArray() OP rhs);           \
  }                                               \
  Tensor operator OP(TYPE lhs, const Tensor& a) { \
    return Tensor(lhs OP a.getArray());           \
  }
#define AF_BINARY_OP_LITERALS_DEF(OP)                   \
  AF_BINARY_OP_TYPE_DEF(OP, const bool&);               \
  AF_BINARY_OP_TYPE_DEF(OP, const int&);                \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned&);           \
  AF_BINARY_OP_TYPE_DEF(OP, const char&);               \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned char&);      \
  AF_BINARY_OP_TYPE_DEF(OP, const long&);               \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned long&);      \
  AF_BINARY_OP_TYPE_DEF(OP, const long long&);          \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned long long&); \
  AF_BINARY_OP_TYPE_DEF(OP, const double&);             \
  AF_BINARY_OP_TYPE_DEF(OP, const float&);              \
  AF_BINARY_OP_TYPE_DEF(OP, const short&);              \
  AF_BINARY_OP_TYPE_DEF(OP, const unsigned short&);
// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define AF_BINARY_OP_DEF(OP, FUNC)                    \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) { \
    return Tensor(lhs.getArray() OP rhs.getArray());  \
  }                                                   \
  AF_BINARY_OP_LITERALS_DEF(OP);
// Definitions
// Since ArrayFire implements operator overloads, map both fl::Tensor functions
// and fl::Tensor operator overloads back to the af::array overloads.
AF_BINARY_OP_DEF(+, add);
AF_BINARY_OP_DEF(-, sub);
AF_BINARY_OP_DEF(*, mul);
AF_BINARY_OP_DEF(/, div);
AF_BINARY_OP_DEF(==, eq);
AF_BINARY_OP_DEF(!=, neq);
AF_BINARY_OP_DEF(<, lessThan);
AF_BINARY_OP_DEF(<=, lessThanEqual);
AF_BINARY_OP_DEF(>, greaterThan);
AF_BINARY_OP_DEF(>=, greaterThanEqual);
AF_BINARY_OP_DEF(||, logicalOr);
AF_BINARY_OP_DEF(&&, logicalAnd);
AF_BINARY_OP_DEF(%, mod);
AF_BINARY_OP_DEF(|, bitwiseOr);
AF_BINARY_OP_DEF(^, bitwiseXor);
AF_BINARY_OP_DEF(<<, lShift);
AF_BINARY_OP_DEF(>>, rShift);

Tensor minimum(const Tensor& lhs, const Tensor& rhs) {
  return Tensor(af::min(lhs.getArray(), rhs.getArray()));
}

Tensor maximum(const Tensor& lhs, const Tensor& rhs) {
  return Tensor(af::max(lhs.getArray(), rhs.getArray()));
}

/************************** Reductions ***************************/

Tensor amin(const Tensor& input, std::optional<const int> dim) {
  // in ArrayFire, -1 --> all axes
  return Tensor(af::min(input.getArray(), dim ? dim.value() : -1));
}

Tensor amax(const Tensor& input, std::optional<const int> dim) {
  // in ArrayFire, -1 --> all axes
  return Tensor(af::max(input.getArray(), dim ? dim.value() : -1));
}

} // namespace fl
