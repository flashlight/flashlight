/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/arith.h>
#include <af/gfor.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

namespace fl {
namespace {

bool canBroadcast(const Shape& lhs, const Shape& rhs) {
  unsigned nDim = std::max(lhs.ndim(), rhs.ndim());

  for (unsigned i = 0; i < nDim; ++i) {
    if (i + 1 > lhs.ndim() || i + 1 > rhs.ndim()) {
      // One Shape has more dimensions than the other - will broadcast to the
      // smaller tensor
      continue;
    }
    if (lhs[i] != rhs[i] && lhs[i] != 1 && rhs[i] != 1) {
      return false;
    }
  }
  return true;
}

// A binary operation on two ArrayFire arrays
using binaryOpFunc_t =
    af::array (*)(const af::array& lhs, const af::array& rhs);

Tensor doBinaryOpOrBroadcast(
    const Tensor& lhs,
    const Tensor& rhs,
    binaryOpFunc_t func) {
  // Dims are the same or scalar <> 1-el tensor - no broadcasting
  if (lhs.shape() == rhs.shape() ||
      (lhs.elements() <= 1 && rhs.elements() <= 1)) {
    return toTensor<ArrayFireTensor>(
        func(toArray(lhs), toArray(rhs)), lhs.ndim());
  }

  if (canBroadcast(lhs.shape(), rhs.shape())) {
    return toTensor<ArrayFireTensor>(
        af::batchFunc(toArray(lhs), toArray(rhs), func),
        std::max(lhs.ndim(), rhs.ndim()));
  } else {
    std::stringstream ss;
    ss << "doBinaryOpOrBroadcast: cannot perform operation "
          "or broadcasting with tensors of shapes "
       << lhs.shape() << " and " << rhs.shape() << " - dimension mismatch.";
    throw std::invalid_argument(ss.str());
  }
}
} // namespace

// For ArrayFire, af::array already implements overloads for all needed
// operators -- use these by default.
#define FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, TYPE)                   \
  Tensor ArrayFireBackend::FUNC(const Tensor& a, TYPE rhs) {       \
    return toTensor<ArrayFireTensor>(toArray(a) OP rhs, a.ndim()); \
  }                                                                \
  Tensor ArrayFireBackend::FUNC(TYPE lhs, const Tensor& a) {       \
    return toTensor<ArrayFireTensor>(lhs OP toArray(a), a.ndim()); \
  }

#define FL_AF_BINARY_OP_LITERALS_DEF(FUNC, OP)                   \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const bool&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const int&);                \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned&);           \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const char&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned char&);      \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const long&);               \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long&);      \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const long long&);          \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned long long&); \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const double&);             \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const float&);              \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const short&);              \
  FL_AF_BINARY_OP_TYPE_DEF(FUNC, OP, const unsigned short&);

// Operations on fl::Tensor call the respective operator overloads that are
// already defined on af::arrays
#define FL_AF_BINARY_OP_DEF(OP, FUNC)                                   \
  Tensor ArrayFireBackend::FUNC(const Tensor& lhs, const Tensor& rhs) { \
    return doBinaryOpOrBroadcast(lhs, rhs, af::operator OP);            \
  }                                                                     \
  FL_AF_BINARY_OP_LITERALS_DEF(FUNC, OP);

// Definitions
// Since ArrayFire implements operator overloads, map both fl::Tensor
// functions and fl::Tensor operator overloads back to the af::array
// overloads.
FL_AF_BINARY_OP_DEF(+, add);
FL_AF_BINARY_OP_DEF(-, sub);
FL_AF_BINARY_OP_DEF(*, mul);
FL_AF_BINARY_OP_DEF(/, div);
FL_AF_BINARY_OP_DEF(==, eq);
FL_AF_BINARY_OP_DEF(!=, neq);
FL_AF_BINARY_OP_DEF(<, lessThan);
FL_AF_BINARY_OP_DEF(<=, lessThanEqual);
FL_AF_BINARY_OP_DEF(>, greaterThan);
FL_AF_BINARY_OP_DEF(>=, greaterThanEqual);
FL_AF_BINARY_OP_DEF(||, logicalOr);
FL_AF_BINARY_OP_DEF(&&, logicalAnd);
FL_AF_BINARY_OP_DEF(%, mod);
FL_AF_BINARY_OP_DEF(&, bitwiseAnd);
FL_AF_BINARY_OP_DEF(|, bitwiseOr);
FL_AF_BINARY_OP_DEF(^, bitwiseXor);
FL_AF_BINARY_OP_DEF(<<, lShift);
FL_AF_BINARY_OP_DEF(>>, rShift);
#undef FL_AF_BINARY_OP_DEF
#undef FL_AF_BINARY_OP_TYPE_DEF
#undef FL_AF_BINARY_OP_LITERALS_DEF

Tensor ArrayFireBackend::minimum(const Tensor& lhs, const Tensor& rhs) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::min);
}

Tensor ArrayFireBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::max);
}

Tensor ArrayFireBackend::power(const Tensor& lhs, const Tensor& rhs) {
  return doBinaryOpOrBroadcast(lhs, rhs, af::pow);
}
} // namespace fl
