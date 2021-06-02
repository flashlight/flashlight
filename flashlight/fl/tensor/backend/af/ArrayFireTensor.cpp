/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/data.h>
#include <af/gfor.h>
#include <af/lapack.h>

namespace fl {

namespace {

typedef af::array (*reduceFunc_t)(const af::array&, const int);

af::array afReduceAxes(
    const af::array& input,
    const std::vector<int>& axes,
    reduceFunc_t func) {
  auto arr = input;
  for (int dim : axes) {
    arr = func(arr, dim);
  }
  return arr;
}

} // namespace

ArrayFireTensor::ArrayFireTensor(af::array&& array)
    : array_(std::move(array)), shape_(detail::afToFlDims(array_.dims())) {}

ArrayFireTensor::ArrayFireTensor() {}

TensorBackendType ArrayFireTensor::backendType() const {
  return TensorBackendType::ArrayFire;
}

TensorBackend& ArrayFireTensor::backend() const {
  // The ArrayFire backend has a single ArrayFireBackend instance per process.
  return ::fl::ArrayFireBackend::getInstance();
}

const Shape& ArrayFireTensor::shape() {
  // Update the Shape in-place
  detail::afToFlDims(array_.dims(), shape_);
  return shape_;
}

fl::dtype ArrayFireTensor::type() const {
  return detail::afToFlType(array_.type());
}

Tensor ArrayFireTensor::astype(const dtype type) {
  auto a = array_.as(detail::flToAfType(type));
  return toTensor<ArrayFireTensor>(std::move(a));
}

af::array& ArrayFireTensor::getHandle() {
  return array_;
}

const af::array& ArrayFireTensor::getHandle() const {
  return array_;
}

af::array& toArray(const Tensor& tensor) {
  if (tensor.backendType() != TensorBackendType::ArrayFire) {
    throw std::invalid_argument("toArray: tensor is not ArrayFire-backed");
  }
  return tensor.getAdapter<ArrayFireTensor>().getHandle();
}

/******************** Tensor Creation Functions ********************/
#define AF_FULL_FUN_DEF(TYPE)                                        \
  template <>                                                        \
  Tensor full(const Shape& dims, TYPE value, const dtype type) {     \
    return toTensor<ArrayFireTensor>(af::constant(                   \
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
  return toTensor<ArrayFireTensor>(
      af::identity({dim, dim}, detail::flToAfType(type)));
}

/************************** Unary Operators ***************************/
Tensor negative(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(-toArray(tensor));
}

Tensor logicalNot(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(!toArray(tensor));
}

Tensor log1p(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log1p(toArray(tensor)));
}

Tensor sin(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sin(toArray(tensor)));
}

Tensor cos(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::cos(toArray(tensor)));
}

Tensor sqrt(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sqrt(toArray(tensor)));
}

Tensor tanh(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::tanh(toArray(tensor)));
}

Tensor absolute(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::abs(toArray(tensor)));
}

Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high) {
  return toTensor<ArrayFireTensor>(
      af::clamp(toArray(tensor), toArray(low), toArray(high)));
}

Tensor clip(const Tensor& tensor, const Tensor& low, const double& high) {
  return clip(tensor, low, full(tensor.shape(), high));
}

Tensor clip(const Tensor& tensor, const double& low, const Tensor& high) {
  return clip(tensor, full(tensor.shape(), low), high);
}

Tensor clip(const Tensor& tensor, const double& low, const double& high) {
  return clip(tensor, full(tensor.shape(), low), full(tensor.shape(), high));
}

Tensor isnan(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isNaN(toArray(tensor)));
}

/************************** Binary Operators ***************************/
// For ArrayFire, af::array already implements overloads for all needed
// operators -- use these by default.
#define AF_BINARY_OP_TYPE_DEF(OP, TYPE)                  \
  Tensor operator OP(const Tensor& a, TYPE rhs) {        \
    return toTensor<ArrayFireTensor>(toArray(a) OP rhs); \
  }                                                      \
  Tensor operator OP(TYPE lhs, const Tensor& a) {        \
    return toTensor<ArrayFireTensor>(lhs OP toArray(a)); \
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
#define AF_BINARY_OP_DEF(OP, FUNC)                                  \
  Tensor FUNC(const Tensor& lhs, const Tensor& rhs) {               \
    return toTensor<ArrayFireTensor>(toArray(lhs) OP toArray(rhs)); \
  }                                                                 \
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
  return toTensor<ArrayFireTensor>(af::min(toArray(lhs), toArray(rhs)));
}

Tensor minimum(const Tensor& lhs, const double& rhs) {
  return minimum(lhs, full(lhs.shape(), rhs));
}

Tensor minimum(const double& lhs, const Tensor& rhs) {
  return minimum(full(rhs.shape(), lhs), rhs);
}

Tensor maximum(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::max(toArray(lhs), toArray(rhs)));
}

Tensor maximum(const Tensor& lhs, const double& rhs) {
  return maximum(lhs, full(lhs.shape(), rhs));
}

Tensor maximum(const double& lhs, const Tensor& rhs) {
  return maximum(full(rhs.shape(), lhs), rhs);
}

Tensor power(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::pow(toArray(lhs), toArray(rhs)));
}

/************************** Reductions ***************************/

Tensor amin(const Tensor& input, const std::vector<int>& axes) {
  return toTensor<ArrayFireTensor>(afReduceAxes(toArray(input), axes, af::min));
}

Tensor amax(const Tensor& input, const std::vector<int>& axes) {
  return toTensor<ArrayFireTensor>(afReduceAxes(toArray(input), axes, af::max));
}

Tensor sum(const Tensor& input, const std::vector<int>& axes) {
  return toTensor<ArrayFireTensor>(afReduceAxes(toArray(input), axes, af::sum));
}

Tensor mean(const Tensor& input, const std::vector<int>& axes) {
  // Cannot use afReduceAxes because sum uses dim instead of int
  auto arr = toArray(input);
  for (int dim : axes) {
    arr = af::mean(arr, dim);
  }
  return toTensor<ArrayFireTensor>(std::move(arr));
}

Tensor var(const Tensor& input, const std::vector<int>& axes, bool bias) {
  // Use arrayfire default for one dimension which may be optimized
  auto& arr = toArray(input);
  if (axes.size() == 1) {
    return toTensor<ArrayFireTensor>(af::var(arr, bias, axes[0]));
  }
  auto meanArr = mean(input, axes);
  // TODO Replace when we have batchFunc for fl::Tensor
  auto x = af::batchFunc(arr, toArray(meanArr), af::operator-);
  x = af::pow(x, 2);
  x = afReduceAxes(x, axes, af::sum);

  int denominator = 1;
  auto dims = toArray(input).dims();
  for (auto dim : axes) {
    denominator *= dims[dim];
  }
  if (bias) {
    denominator--;
  }

  x = x / denominator;
  return toTensor<ArrayFireTensor>(std::move(x));
}

double norm(const Tensor& input) {
  return af::norm(toArray(input));
}

} // namespace fl
