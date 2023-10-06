/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/arith.h>
#include <af/data.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

namespace fl {

Tensor ArrayFireBackend::exp(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::exp(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::log(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::negative(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(-toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::logicalNot(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(!toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::log1p(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log1p(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sin(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sin(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::cos(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::cos(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sqrt(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sqrt(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::tanh(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::tanh(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::floor(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::floor(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::ceil(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::ceil(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::rint(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::round(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::absolute(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::abs(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sigmoid(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sigmoid(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::erf(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::erf(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::flip(const Tensor& tensor, const unsigned dim) {
  return toTensor<ArrayFireTensor>(
      af::flip(toArray(tensor), dim), tensor.ndim());
}

Tensor ArrayFireBackend::clip(
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high) {
  return toTensor<ArrayFireTensor>(
      af::clamp(toArray(tensor), toArray(low), toArray(high)), tensor.ndim());
}

Tensor ArrayFireBackend::roll(
    const Tensor& tensor,
    const int shift,
    const unsigned axis) {
  if (axis > AF_MAX_DIMS) {
    throw std::invalid_argument(
        "ArrayFireBackend::roll - given axis > 3 - unsupported");
  }
  std::vector<Dim> shifts(AF_MAX_DIMS, 0);
  shifts[axis] = shift;
  return toTensor<ArrayFireTensor>(
      af::shift(toArray(tensor), shifts[0], shifts[1], shifts[2], shifts[3]),
      tensor.ndim());
}

Tensor ArrayFireBackend::isnan(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isNaN(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::isinf(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isInf(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sign(const Tensor& tensor) {
  auto wSigned = 1 - 2 * af::sign(toArray(tensor));
  wSigned(toArray(tensor) == 0) = 0;
  return toTensor<ArrayFireTensor>(std::move(wSigned), tensor.ndim());
}

Tensor ArrayFireBackend::tril(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(
      af::lower(toArray(tensor), /* is_unit_diag = */ false), tensor.ndim());
}

Tensor ArrayFireBackend::triu(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(
      af::upper(toArray(tensor), /* is_unit_diag = */ false), tensor.ndim());
}
} // namespace fl
