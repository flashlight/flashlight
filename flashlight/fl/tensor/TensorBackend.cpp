/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {
namespace detail {

bool areBackendsEqual(const Tensor& a, const Tensor& b) {
  return a.backendType() == b.backendType();
}

} // namespace detail

bool TensorBackend::isDataTypeSupported(const fl::dtype& dtype) const {
  bool supported = this->supportsDataType(dtype);
  for (auto& p : extensions_) {
    supported &= p.second->isDataTypeSupported(dtype);
  }
  return supported;
}

Tensor TensorBackend::clip(
    const Tensor& tensor,
    const Tensor& low,
    const double& high) {
  return clip(
      tensor, low, full(tensor.shape(), high, dtype_traits<double>::ctype));
}

Tensor TensorBackend::clip(
    const Tensor& tensor,
    const double& low,
    const Tensor& high) {
  return clip(
      tensor, full(tensor.shape(), low, dtype_traits<double>::ctype), high);
}

Tensor TensorBackend::clip(
    const Tensor& tensor,
    const double& low,
    const double& high) {
  return clip(
      tensor,
      full(tensor.shape(), low, dtype_traits<double>::ctype),
      full(tensor.shape(), high, dtype_traits<double>::ctype));
}

Tensor TensorBackend::where(
    const Tensor& condition,
    const Tensor& x,
    const double& y) {
  return where(condition, x, full(condition.shape(), y, x.type()));
}

Tensor TensorBackend::where(
    const Tensor& condition,
    const double& x,
    const Tensor& y) {
  return where(condition, full(condition.shape(), x, y.type()), y);
}

Tensor TensorBackend::minimum(const Tensor& lhs, const double& rhs) {
  return minimum(lhs, full(lhs.shape(), rhs, dtype_traits<double>::ctype));
}

Tensor TensorBackend::minimum(const double& lhs, const Tensor& rhs) {
  return minimum(full(rhs.shape(), lhs, dtype_traits<double>::ctype), rhs);
}

Tensor TensorBackend::maximum(const Tensor& lhs, const double& rhs) {
  return maximum(lhs, full(lhs.shape(), rhs, dtype_traits<double>::ctype));
}

Tensor TensorBackend::maximum(const double& lhs, const Tensor& rhs) {
  return maximum(full(rhs.shape(), lhs, dtype_traits<double>::ctype), rhs);
}

Tensor TensorBackend::power(const Tensor& lhs, const double& rhs) {
  return power(lhs, full(lhs.shape(), rhs, dtype_traits<double>::ctype));
}

Tensor TensorBackend::power(const double& lhs, const Tensor& rhs) {
  return power(full(rhs.shape(), lhs, dtype_traits<double>::ctype), rhs);
}

} // namespace fl
