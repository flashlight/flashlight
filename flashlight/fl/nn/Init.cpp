/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Reference
// Keras - https://git.io/fNxKN
// PyTorch - https://git.io/fNx6T

#include <cmath>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/tensor/Random.h"

namespace fl {
namespace detail {

Tensor uniform(const Shape& shape, double min, double max, fl::dtype type) {
  Tensor result = fl::rand(shape, type);
  result = (max - min) * result + min;
  return result;
}
Tensor normal(const Shape& shape, double stdv, double mean, fl::dtype type) {
  Tensor result = fl::randn(shape, type);
  result = stdv * result + mean;
  return result;
}

Tensor kaimingUniform(
    const Shape& shape,
    int fanIn,
    fl::dtype type /* = fl::dtype::f32 */) {
  double stdv = std::sqrt(1.0 / static_cast<double>(fanIn));
  double limit = std::sqrt(3.0) * stdv;
  return detail::uniform(shape, -limit, limit, type);
}

Tensor kaimingNormal(
    const Shape& shape,
    int fanIn,
    fl::dtype type /* = fl::dtype::f32 */) {
  double stdv = std::sqrt(1.0 / static_cast<double>(fanIn));
  return detail::normal(shape, stdv, 0, type);
}

Tensor glorotUniform(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type /* = fl::dtype::f32 */) {
  double stdv = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
  double limit = std::sqrt(3.0) * stdv;
  return detail::uniform(shape, -limit, limit, type);
}

Tensor glorotNormal(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type /* = fl::dtype::f32 */) {
  double stdv = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
  return detail::normal(shape, stdv, 0, type);
}

Tensor erfinv(const Tensor& y) {
  if (fl::any(fl::abs(y) >= 1.).scalar<char>()) {
    throw std::runtime_error("[erfinv] input is out of range (-1, 1)");
  }
  double a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
  double b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
  double c[4] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
  double d[2] = {3.543889200, 1.637067800};

  auto centralMask = fl::abs(y) <= 0.7;

  auto z = y * y;
  auto num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
  auto dem = ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0);
  z = y * num / dem;
  auto x = z * centralMask;

  z = fl::sqrt(-fl::log((1.0 - fl::abs(y)) / 2.0));
  num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
  dem = (d[1] * z + d[0]) * z + 1.0;
  // TODO{fl::Tensor}{operator} - check af::sign - zero case?
  z = fl::sign(y).astype(fl::dtype::f32); // -1 for negative, 1 for positive
  z = z * num / dem;
  x = x + z * !centralMask;

  /* Two steps of Newton-Raphson correction */
  x = x - (fl::erf(x) - y) / ((2.0 / std::sqrt(M_PI)) * fl::exp(-x * x));
  x = x - (fl::erf(x) - y) / ((2.0 / std::sqrt(M_PI)) * fl::exp(-x * x));
  if (fl::any(fl::isnan(x)).asScalar<bool>() ||
      fl::any(fl::isinf(x)).asScalar<bool>()) {
    throw std::runtime_error("[erfinv] invalid result");
  }
  return x;
}

} // namespace detail

Variable input(const Tensor& arr) {
  return Variable(arr, false);
}

Variable noGrad(const Tensor& arr) {
  return Variable(arr, false);
}

Variable param(const Tensor& arr) {
  return Variable(arr, true);
}

Variable constant(
    double val,
    int outputSize,
    int inputSize,
    fl::dtype type,
    bool calcGrad) {
  return constant(val, Shape({outputSize, inputSize}), type, calcGrad);
}

Variable
constant(double val, const Shape& dims, fl::dtype type, bool calcGrad) {
  return Variable(fl::full(dims, val, type), calcGrad);
}

Variable
identity(int outputSize, int inputSize, fl::dtype type, bool calcGrad) {
  // TODO{fl::Tensor}{fixme} add non-square identity to API
  if (inputSize != outputSize) {
    throw std::invalid_argument(
        "identity - can't create tensor with "
        "different in and output size - only square identity "
        "tensors supported");
  }
  return identity(Shape({inputSize, outputSize}), type, calcGrad);
}

Variable identity(const Shape& dims, fl::dtype type, bool calcGrad) {
  return Variable(fl::identity(dims.dim(0), type), calcGrad);
}

Variable uniform(
    int outputSize,
    int inputSize,
    double min,
    double max,
    fl::dtype type,
    bool calcGrad) {
  return uniform(Shape({outputSize, inputSize}), min, max, type, calcGrad);
}

Variable uniform(
    const Shape& dims,
    double min,
    double max,
    fl::dtype type,
    bool calcGrad) {
  return Variable(detail::uniform(dims, min, max, type), calcGrad);
}

Variable normal(
    int outputSize,
    int inputSize,
    double stdv,
    double mean,
    fl::dtype type,
    bool calcGrad) {
  return normal(Shape({outputSize, inputSize}), stdv, mean, type, calcGrad);
}

Variable normal(
    const Shape& dims,
    double stdv,
    double mean,
    fl::dtype type,
    bool calcGrad) {
  return Variable(detail::normal(dims, stdv, mean, type), calcGrad);
}

Variable kaimingUniform(
    const Shape& shape,
    int fanIn,
    fl::dtype type /* = fl::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(detail::kaimingUniform(shape, fanIn, type), calcGrad);
}

Variable kaimingNormal(
    const Shape& shape,
    int fanIn,
    fl::dtype type /* = fl::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(detail::kaimingNormal(shape, fanIn, type), calcGrad);
}

Variable glorotUniform(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type /* = fl::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(detail::glorotUniform(shape, fanIn, fanOut, type), calcGrad);
}

Variable glorotNormal(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type /* = fl::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(detail::glorotNormal(shape, fanIn, fanOut, type), calcGrad);
}

Variable truncNormal(
    const Shape& shape,
    double stdv,
    double mean,
    double minCufOff,
    double maxCutOff,
    fl::dtype type,
    bool calcGrad) {
  // following: https://git.io/JYYAr
  auto normCdf = [](double x) {
    return (1. + std::erf(x / std::sqrt(2.))) / 2.;
  };

  auto l = 2 * normCdf((minCufOff - mean) / stdv) - 1;
  auto u = 2 * normCdf((maxCutOff - mean) / stdv) - 1;

  float eps = 1e-7;
  auto result = fl::rand(shape, type) * (u - l) + l;
  result = fl::clip(result, -1 + eps, 1 - eps); // make sure erf is in range
  result = detail::erfinv(result);
  result = mean + result * (stdv * std::sqrt(2.));
  result = fl::clip(result, minCufOff, maxCutOff);
  return Variable(result, calcGrad);
}

} // namespace fl
