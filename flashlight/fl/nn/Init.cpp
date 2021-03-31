/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Reference
// Keras - https://git.io/fNxKN
// PyTorch - https://git.io/fNx6T

#include <cmath>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace af {
af::array uniform(af::dim4 shape, double min, double max, af::dtype type) {
  af::array result = af::randu(shape, type);
  result = (max - min) * result + min;
  return result;
}
af::array normal(af::dim4 shape, double stdv, double mean, af::dtype type) {
  af::array result = af::randn(shape, type);
  result = stdv * result + mean;
  return result;
}

af::array kaimingUniform(
    af::dim4 shape,
    int fanIn,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(1.0 / static_cast<double>(fanIn));
  double limit = std::sqrt(3.0) * stdv;
  return uniform(shape, -limit, limit, type);
}

af::array kaimingNormal(
    af::dim4 shape,
    int fanIn,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(1.0 / static_cast<double>(fanIn));
  return normal(shape, stdv, 0, type);
}

af::array glorotUniform(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
  double limit = std::sqrt(3.0) * stdv;
  return uniform(shape, -limit, limit, type);
}

af::array glorotNormal(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
  return normal(shape, stdv, 0, type);
}

af::array erfinv(const af::array& y) {
  if (af::anyTrue<bool>(af::abs(y) >= 1.)) {
    throw std::runtime_error("[erfinv] input is out of range (-1, 1)");
  }
  double a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
  double b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
  double c[4] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
  double d[2] = {3.543889200, 1.637067800};

  auto centralMask = af::abs(y) <= 0.7;

  auto z = y * y;
  auto num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
  auto dem = ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0);
  z = y * num / dem;
  auto x = z * centralMask;

  z = af::sqrt(-af::log((1.0 - af::abs(y)) / 2.0));
  num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
  dem = (d[1] * z + d[0]) * z + 1.0;
  z = 1 - 2 * af::sign(y).as(f32); // -1 for negative, 1 for positive
  z = z * num / dem;
  x = x + z * !centralMask;

  /* Two steps of Newton-Raphson correction */
  x = x - (af::erf(x) - y) / ((2.0 / std::sqrt(M_PI)) * af::exp(-x * x));
  x = x - (af::erf(x) - y) / ((2.0 / std::sqrt(M_PI)) * af::exp(-x * x));
  if (af::anyTrue<bool>(af::isNaN(x)) || af::anyTrue<bool>(af::isInf(x))) {
    throw std::runtime_error("[erfinv] invalid result");
  }
  return x;
}

} // namespace af

namespace fl {

Variable input(const af::array& arr) {
  return Variable(arr, false);
}

Variable noGrad(const af::array& arr) {
  return Variable(arr, false);
}

Variable param(const af::array& arr) {
  return Variable(arr, true);
}

Variable constant(
    double val,
    int outputSize,
    int inputSize,
    af::dtype type,
    bool calcGrad) {
  return constant(val, af::dim4(outputSize, inputSize), type, calcGrad);
}

Variable constant(double val, af::dim4 dims, af::dtype type, bool calcGrad) {
  return Variable(af::constant(val, dims, type), calcGrad);
}

Variable
identity(int outputSize, int inputSize, af::dtype type, bool calcGrad) {
  return identity(af::dim4(outputSize, inputSize), type, calcGrad);
}

Variable identity(af::dim4 dims, af::dtype type, bool calcGrad) {
  return Variable(af::identity(dims, type), calcGrad);
}

Variable uniform(
    int outputSize,
    int inputSize,
    double min,
    double max,
    af::dtype type,
    bool calcGrad) {
  return uniform(af::dim4(outputSize, inputSize), min, max, type, calcGrad);
}

Variable
uniform(af::dim4 dims, double min, double max, af::dtype type, bool calcGrad) {
  return Variable(af::uniform(dims, min, max, type), calcGrad);
}

Variable normal(
    int outputSize,
    int inputSize,
    double stdv,
    double mean,
    af::dtype type,
    bool calcGrad) {
  return normal(af::dim4(outputSize, inputSize), stdv, mean, type, calcGrad);
}

Variable
normal(af::dim4 dims, double stdv, double mean, af::dtype type, bool calcGrad) {
  return Variable(af::normal(dims, stdv, mean, type), calcGrad);
}

Variable kaimingUniform(
    af::dim4 shape,
    int fanIn,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(af::kaimingUniform(shape, fanIn, type), calcGrad);
}

Variable kaimingNormal(
    af::dim4 shape,
    int fanIn,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(af::kaimingNormal(shape, fanIn, type), calcGrad);
}

Variable glorotUniform(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(af::glorotUniform(shape, fanIn, fanOut, type), calcGrad);
}

Variable glorotNormal(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  return Variable(af::glorotNormal(shape, fanIn, fanOut, type), calcGrad);
}

Variable truncNormal(
    af::dim4 shape,
    double stdv,
    double mean,
    double minCufOff,
    double maxCutOff,
    af::dtype type,
    bool calcGrad) {
  // following: https://git.io/JYYAr
  auto normCdf = [](double x) {
    return (1. + std::erf(x / std::sqrt(2.))) / 2.;
  };

  auto l = 2 * normCdf((minCufOff - mean) / stdv) - 1;
  auto u = 2 * normCdf((maxCutOff - mean) / stdv) - 1;

  float eps = 1e-7;
  auto result = af::randu(shape, type) * (u - l) + l;
  result = af::clamp(result, -1 + eps, 1 - eps); // make sure erf is in range
  result = erfinv(result);
  result = mean + result * (stdv * std::sqrt(2.));
  result = af::clamp(result, minCufOff, maxCutOff);
  return Variable(result, calcGrad);
}

} // namespace fl
