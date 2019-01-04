/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
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


#include "flashlight/nn/Init.h"

#include <cmath>

#include "flashlight/nn/Utils.h"

namespace fl {

using detail::computeFans;

Variable input(const af::array& arr) {
  return Variable(arr, false);
}

Variable noGrad(const af::array& arr) {
  return Variable(arr, false);
}

Variable param(const af::array& arr) {
  return Variable(arr, true);
}

Variable uniform(
    int output_size,
    int input_size,
    double min,
    double max,
    af::dtype type,
    bool calc_grad) {
  return uniform(af::dim4(output_size, input_size), min, max, type, calc_grad);
}

Variable
uniform(af::dim4 dims, double min, double max, af::dtype type, bool calc_grad) {
  af::array result = af::randu(dims, type);
  if (min != 0 || max != 1) {
    result = (max - min) * result + min;
  }
  return Variable(result, calc_grad);
}

Variable normal(
    int output_size,
    int input_size,
    double stdv,
    double mean,
    af::dtype type,
    bool calc_grad) {
  return normal(af::dim4(output_size, input_size), stdv, mean, type, calc_grad);
}

Variable normal(
    af::dim4 dims,
    double stdv,
    double mean,
    af::dtype type,
    bool calc_grad) {
  af::array result = af::randn(dims, type);
  if (mean != 0 || stdv != 1) {
    result = stdv * result + mean;
  }
  return Variable(result, calc_grad);
}

Variable kaimingUniform(
    int output_size,
    int input_size,
    af::dtype type,
    bool calc_grad) {
  return kaimingUniform(af::dim4(output_size, input_size), type, calc_grad);
}

Variable kaimingUniform(af::dim4 dims, af::dtype type, bool calc_grad) {
  dim_t fan_in = computeFans(dims).first;
  double stdv = std::sqrt(1.0 / (double)fan_in);
  double limit = std::sqrt(3.0) * stdv;
  return uniform(dims, -limit, limit, type, calc_grad);
}

Variable
kaimingNormal(int output_size, int input_size, af::dtype type, bool calc_grad) {
  return kaimingNormal(af::dim4(output_size, input_size), type, calc_grad);
}

Variable kaimingNormal(af::dim4 dims, af::dtype type, bool calc_grad) {
  dim_t fan_in = computeFans(dims).first;
  double stdv = std::sqrt(1.0 / (double)fan_in);
  return normal(dims, stdv, 0, type, calc_grad);
}

Variable
glorotUniform(int output_size, int input_size, af::dtype type, bool calc_grad) {
  return glorotUniform(af::dim4(output_size, input_size), type, calc_grad);
}

Variable glorotUniform(af::dim4 dims, af::dtype type, bool calc_grad) {
  auto fans = computeFans(dims);
  dim_t fan_in = fans.first;
  dim_t fan_out = fans.second;
  double stdv = std::sqrt(2.0 / (double)(fan_in + fan_out));
  double limit = std::sqrt(3.0) * stdv;
  return uniform(dims, -limit, limit, type, calc_grad);
}

Variable
glorotNormal(int output_size, int input_size, af::dtype type, bool calc_grad) {
  return glorotNormal(af::dim4(output_size, input_size), type, calc_grad);
}

Variable glorotNormal(af::dim4 dims, af::dtype type, bool calc_grad) {
  auto fans = computeFans(dims);
  dim_t fan_in = fans.first;
  dim_t fan_out = fans.second;
  double stdv = std::sqrt(2.0 / (double)(fan_in + fan_out));
  return normal(dims, stdv, 0, type, calc_grad);
}

Variable constant(
    double val,
    int output_size,
    int input_size,
    af::dtype type,
    bool calc_grad) {
  return constant(val, af::dim4(output_size, input_size), type, calc_grad);
}

Variable constant(double val, af::dim4 dims, af::dtype type, bool calc_grad) {
  return Variable(af::constant(val, dims, type), calc_grad);
}

Variable
identity(int output_size, int input_size, af::dtype type, bool calc_grad) {
  return identity(af::dim4(output_size, input_size), type, calc_grad);
}

Variable identity(af::dim4 dims, af::dtype type, bool calc_grad) {
  return Variable(af::identity(dims, type), calc_grad);
}

} // namespace fl
