/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
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

#include "flashlight/autograd/Functions.h"
#include "flashlight/nn/Init.h"
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

} // namespace fl
