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
    af::dim4 shape,
    int fanIn,
    float gain,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  double stdv = gain / std::sqrt((double)fanIn);
  double limit = std::sqrt(3.0) * stdv;
  return uniform(shape, -limit, limit, type, calcGrad);
}

Variable kaimingNormal(
    af::dim4 shape,
    int fanIn,
    float gain,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  double stdv = gain / std::sqrt((double)fanIn);
  return normal(shape, stdv, 0, type, calcGrad);
}

Variable glorotUniform(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  double stdv = std::sqrt(2.0 / (double)(fanIn + fanOut));
  double limit = std::sqrt(3.0) * stdv;
  return uniform(shape, -limit, limit, type, calcGrad);
}

Variable glorotNormal(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */,
    bool calcGrad /* = true */) {
  double stdv = std::sqrt(2.0 / (double)(fanIn + fanOut));
  return normal(fanIn, stdv, 0, type, calcGrad);
}

} // namespace fl
