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

af::array kaimingUniform(
    af::dim4 shape,
    int fanIn,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(1.0 / static_cast<double>(fanIn));
  double limit = std::sqrt(3.0) * stdv;
  af::array result = af::randu(shape, type);
  result = 2 * limit * result - limit;
  return result;
}

af::array kaimingNormal(
    af::dim4 shape,
    int fanIn,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(1.0 / static_cast<double>(fanIn));
  af::array result = af::randn(shape, type);
  if (stdv != 1) {
    result = stdv * result;
  }
  return result;
}

af::array glorotUniform(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
  double limit = std::sqrt(3.0) * stdv;
  af::array result = af::randu(shape, type);
  result = 2 * limit * result - limit;
  return result;
}

af::array glorotNormal(
    af::dim4 shape,
    int fanIn,
    int fanOut,
    af::dtype type /* = af::dtype::f32 */) {
  double stdv = std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
  af::array result = af::randn(shape, type);
  if (stdv != 1) {
    result = stdv * result;
  }
  return result;
}

} // namespace fl
