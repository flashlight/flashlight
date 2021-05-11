/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Compute.h"

#include <af/array.h>
#include <af/device.h>

namespace fl {

// TODO:fl::Tensor {move,build}  move this and other files into
// `flashlight/fl/tensor/backend/af` or a directory with a similar structure.
// For now, keep everything flat to make life easier.

void sync() {
  af::sync();
}

void eval(af::array& tensor) {
  tensor.eval();
}

} // namespace fl
