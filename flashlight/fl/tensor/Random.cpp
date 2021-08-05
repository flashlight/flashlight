/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Random.h"

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

void setSeed(int seed) {
  Tensor().backend().setSeed(seed);
}

Tensor randn(const Shape& shape, dtype type) {
  return Tensor().backend().randn(shape, type);
}

Tensor rand(const Shape& shape, dtype type) {
  return Tensor().backend().rand(shape, type);
}

} // namespace fl
