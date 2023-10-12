/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Random.h"

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

void setSeed(const int seed) {
  defaultTensorBackend().setSeed(seed);
}

Tensor randn(const Shape& shape, dtype type) {
  return defaultTensorBackend().randn(shape, type);
}

Tensor rand(const Shape& shape, dtype type) {
  return defaultTensorBackend().rand(shape, type);
}

} // namespace fl
