/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Random.h"

#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <af/random.h>

namespace fl {

void setSeed(int seed) {
  af::setSeed(seed);
}

Tensor randn(const Shape& shape, dtype type) {
  return Tensor(af::randn(detail::flToAfDims(shape), detail::flToAfType(type)));
}

Tensor rand(const Shape& shape, dtype type) {
  return Tensor(af::randu(detail::flToAfDims(shape), detail::flToAfType(type)));
}

} // namespace fl
