/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

Tensor::Tensor(af::array&& array) : array_(std::move(array)) {}

af::array& Tensor::getArray() {
  return array_;
}

} // namespace fl
