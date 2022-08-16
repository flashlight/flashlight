/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/DefaultTensorType.h"

namespace fl {

TensorBackend& defaultTensorBackend() {
  return Tensor().backend();
}

} // namespace fl
