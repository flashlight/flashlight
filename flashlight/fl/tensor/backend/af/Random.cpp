/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Random.h"

#include <af/random.h>

namespace fl {

void setSeed(int seed) {
  af::setSeed(seed);
}

} // namespace fl
