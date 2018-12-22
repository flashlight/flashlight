/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include <flashlight/autograd/Functions.h>
#include <flashlight/autograd/Utils.h>
#include <flashlight/autograd/Variable.h>
#include <flashlight/common/DevicePtr.h>

namespace fl {

Variable pool2d(
    const Variable& input,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode) {
  throw std::runtime_error("pool2d not yet implemented on CPU");
}

} // namespace fl
