/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/autograd/Functions.h>
#include <flashlight/autograd/Utils.h>
#include <flashlight/autograd/Variable.h>
#include <flashlight/common/DevicePtr.h>
#include <flashlight/common/Exception.h>

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
  AFML_THROW_ERR("CPU Pool2D is not yet supported.", AF_ERR_NOT_SUPPORTED);
}

} // namespace fl
