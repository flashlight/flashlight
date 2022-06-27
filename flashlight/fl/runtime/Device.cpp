/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/Device.h"

namespace fl {

void X64Device::setActive() const {
  // no op, CPU device is always active
}

} // namespace fl
