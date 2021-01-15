/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/OpenClUtils.h"

namespace fl {
namespace ocl {

cl_context getContext() {
  return afcl::getContext(/*retain=*/true);
}

cl_command_queue getQueue() {
  return afcl::getQueue(/*retain=*/true);
}

cl_device_id getDeviceId() {
  return afcl::getDeviceId();
}

} // namespace ocl
} // namespace fl
