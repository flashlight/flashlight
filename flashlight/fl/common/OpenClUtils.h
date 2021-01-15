/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Version 2.2
#define CL_TARGET_OPENCL_VERSION 220

#include <CL/cl.h>
#include <af/opencl.h>

namespace fl {
namespace ocl {

cl_context getContext();

cl_command_queue getQueue();

cl_device_id getDeviceId();

} // namespace ocl
} // namespace fl
