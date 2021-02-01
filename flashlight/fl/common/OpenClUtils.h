/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
// Version 2.2
#define CL_TARGET_OPENCL_VERSION 220
#endif

#include <CL/cl.h>
#include <af/opencl.h>

#include "flashlight/fl/common/DevicePtr.h"

namespace fl {
namespace ocl {

/**
 * A device pointer subclass for the OpenCL backend to handle cl_mem that
 * follows retain/release OpenCL semantics when using OpenCL objects during
 * construction.
 *
 * Handles the cl_mem provided by ArrayFire that must be explicitly deleted.
 */
class DevicePtrOpenCl : public fl::DevicePtr {
 public:
  DevicePtrOpenCl(const af::array& in) : fl::DevicePtr(in) {
    clMemBuf_ = in.device<cl_mem>();
  }

  ~DevicePtrOpenCl() {
    delete clMemBuf_;
  }

  cl_mem* getAsClMem() const {
    return clMemBuf_;
  }

 private:
  cl_mem* clMemBuf_;
};

/**
 * Gets the Arrayfire OpenCL context for the current device
 */
cl_context getContext();

/**
 * Gets the Arrayfire OpenCL queue for the current device.
 */
cl_command_queue getQueue();

/**
 * Gets the Arrayfire OpenCL device ID for the current device.
 */
cl_device_id getDeviceId();

} // namespace ocl
} // namespace fl
