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

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <CL/cl.h>
#include <af/opencl.h>

#include "flashlight/fl/common/DevicePtr.h"

#define FL_OPENCL_CHECK(err) \
  ::fl::ocl::detail::check(err, __FILE__, __LINE__, #err)

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

/**
 * Singleton warpper to a single opencl context. Used for creating and
 * enqueuing opencl kernels in that context.
 *
 * Example:
 * DevicePtrOpenCl inputPtr(input);
 * int h = input.dims(0);
 * int w = input.dims(1);
 * cl_kernel kernel =
 *      OpenClStream::instance()->getOrCreateKernel("kernelName", kernelCode);
 * addArgs(kernel, inputPtr.getAsClMem(), &h, &w);
 * OpenClStream::instance()->enqueue(kernel, {h * w});
 */
class OpenClStream {
 public:
  static OpenClStream* instance();

  /**
   * Returns existing kernel if one is found by that name. Otherwise, building
   * the kernel from source and caches that kernel by name for future calls.
   */
  cl_kernel getOrCreateKernel(const std::string& name, const char* source);

  /**
   * globalWorkSize specifies the number of threads in each dimension. OpenCL
   * supports 1,2, and 3 dimension.
   */
  void enqueue(cl_kernel kernel, const std::vector<size_t>& globalWorkSize);

 private:
  OpenClStream();

  static std::unique_ptr<OpenClStream> instance_;
  cl_context context_;
  cl_device_id deviceId_;
  cl_command_queue queue_;
  std::unordered_map<std::string, cl_kernel> nameToKernel_;
};

namespace detail {

void check(cl_int err, const char* file, int line, const char* cmd);

template <class First>
inline void addArgsRecurse(cl_kernel& kernel, int i, const First& first) {
  FL_OPENCL_CHECK(clSetKernelArg(kernel, i, sizeof(*first), first));
}

template <class First, class... Rest>
inline void addArgsRecurse(
    cl_kernel& kernel,
    int i,
    const First& first,
    const Rest&... rest) {
  FL_OPENCL_CHECK(clSetKernelArg(kernel, i, sizeof(*first), first));
  addArgsRecurse(kernel, i + 1, rest...);
}

} // namespace detail

/**
 * Utility for adding parameters to a kernel in a single instruction that looks
 * like a simple function call. Support's any type and any number of opencl
 * kernel args. See use example in the OpenClStream doc.
 */
template <class... Args>
void addArgs(cl_kernel& kernel, const Args&... args) {
  detail::addArgsRecurse(kernel, 0, args...);
}

} // namespace ocl
} // namespace fl
