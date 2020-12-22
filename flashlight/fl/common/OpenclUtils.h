/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <CL/cl2.hpp>
#include <af/array.h>
#include <af/opencl.h>
#include "flashlight/fl/common/DevicePtr.h"

#define FL_OPENCL_CHECK(err) \
  ::fl::opencl::detail::check(err, __FILE__, __LINE__, #err)

namespace fl {
namespace opencl {
namespace detail {

void check(cl_int err, const char* file, int line, const char* cmd);

class DevicePtrOpenCl : public fl::DevicePtr {
 public:
  DevicePtrOpenCl(const af::array& in) : DevicePtr(in) {
    clMemBuf_ = in.device<cl_mem>();
  }

  ~DevicePtrOpenCl() {
    delete clMemBuf_;
  }

  cl_mem* getAsClMem() {
    return clMemBuf_;
  }

 private:
  cl_mem* clMemBuf_;
};

/**
 *  Singleton warpper to a single opencl context. Used for creating and
 * enqueuing opencl kernel in that context.
 */
class OpenClContext {
 public:
  static OpenClContext* instance();

  /**
   * Returns existing kernel if one is found by that name. Otherwise, building
   * the kernel from source and caches the kernell by name for future calls.
   * When size==0 source is assumed to be null terminated.
   */
  cl_kernel getOrCreateKernel(const std::string& name, const char* source);

  /**
   * globalWorkSize specifies the size in each dimension. OpenCL support 1..3
   * dimension.
   */
  void enqueue(cl_kernel kernel, const std::vector<size_t>& globalWorkSize);

 private:
  OpenClContext();

  static std::unique_ptr<OpenClContext> instance_;
  cl_context context_;
  cl_device_id deviceId_;
  cl_command_queue queue_;
  std::unordered_map<std::string, cl_kernel> nameToKernel_;
};

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

template <class... Args>
void addArgs(cl_kernel& kernel, const Args&... args) {
  addArgsRecurse(kernel, 0, args...);
}

} // namespace detail
} // namespace opencl
} // namespace fl
