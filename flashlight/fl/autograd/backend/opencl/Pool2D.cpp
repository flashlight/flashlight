/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unordered_map>
#include <vector>

#include <CL/cl2.hpp>
#include <af/opencl.h>
#include <arrayfire.h>

#include "opencl_kernels/Pool2D_cl.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Utils.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/OpenClUtils.h"

namespace {
#include <type_traits>

#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable
#define __CL_ENABLE_EXCEPTIONS

using namespace ::fl::ocl;

constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

void maxPoolFwd(
    const af::array& input,
    af::array& output,
    af::array& index,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py) {
  static cl_kernel kernel = OpenClStream::instance()->getOrCreateKernel(
      "max_pool_fwd", opencl::Pool2D_cl);

  DevicePtrOpenCl inputPtr(input);
  DevicePtrOpenCl outputPtr(output);
  DevicePtrOpenCl indexPtr(index);

  int ix = input.dims(0);
  int iy = input.dims(1);
  int C = input.dims(2);
  int N = input.dims(3);
  const int ox = 1 + (ix + 2 * px - wx) / sx;
  const int oy = 1 + (iy + 2 * py - wy) / sy;

  addArgs(
      kernel,
      inputPtr.getAsClMem(),
      outputPtr.getAsClMem(),
      indexPtr.getAsClMem(),
      &ix,
      &iy,
      &C,
      &N,
      &ox,
      &oy,
      &wx,
      &wy,
      &sx,
      &sy,
      &px,
      &py);

  size_t nWins = output.dims(kWIdx) * output.dims(kHIdx);

  const std::vector<size_t> globalWorkSize = {
      nWins, static_cast<size_t>(C), static_cast<size_t>(N)};
  OpenClStream::instance()->enqueue(kernel, globalWorkSize);
}

void maxPoolBwd(
    const af::array& output,
    const af::array& index,
    af::array& grad) {
  static cl_kernel kernel = OpenClStream::instance()->getOrCreateKernel(
      "max_pool_bwd", opencl::Pool2D_cl);

  DevicePtrOpenCl gradPtr(grad);
  DevicePtrOpenCl outputPtr(output);
  DevicePtrOpenCl indexPtr(index);

  addArgs(
      kernel,
      gradPtr.getAsClMem(),
      outputPtr.getAsClMem(),
      indexPtr.getAsClMem());

  const std::vector<size_t> globalWorkSize = {
      static_cast<size_t>(output.elements())};
  OpenClStream::instance()->enqueue(kernel, globalWorkSize);
}

} // namespace

namespace fl {
// Input, output: WHCN; weights: WHCN
Variable pool2d(
    const Variable& input,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode /* = PoolingMode::MAX */) {
  if (mode != PoolingMode::MAX) {
    throw std::runtime_error("pool2d unsupported mode");
  }
  // padding should be smaller than window
  px %= wx;
  py %= wy;

  auto ix = input.dims(kWIdx);
  auto iy = input.dims(kHIdx);
  auto ox = 1 + (ix + 2 * px - wx) / sx;
  auto oy = 1 + (iy + 2 * py - wy) / sy;
  af::dim4 outDims = {
      ox, oy, input.dims(kChannelSizeIdx), input.dims(kBatchSizeIdx)};

  auto index = af::constant(0, outDims, u32);
  auto output = af::constant(0, outDims, input.type());

  maxPoolFwd(input.array(), output, index, wx, wy, sx, sy, px, py);

  auto gradFunc = [index, output](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto& in = inputs[0];
    if (!in.isCalcGrad()) {
      return;
    }
    auto grad = af::constant(0, in.dims());
    maxPoolBwd(gradOutput.array(), index, grad);
    in.addGrad(Variable(grad, false));
  };

  return fl::Variable(output, {input}, gradFunc);
} // namespace fl

} // namespace fl
