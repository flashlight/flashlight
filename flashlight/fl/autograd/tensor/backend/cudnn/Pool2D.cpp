/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {

Tensor CudnnAutogradExtension::pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode,
    std::shared_ptr<detail::AutogradPayload>) {
  auto inDesc = TensorDescriptor(input);

  // init pooling descriptor
  auto poolDesc = PoolingDescriptor(wx, wy, sx, sy, px, py, mode);

  // init output descriptor
  auto ix = input.dim(0);
  auto iy = input.ndim() < 2 ? 1 : input.dim(1);
  auto ox = 1 + (ix + 2 * px - wx) / sx;
  auto oy = 1 + (iy + 2 * py - wy) / sy;

  auto output = Tensor(
      {ox,
       oy,
       input.ndim() < 3 ? 1 : input.dim(2),
       input.ndim() < 4 ? 1 : input.dim(3)},
      input.type());
  auto outDesc = TensorDescriptor(output);
  {
    DevicePtr inputraw(input);
    DevicePtr resultraw(output);
    const auto& cudnnStream = getCudnnStream();
    // ensure cudnn compute stream waits on streams of input/output tensors
    relativeSync(cudnnStream, {input, output});

    auto handle = getCudnnHandle();
    const void* one = kOne(input.type());
    const void* zero = kZero(input.type());

    CUDNN_CHECK_ERR(cudnnPoolingForward(
        handle,
        poolDesc.descriptor,
        one,
        inDesc.descriptor,
        inputraw.get(),
        zero,
        outDesc.descriptor,
        resultraw.get()));

    // ensure output tensor stream waits on cudnn compute stream
    relativeSync({output}, cudnnStream);
  }

  return output;
}

Tensor CudnnAutogradExtension::pool2dBackward(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& poolOutput,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode,
    std::shared_ptr<detail::AutogradPayload>) {
  auto i_desc = TensorDescriptor(input);
  auto o_desc = TensorDescriptor(poolOutput);
  auto p_desc = PoolingDescriptor(wx, wy, sx, sy, px, py, mode);

  auto gradInput = Tensor(input.shape(), input.type());

  auto hndl = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();
  const void* oneg = kOne(input.type());
  const void* zerog = kZero(input.type());

  {
    DevicePtr inraw(input);
    DevicePtr outraw(poolOutput);
    DevicePtr gradresultraw(gradOutput);
    DevicePtr gradinputraw(gradInput);
    // ensure cudnn compute stream waits on input/output tensor streams
    relativeSync(cudnnStream, {input, poolOutput, gradOutput, gradInput});

    CUDNN_CHECK_ERR(cudnnPoolingBackward(
        hndl,
        p_desc.descriptor,
        oneg,
        o_desc.descriptor,
        outraw.get(),
        o_desc.descriptor,
        gradresultraw.get(),
        i_desc.descriptor,
        inraw.get(),
        zerog,
        i_desc.descriptor,
        gradinputraw.get()));
    // ensure gradient input tensor stream waits on cudnn compute stream
    relativeSync({gradInput}, cudnnStream);
  }

  return gradInput;
}

} // namespace fl
