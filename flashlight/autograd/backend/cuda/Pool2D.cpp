/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/autograd/Functions.h"

#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/common/DevicePtr.h"

namespace fl {

Variable pool2d(
    const Variable& input,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode /* = PoolingMode::MAX */) {
  auto in_desc = TensorDescriptor(input);

  // init pooling descriptor
  auto pool_desc = PoolingDescriptor(wx, wy, sx, sy, px, py, mode);

  // init output descriptor
  auto ix = input.dims(0);
  auto iy = input.dims(1);
  auto ox = 1 + (ix + 2 * px - wx) / sx;
  auto oy = 1 + (iy + 2 * py - wy) / sy;

  auto output = af::randu(ox, oy, input.dims(2), input.dims(3), input.type());
  auto out_desc = TensorDescriptor(output);
  {
    DevicePtr inputraw(input.array());
    DevicePtr resultraw(output);

    auto handle = getCudnnHandle();
    const void* one = kOne(input.type());
    const void* zero = kZero(input.type());

    CUDNN_CHECK_ERR(cudnnPoolingForward(
        handle,
        pool_desc.descriptor,
        one,
        in_desc.descriptor,
        inputraw.get(),
        zero,
        out_desc.descriptor,
        resultraw.get()));
  }
  auto gradFunc = [wx, wy, sx, sy, px, py, mode, output](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto& in = inputs[0];
    if (!in.isCalcGrad()) {
      return;
    }
    auto i_desc = TensorDescriptor(in);
    auto o_desc = TensorDescriptor(output);
    auto p_desc = PoolingDescriptor(wx, wy, sx, sy, px, py, mode);

    auto grad_input = Variable(af::array(in.dims(), in.type()), false);

    auto hndl = getCudnnHandle();
    const void* oneg = kOne(in.type());
    const void* zerog = kZero(in.type());

    {
      DevicePtr inraw(in.array());
      DevicePtr outraw(output);
      DevicePtr gradresultraw(grad_output.array());
      DevicePtr gradinputraw(grad_input.array());

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
    }
    in.addGrad(grad_input);
  };
  return Variable(output, {input}, gradFunc);
}

} // namespace fl
