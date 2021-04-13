/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Utils.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/miopen/MiOpenUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/backend/miopen/MiOpenUtils.h"

using namespace ::fl::miopen;

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
  size_t workSpaceSize = 0;
  MIOPEN_CHECK_ERR(miopenPoolingGetWorkSpaceSizeV2(
      pool_desc.descriptor, out_desc.descriptor, &workSpaceSize));
  workSpaceSize = std::max(workSpaceSize, sizeof(float));
  auto wspace = af::array(workSpaceSize, af::dtype::b8);
  {
    DevicePtr inputraw(input.array());
    DevicePtr resultraw(output);
    DevicePtr wspacePtr(wspace);

    const void* one = kOne(input.type());
    const void* zero = kZero(input.type());

    MIOPEN_CHECK_ERR(miopenPoolingForward(
        getMiOpenHandle(),
        pool_desc.descriptor,
        /* alpha= */ one,
        in_desc.descriptor,
        inputraw.get(),
        /* beta = */ zero,
        out_desc.descriptor,
        resultraw.get(),
        /* do_backward= */ true,
        wspacePtr.get(),
        workSpaceSize));
  }
  auto gradFunc = [wx, wy, sx, sy, px, py, mode, output, wspace, workSpaceSize](
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

    auto hndl = getMiOpenHandle();
    const void* oneg = kOne(in.type());
    const void* zerog = kZero(in.type());

    {
      DevicePtr inraw(in.array());
      DevicePtr outraw(output);
      DevicePtr gradresultraw(grad_output.array());
      DevicePtr gradinputraw(grad_input.array());
      DevicePtr wspacePtr(wspace);

      MIOPEN_CHECK_ERR(miopenPoolingBackward(
          hndl,
          p_desc.descriptor,
          /* alpha= */ oneg,
          o_desc.descriptor,
          outraw.get(),
          o_desc.descriptor,
          gradresultraw.get(),
          i_desc.descriptor,
          inraw.get(),
          /* beta = */ zerog,
          i_desc.descriptor,
          gradinputraw.get(),
          wspacePtr.get()));
    }
    in.addGrad(grad_input);
  };
  return Variable(output, {input}, gradFunc);
}

} // namespace fl
