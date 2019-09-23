/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include <cudnn.h>

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/common/DevicePtr.h"
#include "flashlight/nn/Utils.h"

namespace fl {

Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& running_mean,
    Variable& running_var,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon) {
  typeTrace("Batchnorm FWD - input", input.type());

  auto output = af::array(input.dims(), input.type());

  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= input.dims(ax);
  }

  cudnnBatchNormMode_t mode;
  af::dim4 in_desc_dims, wt_desc_dims;

  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());

  // assuming no duplicates
  bool axes_continuous = (axes.size() == (max_axis - min_axis + 1));
  if (!axes_continuous) {
    throw std::invalid_argument("unsupported axis config for cuDNN batchnorm");
  }

  if (min_axis == 0) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    in_desc_dims = af::dim4(1, 1, nfeatures, input.elements() / nfeatures);
    wt_desc_dims = af::dim4(1, 1, nfeatures);
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7003
    if (train) {
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    }
#endif
    int batchsz = 1;
    for (int i = max_axis + 1; i < 4; ++i) {
      batchsz *= input.dims(i);
    }
    in_desc_dims = af::dim4(
        1, input.elements() / (nfeatures * batchsz), nfeatures, batchsz);
    wt_desc_dims = af::dim4(1, 1, nfeatures);
  }
  auto weightNonempty = weight.isempty()
      ? Variable(af::constant(1.0, wt_desc_dims, input.type()), false)
      : weight;
  auto biasNonempty = bias.isempty()
      ? Variable(af::constant(0.0, wt_desc_dims, input.type()), false)
      : bias;

  af::array weightArray;
  if (weightNonempty.type() == f32) {
    weightArray = weightNonempty.array();
  } else {
    weightArray = weightNonempty.array().as(f32);
  }

  af::array biasArray;
  if (biasNonempty.type() == f32) {
    biasArray = biasNonempty.array();
  } else {
    biasArray = biasNonempty.array().as(f32);
  }

  auto scalarsType = input.type() == f16 ? f32 : input.type();
  const void* one = kOne(scalarsType);
  const void* zero = kZero(scalarsType);

  auto in_desc = TensorDescriptor(input.type(), in_desc_dims);
  auto wt_desc = TensorDescriptor(weightArray.type(), wt_desc_dims);

  af::array save_mean, save_var;
  {
    DevicePtr in_raw(input.array());
    DevicePtr out_raw(output);
    DevicePtr wt_raw(weightArray);
    DevicePtr bs_raw(biasArray);
    DevicePtr run_mean_raw(running_mean.array());
    DevicePtr run_var_raw(running_var.array());

    if (train) {
      save_mean = af::array(nfeatures, input.type());
      save_var = af::array(nfeatures, input.type());

      DevicePtr save_mean_raw(save_mean);
      DevicePtr save_var_raw(save_var);
      CUDNN_CHECK_ERR(cudnnBatchNormalizationForwardTraining(
          getCudnnHandle(),
          mode,
          one,
          zero,
          in_desc.descriptor,
          in_raw.get(),
          in_desc.descriptor,
          out_raw.get(),
          wt_desc.descriptor,
          wt_raw.get(),
          bs_raw.get(),
          momentum,
          run_mean_raw.get(),
          run_var_raw.get(),
          epsilon,
          save_mean_raw.get(),
          save_var_raw.get()));
    } else {
      CUDNN_CHECK_ERR(cudnnBatchNormalizationForwardInference(
          getCudnnHandle(),
          mode,
          one,
          zero,
          in_desc.descriptor,
          in_raw.get(),
          in_desc.descriptor,
          out_raw.get(),
          wt_desc.descriptor,
          wt_raw.get(),
          bs_raw.get(),
          run_mean_raw.get(),
          run_var_raw.get(),
          epsilon));
    }
  }
  auto gradFunc =
      [train, save_mean, save_var, mode, in_desc_dims, wt_desc_dims, epsilon](
          std::vector<Variable>& inputs, const Variable& grad_output) {
        typeTrace("Batchnorm BWD - upstream grad", grad_output.type());
        typeTrace("Batchnorm BWD - input", inputs[0].type());
        if (!train) {
          throw std::logic_error(
              "can't compute batchnorm grad when train was not specified");
        }

        auto& in = inputs[0];
        auto wt = inputs[1].isempty()
            ? Variable(af::constant(1.0, wt_desc_dims, in.type()), false)
            : inputs[1];
        auto& bs = inputs[2];

        af::array grad_output_array;
        if (grad_output.type() == in.type()) {
          grad_output_array = grad_output.array();
        } else {
          grad_output_array = grad_output.array().as(in.type());
        }

        af::array wt_array;
        if (wt.type() == f32) {
          wt_array = wt.array();
        } else {
          wt_array = wt.array().as(f32);
        }

        auto scalarsType = in.type() == f16 ? f32 : in.type();
        const void* one1 = kOne(scalarsType);
        const void* zero0 = kZero(scalarsType);

        auto i_desc = TensorDescriptor(in.type(), in_desc_dims);
        auto w_desc = TensorDescriptor(wt_array.type(), wt_desc_dims);
        // CuDNN doesn't support calculating only the gradients
        // required for batchnorm
        auto grad_in = Variable(af::array(in.dims(), in.type()), false);
        auto grad_wt =
            Variable(af::array(wt_array.dims(), wt_array.type()), false);
        auto grad_bs =
            Variable(af::array(wt_array.dims(), wt_array.type()), false);
        {
          DevicePtr i_raw(in.array());
          DevicePtr w_raw(wt_array);

          DevicePtr grad_in_raw(grad_in.array());
          DevicePtr grad_wt_raw(grad_wt.array());
          DevicePtr grad_bs_raw(grad_bs.array());

          DevicePtr grad_op_raw(grad_output_array);

          DevicePtr save_mean_raw(save_mean);
          DevicePtr save_var_raw(save_var);

          CUDNN_CHECK_ERR(cudnnBatchNormalizationBackward(
              getCudnnHandle(),
              mode,
              one1,
              zero0,
              one1,
              zero0,
              i_desc.descriptor,
              i_raw.get(),
              i_desc.descriptor,
              grad_op_raw.get(),
              i_desc.descriptor,
              grad_in_raw.get(),
              w_desc.descriptor,
              w_raw.get(),
              grad_wt_raw.get(),
              grad_bs_raw.get(),
              epsilon,
              save_mean_raw.get(),
              save_var_raw.get()));
        }
        in.addGrad(grad_in);
        wt.addGrad(grad_wt);
        if (!bs.isempty()) {
          bs.addGrad(grad_bs);
        }
      };
  return Variable(output, {input, weight, bias}, gradFunc);
}

} // namespace fl
