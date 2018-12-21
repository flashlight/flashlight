/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/autograd/Functions.h>

#include <algorithm>

#include <cudnn.h>

#include <flashlight/autograd/Variable.h>
#include <flashlight/common/DevicePtr.h>
#include "CudnnUtils.h"

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
  auto output = af::array(input.dims(), input.type());

  intl nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= input.dims(ax);
  }

  cudnnBatchNormMode_t mode;
  af::dim4 in_desc_dims, wt_desc_dims;

  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());

  // assuming no duplicates
  bool axes_continuous = (axes.size() == (max_axis - min_axis + 1));

  if (axes_continuous && (min_axis == 0)) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    in_desc_dims = af::dim4(1, 1, nfeatures, input.elements() / nfeatures);
    wt_desc_dims = af::dim4(1, 1, nfeatures);
  } else if (axes_continuous) {
    mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7003
    if (train) {
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    }
#endif
    intl batchsz = 1;
    for (int i = max_axis + 1; i < 4; ++i) {
      batchsz *= input.dims(i);
    }
    in_desc_dims = af::dim4(
        1, input.elements() / (nfeatures * batchsz), nfeatures, batchsz);
    wt_desc_dims = af::dim4(1, 1, nfeatures);
  } else {
    AFML_THROW_ERR("Unsupported BatchNorm config for CuDNN", AF_ERR_ARG);
  }

  const void* one = kOne(input.type());
  const void* zero = kZero(input.type());

  auto weight_nonempty = weight.isempty()
      ? Variable(af::constant(1.0, wt_desc_dims, input.type()), false)
      : weight;
  auto bias_nonempty = bias.isempty()
      ? Variable(af::constant(0.0, wt_desc_dims, input.type()), false)
      : bias;

  auto in_desc = TensorDescriptor(input.type(), in_desc_dims);
  auto wt_desc = TensorDescriptor(weight_nonempty.type(), wt_desc_dims);

  af::array save_mean, save_var;
  {
    DevicePtr in_raw(input.array());
    DevicePtr out_raw(output);
    DevicePtr wt_raw(weight_nonempty.array());
    DevicePtr bs_raw(bias_nonempty.array());
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
        AFML_ASSERT(
            train,
            "backward algorithm supported only for train",
            AF_ERR_NOT_SUPPORTED);

        auto& in = inputs[0];
        auto wt = inputs[1].isempty()
            ? Variable(af::constant(1.0, wt_desc_dims, in.type()), false)
            : inputs[1];
        auto& bs = inputs[2];

        const void* one1 = kOne(in.type());
        const void* zero0 = kZero(in.type());

        auto i_desc = TensorDescriptor(in.type(), in_desc_dims);
        auto w_desc = TensorDescriptor(wt.type(), wt_desc_dims);
        // CuDNN doesn't support calculating only the gradients
        // required for batchnorm
        auto grad_in = Variable(af::array(in.dims(), in.type()), false);
        auto grad_wt = Variable(af::array(wt.dims(), wt.type()), false);
        auto grad_bs = Variable(af::array(wt.dims(), wt.type()), false);
        {
          DevicePtr i_raw(in.array());
          DevicePtr w_raw(wt.array());

          DevicePtr grad_in_raw(grad_in.array());
          DevicePtr grad_wt_raw(grad_wt.array());
          DevicePtr grad_bs_raw(grad_bs.array());

          DevicePtr grad_op_raw(grad_output.array());

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
