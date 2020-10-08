/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include <cudnn.h>

#include "flashlight/flashlight/autograd/Functions.h"
#include "flashlight/flashlight/autograd/Variable.h"
#include "flashlight/flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/flashlight/common/DevicePtr.h"

namespace fl {

Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon) {
  if (input.type() == af::dtype::f16 && weight.type() != af::dtype::f32) {
    throw std::invalid_argument(
        "fl::batchnorm: non-input tensors must be of type f32");
  }
  FL_VARIABLE_DTYPES_MATCH_CHECK(weight, bias, runningMean, runningVar);

  auto output = af::array(input.dims(), input.type());

  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= input.dims(ax);
  }

  cudnnBatchNormMode_t mode;
  af::dim4 inDescDims, wtDescDims;

  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());

  // assuming no duplicates
  bool axes_continuous = (axes.size() == (max_axis - min_axis + 1));
  if (!axes_continuous) {
    throw std::invalid_argument("unsupported axis config for cuDNN batchnorm");
  }

  if (min_axis == 0) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    inDescDims = af::dim4(1, 1, nfeatures, input.elements() / nfeatures);
    wtDescDims = af::dim4(1, 1, nfeatures);
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
    inDescDims = af::dim4(
        1, input.elements() / (nfeatures * batchsz), nfeatures, batchsz);
    wtDescDims = af::dim4(1, 1, nfeatures);
  }

  if (!weight.isempty() && weight.elements() != wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for weight.");
  }

  if (!bias.isempty() && bias.elements() != wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for bias.");
  }
  // Weight, bias, and running mean/var arrays can't be fp16 (must be fp32)
  af::array weightArray = weight.isempty()
      ? af::constant(1.0, wtDescDims, af::dtype::f32)
      : weight.array().as(af::dtype::f32);
  af::array biasArray = bias.isempty()
      ? af::constant(0.0, wtDescDims, af::dtype::f32)
      : bias.array().as(af::dtype::f32);

  auto scalarsType = input.type() == f16 ? f32 : input.type();

  auto inDesc = TensorDescriptor(input.type(), inDescDims);
  auto wtDesc = TensorDescriptor(weightArray.type(), wtDescDims);

  af::array saveMean, saveVar;
  {
    DevicePtr inRaw(input.array());
    DevicePtr outRaw(output);
    DevicePtr wtRaw(weightArray);
    DevicePtr bsRaw(biasArray);
    DevicePtr runMeanRaw(runningMean.array());
    DevicePtr runVarRaw(runningVar.array());

    if (train) {
      saveMean = af::array(nfeatures, input.type());
      saveVar = af::array(nfeatures, input.type());

      DevicePtr saveMeanRaw(saveMean);
      DevicePtr saveVarRaw(saveVar);
      CUDNN_CHECK_ERR(cudnnBatchNormalizationForwardTraining(
          getCudnnHandle(),
          mode,
          kOne(scalarsType),
          kZero(scalarsType),
          inDesc.descriptor,
          inRaw.get(),
          inDesc.descriptor,
          outRaw.get(),
          wtDesc.descriptor,
          wtRaw.get(),
          bsRaw.get(),
          momentum,
          runMeanRaw.get(),
          runVarRaw.get(),
          epsilon,
          saveMeanRaw.get(),
          saveVarRaw.get()));
    } else {
      CUDNN_CHECK_ERR(cudnnBatchNormalizationForwardInference(
          getCudnnHandle(),
          mode,
          kOne(scalarsType),
          kZero(scalarsType),
          inDesc.descriptor,
          inRaw.get(),
          inDesc.descriptor,
          outRaw.get(),
          wtDesc.descriptor,
          wtRaw.get(),
          bsRaw.get(),
          runMeanRaw.get(),
          runVarRaw.get(),
          epsilon));
    }
  }
  auto gradFunc =
      [train, saveMean, saveVar, mode, inDescDims, wtDescDims, epsilon](
          std::vector<Variable>& inputs, const Variable& grad_output) {
        if (!train) {
          throw std::logic_error(
              "can't compute batchnorm grad when train was not specified");
        }

        auto& in = inputs[0];
        // Weight, bias, and running mean/var arrays can't be fp16 (must be
        // fp32)
        auto wt = inputs[1].isempty()
            ? Variable(af::constant(1.0, wtDescDims, af::dtype::f32), false)
            : inputs[1];
        auto& bs = inputs[2];

        auto scalarsType = in.type() == f16 ? f32 : in.type();
        const void* one1 = kOne(scalarsType);
        const void* zero0 = kZero(scalarsType);

        auto iDesc = TensorDescriptor(in.type(), inDescDims);
        auto wDesc = TensorDescriptor(wt.type(), wtDescDims);
        // CuDNN doesn't support calculating only the gradients
        // required for batchnorm
        auto gradIn = af::array(in.dims(), in.type());
        auto gradWt = af::array(wt.dims(), wt.type());
        auto gradBs = af::array(wt.dims(), wt.type());
        {
          DevicePtr iRaw(in.array());
          DevicePtr wRaw(wt.array());

          DevicePtr gradInRaw(gradIn);
          DevicePtr gradWtRaw(gradWt);
          DevicePtr gradBsRaw(gradBs);

          DevicePtr grad_opRaw(grad_output.array());

          DevicePtr saveMeanRaw(saveMean);
          DevicePtr saveVarRaw(saveVar);

          CUDNN_CHECK_ERR(cudnnBatchNormalizationBackward(
              getCudnnHandle(),
              mode,
              one1,
              zero0,
              one1,
              zero0,
              iDesc.descriptor,
              iRaw.get(),
              iDesc.descriptor,
              grad_opRaw.get(),
              iDesc.descriptor,
              gradInRaw.get(),
              wDesc.descriptor,
              wRaw.get(),
              gradWtRaw.get(),
              gradBsRaw.get(),
              epsilon,
              saveMeanRaw.get(),
              saveVarRaw.get()));
        }
        in.addGrad(Variable(gradIn, false));
        wt.addGrad(Variable(gradWt.as(wt.type()), false));
        if (!bs.isempty()) {
          bs.addGrad(Variable(gradBs.as(bs.type()), false));
        }
      };
  return Variable(output, {input, weight, bias}, gradFunc);
}

} // namespace fl
