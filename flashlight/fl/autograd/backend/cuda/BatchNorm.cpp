/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include <cudnn.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

namespace fl {

Variable batchnorm(
    const Variable& in,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon) {
  auto input = FL_ADJUST_INPUT_TYPE(in);

  if (input.type() == fl::dtype::f16 && weight.type() != fl::dtype::f32) {
    throw std::invalid_argument(
        "fl::batchnorm: non-input tensors must be of type f32");
  }
  FL_VARIABLE_DTYPES_MATCH_CHECK(weight, bias, runningMean, runningVar);

  auto output = Tensor(input.dims(), input.type());

  int nfeatures = 1;
  for (auto ax : axes) {
    if (ax > input.numdims() - 1) {
      throw std::invalid_argument(
          "batchnorm - passed axes (axis value " + std::to_string(ax) +
          ") exceeds the number of dimensions of the input (" +
          std::to_string(input.numdims()) + ")");
    }
    nfeatures *= input.dims(ax);
  }

  cudnnBatchNormMode_t mode;
  Shape inDescDims, wtDescDims;

  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());

  // assuming no duplicates
  bool axes_continuous = (axes.size() == (max_axis - min_axis + 1));
  if (!axes_continuous) {
    throw std::invalid_argument("unsupported axis config for cuDNN batchnorm");
  }

  if (min_axis == 0) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
    inDescDims = Shape({1, 1, nfeatures, input.elements() / nfeatures});
    wtDescDims = Shape({1, 1, nfeatures});
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
    inDescDims = Shape(
        {1, input.elements() / (nfeatures * batchsz), nfeatures, batchsz});
    wtDescDims = Shape({1, 1, nfeatures});
  }

  if (!weight.isempty() && weight.elements() != wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for weight.");
  }

  if (!bias.isempty() && bias.elements() != wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for bias.");
  }
  // Weight, bias, and running mean/var arrays can't be fp16 (must be fp32)
  Tensor weightArray = weight.isempty()
      ? fl::full(wtDescDims, 1.0, fl::dtype::f32)
      : weight.tensor().astype(fl::dtype::f32);
  Tensor biasArray = bias.isempty() ? fl::full(wtDescDims, 0.0, fl::dtype::f32)
                                    : bias.tensor().astype(fl::dtype::f32);

  fl::dtype scalarsType =
      input.type() == fl::dtype::f16 ? fl::dtype::f32 : input.type();

  auto inDesc = TensorDescriptor(input.type(), inDescDims);
  auto wtDesc = TensorDescriptor(weightArray.type(), wtDescDims);

  Tensor saveMean, saveVar;
  {
    DevicePtr inRaw(input.tensor());
    DevicePtr outRaw(output);
    DevicePtr wtRaw(weightArray);
    DevicePtr bsRaw(biasArray);
    DevicePtr runMeanRaw(runningMean.tensor());
    DevicePtr runVarRaw(runningVar.tensor());

    if (train) {
      saveMean = Tensor({nfeatures}, scalarsType);
      saveVar = Tensor({nfeatures}, scalarsType);

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
          std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!train) {
          throw std::logic_error(
              "can't compute batchnorm grad when train was not specified");
        }

        auto& in = inputs[0];
        auto inTensor = detail::adjustInputType(in.tensor(), "batchnorm");
        auto gradOutputArray =
            detail::adjustInputType(gradOutput.tensor(), "batchnorm");
        // Weight, bias, and running mean/var arrays can't be fp16 (must be
        // fp32)
        auto wt = inputs[1].isempty()
            ? Variable(fl::full(wtDescDims, 1.0, fl::dtype::f32), false)
            : inputs[1];
        auto& bs = inputs[2];

        auto scalarsType = inTensor.type() == fl::dtype::f16 ? fl::dtype::f32
                                                             : inTensor.type();
        const void* one1 = kOne(scalarsType);
        const void* zero0 = kZero(scalarsType);

        auto iDesc = TensorDescriptor(inTensor.type(), inDescDims);
        auto wDesc = TensorDescriptor(wt.type(), wtDescDims);
        // CuDNN doesn't support calculating only the gradients
        // required for batchnorm
        auto gradIn = Tensor(inTensor.shape(), inTensor.type());
        auto gradWt = Tensor(wt.dims(), wt.type());
        auto gradBs = Tensor(wt.dims(), wt.type());
        {
          DevicePtr iRaw(inTensor);
          DevicePtr wRaw(wt.tensor());

          DevicePtr gradInRaw(gradIn);
          DevicePtr gradWtRaw(gradWt);
          DevicePtr gradBsRaw(gradBs);

          DevicePtr gradOpRaw(gradOutputArray);

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
              gradOpRaw.get(),
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
        in.addGrad(Variable(gradIn.astype(in.type()), false));
        wt.addGrad(Variable(gradWt.astype(wt.type()), false));
        if (!bs.isempty()) {
          bs.addGrad(Variable(gradBs.astype(bs.type()), false));
        }
      };
  return Variable(output, {in, weight, bias}, gradFunc);
}

} // namespace fl
