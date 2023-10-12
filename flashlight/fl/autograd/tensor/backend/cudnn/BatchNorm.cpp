/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"

#include <algorithm>
#include <stdexcept>

#include <cudnn.h>

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {
namespace {

void getBatchnormMetadata(
    cudnnBatchNormMode_t& modeOut,
    Shape& inDescDimsOut,
    Shape& wtDescDimsOut,
    const Tensor& input,
    const std::vector<int>& axes,
    const bool train) {
  int nfeatures = 1;
  for (auto ax : axes) {
    if (ax > input.ndim() - 1) {
      throw std::invalid_argument(
          "batchnorm - passed axes (axis value " + std::to_string(ax) +
          ") exceeds the number of dimensions of the input (" +
          std::to_string(input.ndim()) + ")");
    }
    nfeatures *= input.dim(ax);
  }

  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());

  // assuming no duplicates
  bool axes_continuous = (axes.size() == (maxAxis - minAxis + 1));
  if (!axes_continuous) {
    throw std::invalid_argument("unsupported axis config for cuDNN batchnorm");
  }

  if (minAxis == 0) {
    modeOut = CUDNN_BATCHNORM_PER_ACTIVATION;
    inDescDimsOut = Shape(
        {1,
         1,
         nfeatures,
         static_cast<long long>(input.elements() / nfeatures)});
    wtDescDimsOut = Shape({1, 1, nfeatures});
  } else {
    modeOut = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7003
    if (train) {
      modeOut = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    }
#endif
    int batchsz = 1;
    for (int i = maxAxis + 1; i < input.ndim(); ++i) {
      batchsz *= input.dim(i);
    }
    inDescDimsOut = Shape(
        {1,
         static_cast<long long>(input.elements() / (nfeatures * batchsz)),
         nfeatures,
         batchsz});
    wtDescDimsOut = Shape({1, 1, nfeatures});
  }
}

} // namespace

Tensor CudnnAutogradExtension::batchnorm(
    Tensor& saveMean,
    Tensor& saveVar,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& runningMean,
    Tensor& runningVar,
    const std::vector<int>& axes,
    const bool train,
    const double momentum,
    const double epsilon,
    std::shared_ptr<detail::AutogradPayload>) {
  if (input.type() == fl::dtype::f16 && weight.type() != fl::dtype::f32) {
    throw std::invalid_argument(
        "fl::batchnorm: non-input tensors must be of type f32");
  }
  FL_TENSOR_DTYPES_MATCH_CHECK(weight, bias, runningMean, runningVar);

  auto output = Tensor(input.shape(), input.type());

  cudnnBatchNormMode_t mode;
  Shape inDescDims, wtDescDims;
  getBatchnormMetadata(mode, inDescDims, wtDescDims, input, axes, train);

  if (!weight.isEmpty() && weight.elements() != wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for weight.");
  }

  if (!bias.isEmpty() && bias.elements() != wtDescDims.elements()) {
    throw std::invalid_argument("[BatchNorm] Invalid shape for bias.");
  }
  // Weight, bias, and running mean/var arrays can't be fp16 (must be fp32)
  Tensor weightArray = weight.isEmpty()
      ? fl::full(wtDescDims, 1.0, fl::dtype::f32)
      : weight.astype(fl::dtype::f32);
  Tensor biasArray = bias.isEmpty() ? fl::full(wtDescDims, 0.0, fl::dtype::f32)
                                    : bias.astype(fl::dtype::f32);

  fl::dtype scalarsType =
      input.type() == fl::dtype::f16 ? fl::dtype::f32 : input.type();

  auto inDesc = TensorDescriptor(input.type(), inDescDims);
  auto wtDesc = TensorDescriptor(weightArray.type(), wtDescDims);

  {
    DevicePtr inRaw(input);
    DevicePtr outRaw(output);
    DevicePtr wtRaw(weightArray);
    DevicePtr bsRaw(biasArray);
    DevicePtr runMeanRaw(runningMean);
    DevicePtr runVarRaw(runningVar);
    const auto& cudnnStream = getCudnnStream();
    // ensure cudnn compute stream waits on streams of input/output tensors
    relativeSync(
        cudnnStream,
        {input, output, weightArray, biasArray, runningMean, runningVar});

    if (train) {
      saveMean = Tensor({wtDescDims[2]}, scalarsType);
      saveVar = Tensor({wtDescDims[2]}, scalarsType);

      DevicePtr saveMeanRaw(saveMean);
      DevicePtr saveVarRaw(saveVar);
      // ensure cudnn compute stream waits on streams of saveMean/Var tensors
      relativeSync(cudnnStream, {saveMean, saveVar});
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
    // ensure output stream waits on cudnn compute stream
    relativeSync({output}, cudnnStream);
  }
  return output;
}

std::tuple<Tensor, Tensor, Tensor> CudnnAutogradExtension::batchnormBackward(
    const Tensor& gradOutput,
    const Tensor& saveMean,
    const Tensor& saveVar,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<int>& axes,
    const bool train, // TODO(jacobkahn): remove this arg
    const float epsilon,
    std::shared_ptr<detail::AutogradPayload>) {
  if (!train) {
    throw std::logic_error(
        "can't compute batchnorm grad when train was not specified");
  }

  cudnnBatchNormMode_t mode;
  Shape inDescDims, wtDescDims;
  getBatchnormMetadata(mode, inDescDims, wtDescDims, input, axes, train);

  auto wt =
      weight.isEmpty() ? fl::full(wtDescDims, 1.0, fl::dtype::f32) : weight;

  // Weight, bias, and running mean/var arrays can't be fp16 (must be
  // fp32)
  auto scalarsType =
      input.type() == fl::dtype::f16 ? fl::dtype::f32 : input.type();
  const void* one1 = kOne(scalarsType);
  const void* zero0 = kZero(scalarsType);

  auto iDesc = TensorDescriptor(input.type(), inDescDims);
  auto wDesc = TensorDescriptor(wt.type(), wtDescDims);
  // CuDNN doesn't support calculating only the gradients
  // required for batchnorm
  auto gradIn = Tensor(input.shape(), input.type());
  auto gradWt = Tensor(wt.shape(), wt.type());
  auto gradBs = Tensor(wt.shape(), wt.type());
  {
    DevicePtr iRaw(input);
    DevicePtr wRaw(wt);

    DevicePtr gradInRaw(gradIn);
    DevicePtr gradWtRaw(gradWt);
    DevicePtr gradBsRaw(gradBs);

    DevicePtr gradOpRaw(gradOutput);

    DevicePtr saveMeanRaw(saveMean);
    DevicePtr saveVarRaw(saveVar);
    const auto& cudnnStream = getCudnnStream();
    // ensure cudnn compute stream waits on streams of input/output tensors
    relativeSync(
        cudnnStream,
        {input, gradOutput, gradIn, wt, gradWt, gradBs, saveMean, saveVar});

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
    // ensure streams of gradients wait on the cudnn compute stream
    relativeSync({gradIn, gradWt, gradBs}, cudnnStream);
  }

  return std::make_tuple(gradIn, gradWt, gradBs);
}

} // namespace fl
