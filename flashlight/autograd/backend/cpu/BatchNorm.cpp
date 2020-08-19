/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include <arrayfire.h>
#include <mkldnn.hpp>

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cpu/MkldnnUtils.h"
#include "flashlight/common/DevicePtr.h"

namespace fl {

namespace {

// Flashlight accept HWCN order according to docs
constexpr size_t kHIdx = 0;
constexpr size_t kWIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

} // namespace

Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double epsilon) {
  auto output = af::array(input.dims(), input.type());

  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= input.dims(ax);
  }

  if (runningVar.isempty()) {
    runningVar = Variable(af::constant(1.0, nfeatures, input.type()), false);
  }

  if (runningMean.isempty()) {
    runningMean = Variable(af::constant(0.0, nfeatures, input.type()), false);
  }

  // Check if axes is valid
  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (max_axis - min_axis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  auto dType = detail::mkldnnMapToType(input.type());
  auto mkldnnEngine = detail::MkldnnEngine::getInstance().getEngine();
  auto formatX = mkldnn::memory::format::x;
  auto format2d = mkldnn::memory::format::nc;
  // MKL-DNN requires NCHW order, and it thinks data are in ROW-MAJOR layout.
  // The input tesor is in WHCN order and layout in COLUMN-MAJOR (arrayfire).
  // Thus, MKL-DNN can access the required element correctly.
  auto formatNCHW = mkldnn::memory::format::nchw;

  /****************************************************************************/
  // Prepare combined weights

  // If empty, user specifies affine to false. Both not trainable.
  auto weightNonempty = weight.isempty()
      ? Variable(af::constant(1.0, nfeatures, input.type()), false)
      : weight;
  auto biasNonempty = bias.isempty()
      ? Variable(af::constant(0.0, nfeatures, input.type()), false)
      : bias;

  // MKLDNN only accept weight and bias as a combined input.
  // https://fburl.com/l0bctocp
  auto weightsMkldnn =
      af::join(0, weightNonempty.array(), biasNonempty.array());

  /****************************************************************************/
  // Prepare the fwd operator descriptor

  auto rawDims = std::vector<int>{(int)input.dims(kWIdx),
                                  (int)input.dims(kHIdx),
                                  (int)input.dims(kChannelSizeIdx),
                                  (int)input.dims(kBatchSizeIdx)};
  if (axes.size() > 1) {
    // if norm on multiple axes, we view all axes as on channel axis
    for (auto ax : axes) {
      rawDims[ax] = (ax != kChannelSizeIdx) ? 1 : nfeatures;
    }
  }
  auto inputOutputDims = detail::convertAfToMklDnnDims({
      rawDims[kBatchSizeIdx],
      rawDims[kChannelSizeIdx],
      rawDims[kHIdx],
      rawDims[kWIdx],
  });
  auto inputOutputMemDesc =
      mkldnn::memory::desc({inputOutputDims}, dType, formatNCHW);

  // https://fburl.com/6latj733
  unsigned flag = train ? !mkldnn::use_global_stats : mkldnn::use_global_stats;
  flag = flag | mkldnn::use_scale_shift;

  // FWD primitive descriptor construction
  auto kind = train ? mkldnn::prop_kind::forward_training
                    : mkldnn::prop_kind::forward_inference;
  auto fwdDesc = mkldnn::batch_normalization_forward::desc(
      kind, inputOutputMemDesc, epsilon, flag);
  auto fwdPrimDesc =
      std::make_shared<mkldnn::batch_normalization_forward::primitive_desc>(
          fwdDesc, mkldnnEngine);

  /****************************************************************************/
  // Prepare memories

  // input
  DevicePtr inputRaw(input.array());
  auto inputMemDesc =
      mkldnn::memory::desc({inputOutputDims}, dType, formatNCHW);
  auto inputMemPrimDesc =
      mkldnn::memory::primitive_desc(inputMemDesc, mkldnnEngine);
  auto inputMemInit = mkldnn::memory(inputMemPrimDesc, inputRaw.get());

  // out
  DevicePtr outputRaw(output);
  auto outputMemDesc =
      mkldnn::memory::desc({inputOutputDims}, dType, formatNCHW);
  auto outputMemPrimDesc =
      mkldnn::memory::primitive_desc(outputMemDesc, mkldnnEngine);
  auto outputMem = mkldnn::memory(outputMemPrimDesc, outputRaw.get());

  // mean
  DevicePtr meanRaw(runningMean.array());
  auto meanDims = detail::convertAfToMklDnnDims({runningMean.dims(0)});
  auto meanMemDesc = mkldnn::memory::desc({meanDims}, dType, formatX);
  auto meanMemPrimDesc =
      mkldnn::memory::primitive_desc(meanMemDesc, mkldnnEngine);
  auto meanMemInit = mkldnn::memory(meanMemPrimDesc, meanRaw.get());

  // var
  DevicePtr varRaw(runningVar.array());
  auto varDims = detail::convertAfToMklDnnDims({runningVar.dims(0)});
  auto varMemDesc = mkldnn::memory::desc({varDims}, dType, formatX);
  auto varMemPrimDesc =
      mkldnn::memory::primitive_desc(varMemDesc, mkldnnEngine);
  auto varMemInit = mkldnn::memory(varMemPrimDesc, varRaw.get());

  // weightMKLDNN
  DevicePtr weightsMkldnnRaw(weightsMkldnn);
  auto weightsMkldnnDims = detail::convertAfToMklDnnDims({2, nfeatures});
  auto weightsMkldnnMemDesc =
      mkldnn::memory::desc({weightsMkldnnDims}, dType, format2d);
  auto weightsMkldnnMemPrimDesc =
      mkldnn::memory::primitive_desc(weightsMkldnnMemDesc, mkldnnEngine);
  auto weightsMkldnnMemInit =
      mkldnn::memory(weightsMkldnnMemPrimDesc, weightsMkldnnRaw.get());

  /****************************************************************************/
  // Setup primitive operator

  std::shared_ptr<mkldnn::batch_normalization_forward> bn;
  if (train) {
    // train: 2 in, 3 out
    bn = std::make_shared<mkldnn::batch_normalization_forward>(
        *fwdPrimDesc,
        (mkldnn::primitive::at)inputMemInit,
        (mkldnn::primitive::at)weightsMkldnnMemInit,
        outputMem,
        meanMemInit,
        varMemInit);
  } else {
    // inference: 4 in, 1 out
    bn = std::make_shared<mkldnn::batch_normalization_forward>(
        *fwdPrimDesc,
        (mkldnn::primitive::at)inputMemInit,
        (mkldnn::primitive::at)meanMemInit,
        (mkldnn::primitive::at)varMemInit,
        (mkldnn::primitive::at)weightsMkldnnMemInit,
        outputMem);
  }

  /****************************************************************************/
  // Setup execution network

  std::vector<mkldnn::primitive> network;
  network.push_back(*bn);
  detail::MkldnnStream::getInstance().getStream().submit(network);

  /****************************************************************************/
  // Setup backward func

  auto gradFunc = [train,
                   epsilon,
                   nfeatures,
                   fwdPrimDesc,
                   outputMemDesc,
                   inputOutputDims,
                   formatNCHW,
                   format2d,
                   dType,
                   weightsMkldnn,
                   weightsMkldnnDims,
                   inputMemInit,
                   meanMemInit,
                   varMemInit,
                   weightsMkldnnMemInit](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    if (!train) {
      throw std::logic_error(
          "can't compute batchnorm grad when train was not specified");
    }

    auto mkldnnEngineBwd = detail::MkldnnEngine::getInstance().getEngine();

    auto& inputRef = inputs[0];
    auto weightRef = inputs[1].isempty()
        ? Variable(af::constant(1.0, nfeatures, inputRef.type()), false)
        : inputs[1];
    auto biasRef = inputs[2].isempty()
        ? Variable(af::constant(0.0, nfeatures, inputRef.type()), false)
        : inputs[2];
    ;

    auto grad_input =
        Variable(af::array(inputRef.dims(), inputRef.type()), false);

    auto grad_weightsMKLDNN =
        Variable(af::array(weightsMkldnn.dims(), weightsMkldnn.type()), false);

    /********************************************************************/
    // Prepare memories for grad_output
    DevicePtr gradOutputRaw(grad_output.array());
    auto gradOutputMemDesc =
        mkldnn::memory::desc({inputOutputDims}, dType, formatNCHW);
    auto gradOutputMemPrimDesc =
        mkldnn::memory::primitive_desc(gradOutputMemDesc, mkldnnEngineBwd);
    auto gradOutputMemInit =
        mkldnn::memory(gradOutputMemPrimDesc, gradOutputRaw.get());

    DevicePtr gradInputRaw(grad_input.array());
    auto gradInputMemDesc =
        mkldnn::memory::desc({inputOutputDims}, dType, formatNCHW);
    auto gradInputMemPrimDesc =
        mkldnn::memory::primitive_desc(gradInputMemDesc, mkldnnEngineBwd);
    auto gradInputMemInit =
        mkldnn::memory(gradInputMemPrimDesc, gradInputRaw.get());

    DevicePtr gradWeightsMkldnnRaw(grad_weightsMKLDNN.array());
    auto gradWeightsMkldnnMemDesc =
        mkldnn::memory::desc({weightsMkldnnDims}, dType, format2d);
    auto gradWeightsMkldnnMemPrimDesc = mkldnn::memory::primitive_desc(
        gradWeightsMkldnnMemDesc, mkldnnEngineBwd);
    auto gradWeightsMkldnnMemInit = mkldnn::memory(
        gradWeightsMkldnnMemPrimDesc, gradWeightsMkldnnRaw.get());

    /********************************************************************/
    // Setup backward descriptor:

    auto bwdDesc = mkldnn::batch_normalization_backward::desc(
        mkldnn::prop_kind::backward,
        gradOutputMemDesc,
        outputMemDesc,
        epsilon,
        mkldnn::use_scale_shift);

    /********************************************************************/
    // Setup backward prim descriptor:
    auto bwdPrimDesc = mkldnn::batch_normalization_backward::primitive_desc(
        bwdDesc, mkldnnEngineBwd, *fwdPrimDesc);

    /********************************************************************/
    // Construct bwd op

    auto bwdPrim = std::make_shared<mkldnn::batch_normalization_backward>(
        bwdPrimDesc,
        inputMemInit,
        meanMemInit,
        varMemInit,
        gradOutputMemInit,
        weightsMkldnnMemInit,
        gradInputMemInit,
        gradWeightsMkldnnMemInit);

    /********************************************************************/
    // Setup execution network

    std::vector<mkldnn::primitive> networkBackwards;
    networkBackwards.push_back(*bwdPrim);
    detail::MkldnnStream::getInstance().getStream().submit(networkBackwards);

    /********************************************************************/
    // Update grad

    inputRef.addGrad(grad_input);

    // extracting grads from grad_weightsMKLDNN for weight and bias
    if (weightRef.isCalcGrad()) {
      auto gradWeight = Variable(
          grad_weightsMKLDNN.array()(
              af::seq(0, nfeatures - 1), af::span, af::span, af::span),
          false);
      weightRef.addGrad(gradWeight);

      auto gradBias = Variable(
          grad_weightsMKLDNN.array()(
              af::seq(nfeatures, 2 * nfeatures - 1),
              af::span,
              af::span,
              af::span),
          false);
      if (!biasRef.isempty()) {
        biasRef.addGrad(gradBias);
      }
    }
  };

  /****************************************************************************/
  // return

  return Variable(output, {input, weight, bias}, gradFunc);
}

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
  // CPU backend MKL-DNN does not support momentum factor.
  // If momentum enabled, throw error.
  if (momentum == 0.0) {
    return batchnorm(
        input, weight, bias, runningMean, runningVar, axes, train, epsilon);
  } else {
    throw std::runtime_error("BatchNorm CPU backend doesn't support momentum.");
  }
}

} // namespace fl
