/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include <arrayfire.h>
#include <dnnl.hpp>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cpu/DnnlUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

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

  // Check if axes are valid
  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (max_axis - min_axis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  auto dType = detail::dnnlMapToType(input.type());
  auto dnnlEngine = detail::DnnlEngine::getInstance().getEngine();
  auto formatX = dnnl::memory::format_tag::x;
  auto format2d = dnnl::memory::format_tag::nc;
  // MKL-DNN requires NCHW order and expects row-major tensors.
  // The input tensor is in WHCN and column-major as per AF, which is
  // equivalent; no reorder needed
  auto formatNCHW = dnnl::memory::format_tag::nchw;

  // Prepare combined weights
  // If empty, user specifies affine to false. Both not trainable.
  auto weightNonempty = weight.isempty()
      ? Variable(af::constant(1.0, nfeatures, input.type()), false)
      : weight;
  auto biasNonempty = bias.isempty()
      ? Variable(af::constant(0.0, nfeatures, input.type()), false)
      : bias;

  // DNNL only accept weight and bias as a combined input.
  // https://fburl.com/l0bctocp
  auto weightsDnnl = af::join(0, weightNonempty.array(), biasNonempty.array());
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
  auto inputOutputDims = detail::convertAfToDnnlDims({
      rawDims[kBatchSizeIdx],
      rawDims[kChannelSizeIdx],
      rawDims[kHIdx],
      rawDims[kWIdx],
  });
  auto inputOutputMemDesc =
      dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);

  // Memory for forward
  // input
  DevicePtr inputRaw(input.array());
  auto inputMemDesc = dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
  auto inputMemInit = dnnl::memory(inputMemDesc, dnnlEngine);
  inputMemInit.set_data_handle(inputRaw.get());
  // out
  DevicePtr outputRaw(output);
  auto outputMemDesc = dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
  auto outputMem = dnnl::memory(outputMemDesc, dnnlEngine);
  outputMem.set_data_handle(outputRaw.get());
  // mean
  DevicePtr meanRaw(runningMean.array());
  auto meanDims = detail::convertAfToDnnlDims({runningMean.dims(0)});
  auto meanMemDesc = dnnl::memory::desc({meanDims}, dType, formatX);
  auto meanMemInit = dnnl::memory(meanMemDesc, dnnlEngine);
  meanMemInit.set_data_handle(meanRaw.get());
  // var
  DevicePtr varRaw(runningVar.array());
  auto varDims = detail::convertAfToDnnlDims({runningVar.dims(0)});
  auto varMemDesc = dnnl::memory::desc({varDims}, dType, formatX);
  auto varMemInit = dnnl::memory(varMemDesc, dnnlEngine);
  varMemInit.set_data_handle(varRaw.get());
  // weightDNNL - combined scale and shift (weight and bias)
  DevicePtr weightsDnnlRaw(weightsDnnl);
  auto weightsDnnlDims = detail::convertAfToDnnlDims({2, nfeatures});
  auto weightsDnnlMemDesc =
      dnnl::memory::desc({weightsDnnlDims}, dType, format2d);
  auto weightsDnnlMemInit = dnnl::memory(weightsDnnlMemDesc, dnnlEngine);
  weightsDnnlMemInit.set_data_handle(weightsDnnlRaw.get());

  // Primitives and descriptors
  auto kind = train ? dnnl::prop_kind::forward_training
                    : dnnl::prop_kind::forward_inference;
  // https://fburl.com/6latj733
  dnnl::normalization_flags flag = train
      ? dnnl::normalization_flags::none
      : dnnl::normalization_flags::use_global_stats;
  flag = flag | dnnl::normalization_flags::use_scale_shift;
  auto fwdDesc = dnnl::batch_normalization_forward::desc(
      kind, inputOutputMemDesc, epsilon, flag);
  auto fwdPrimDesc =
      std::make_shared<dnnl::batch_normalization_forward::primitive_desc>(
          fwdDesc, dnnlEngine);
  auto bn = dnnl::batch_normalization_forward(*fwdPrimDesc);
  std::unordered_map<int, dnnl::memory> bnFwdArgs = {
      {DNNL_ARG_SRC, inputMemInit},
      {DNNL_ARG_MEAN, meanMemInit},
      {DNNL_ARG_VARIANCE, varMemInit},
      {DNNL_ARG_DST, outputMem},
      {DNNL_ARG_SCALE_SHIFT, weightsDnnlMemInit}};

  // Execute
  std::vector<dnnl::primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs = {bnFwdArgs};
  network.push_back(bn);
  detail::executeNetwork(network, fwdArgs);

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
                   weightsDnnl,
                   weightsDnnlDims,
                   inputMemInit,
                   meanMemInit,
                   varMemInit,
                   weightsDnnlMemInit](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    if (!train) {
      throw std::logic_error(
          "can't compute batchnorm grad when train was not specified");
    }

    auto dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();
    auto& inputRef = inputs[0];
    auto weightRef = inputs[1].isempty()
        ? Variable(af::constant(1.0, nfeatures, inputRef.type()), false)
        : inputs[1];
    auto biasRef = inputs[2].isempty()
        ? Variable(af::constant(0.0, nfeatures, inputRef.type()), false)
        : inputs[2];
    auto grad_input =
        Variable(af::array(inputRef.dims(), inputRef.type()), false);
    auto grad_weightsDNNL =
        Variable(af::array(weightsDnnl.dims(), weightsDnnl.type()), false);

    // Memory for gradient computation
    DevicePtr gradOutputRaw(grad_output.array());
    auto gradOutputMemDesc =
        dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
    auto gradOutputMemInit = dnnl::memory(gradOutputMemDesc, dnnlEngineBwd);
    gradOutputMemInit.set_data_handle(gradOutputRaw.get());

    DevicePtr gradInputRaw(grad_input.array());
    auto gradInputMemDesc =
        dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
    auto gradInputMemInit = dnnl::memory(gradInputMemDesc, dnnlEngineBwd);
    gradInputMemInit.set_data_handle(gradInputRaw.get());

    DevicePtr gradWeightsDnnlRaw(grad_weightsDNNL.array());
    auto gradWeightsDnnlMemDesc =
        dnnl::memory::desc({weightsDnnlDims}, dType, format2d);
    auto gradWeightsDnnlMemInit =
        dnnl::memory(gradWeightsDnnlMemDesc, dnnlEngineBwd);
    gradWeightsDnnlMemInit.set_data_handle(gradWeightsDnnlRaw.get());

    // Primitives and descriptors
    auto bwdDesc = dnnl::batch_normalization_backward::desc(
        dnnl::prop_kind::backward,
        gradOutputMemDesc,
        outputMemDesc,
        epsilon,
        dnnl::normalization_flags::use_scale_shift);
    auto bwdPrimDesc = dnnl::batch_normalization_backward::primitive_desc(
        bwdDesc, dnnlEngineBwd, *fwdPrimDesc);
    auto bwdPrim =
        std::make_shared<dnnl::batch_normalization_backward>(bwdPrimDesc);

    // Execute
    std::vector<dnnl::primitive> networkBackwards;
    std::vector<std::unordered_map<int, dnnl::memory>> bwdArgs = {
        {{DNNL_ARG_SRC, inputMemInit},
         {DNNL_ARG_MEAN, meanMemInit},
         {DNNL_ARG_VARIANCE, varMemInit},
         {DNNL_ARG_SCALE_SHIFT, weightsDnnlMemInit},
         {DNNL_ARG_DIFF_SRC, gradInputMemInit},
         {DNNL_ARG_DIFF_DST, gradOutputMemInit},
         {DNNL_ARG_DIFF_SCALE_SHIFT, gradWeightsDnnlMemInit}}};
    networkBackwards.push_back(*bwdPrim);
    detail::executeNetwork(networkBackwards, bwdArgs);

    // Update grad
    inputRef.addGrad(grad_input);
    // extracting grads from grad_weightsDNNL for weight and bias
    if (weightRef.isCalcGrad()) {
      auto gradWeight = Variable(
          grad_weightsDNNL.array()(
              af::seq(0, nfeatures - 1), af::span, af::span, af::span),
          false);
      weightRef.addGrad(gradWeight);

      auto gradBias = Variable(
          grad_weightsDNNL.array()(
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
  if (input.type() == f16) {
    throw std::runtime_error("Half precision is not supported in CPU.");
  }
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
