/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <arrayfire.h>
#include <dnnl.hpp>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cpu/DnnlUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

using namespace dnnl;

namespace fl {

namespace detail {
class ConvBenchmark;
} // namespace detail

namespace {

// Input, output: WHCN; weights: WHIO
constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kIOChannelSizeIdx = 2;
constexpr size_t kIOBatchSizeIdx = 3;
constexpr size_t kWeightOutputChannelSizeIdx = 3;

} // namespace

Variable conv2d(
    const Variable& input,
    const Variable& weights,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks) {
  if (input.type() == f16) {
    throw std::runtime_error("Half precision is not supported in CPU.");
  }
  auto dummy_bias = Variable(af::array(), false);
  return conv2d(input, weights, dummy_bias, sx, sy, px, py, dx, dy, groups);
}

Variable conv2d(
    const Variable& input,
    const Variable& weights,
    const Variable& bias,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks) {
  if (input.type() == f16) {
    throw std::runtime_error("Half precision is not supported in CPU.");
  }
  auto output = af::array(
      1 +
          (input.dims(kWIdx) + (2 * px) -
           (1 + (weights.dims(kWIdx) - 1) * dx)) /
              sx,
      1 +
          (input.dims(kHIdx) + (2 * py) -
           (1 + (weights.dims(kHIdx) - 1) * dy)) /
              sy,
      weights.dims(kWeightOutputChannelSizeIdx),
      input.dims(kIOBatchSizeIdx));
  auto hasBias = bias.elements() > 0;

  // flashlight input, weight, and output shapes in column-major:
  // - Input is WHCN
  // - Weights are WHIO
  // - Output is WHCN
  // Since ArrayFire is column major, getting a raw pointer (1D
  // representation) of these shapes and viewing as if the representation is
  // row major transposes along all axis into NCHW for the input and output
  // and OIHW for the weights - these are the shapes we use for the
  // convolution.
  auto dataType = detail::dnnlMapToType(input.type());
  // Use memory::format_tag::any for memory formatting even if convolution
  // inputs are shaped a particular way.
  auto formatAny = memory::format_tag::any;
  // DNNL convention is to always shape the input and output as NHWC and
  // weights as HWIO regardless of the actual data shape. One cannot create a
  // convolution descriptor with other formatting options.
  auto formatNCHW = memory::format_tag::nchw;
  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;

  /********************************* Forward *******************************/
  // Create memory dims
  memory::dims mInputDims =
      detail::convertAfToDnnlDims({input.dims(kIOBatchSizeIdx),
                                   input.dims(kIOChannelSizeIdx),
                                   input.dims(kHIdx),
                                   input.dims(kWIdx)});
  memory::dims mWeightDims;
  if (groups == 1) {
    mWeightDims =
        detail::convertAfToDnnlDims({weights.dims(kWeightOutputChannelSizeIdx),
                                     input.dims(kIOChannelSizeIdx),
                                     weights.dims(kHIdx),
                                     weights.dims(kWIdx)});
  } else {
    mWeightDims = detail::convertAfToDnnlDims(
        {groups,
         weights.dims(kWeightOutputChannelSizeIdx) / groups,
         input.dims(kIOChannelSizeIdx) / groups,
         weights.dims(kHIdx),
         weights.dims(kWIdx)});
  }
  memory::dims mOutputDims =
      detail::convertAfToDnnlDims({input.dims(kIOBatchSizeIdx),
                                   weights.dims(kWeightOutputChannelSizeIdx),
                                   output.dims(kHIdx),
                                   output.dims(kWIdx)});
  memory::dims mBiasDims =
      detail::convertAfToDnnlDims({weights.dims(kWeightOutputChannelSizeIdx)});
  memory::dims mStrideDims = {sy, sx};
  memory::dims mPaddingDims = {py, px};
  // NB: DNNL treats a dilation of 0 as a standard convolution and indexes
  // larger dilations accordingly. See https://git.io/fhAT2 for more.
  memory::dims mDilationDims = {dy - 1, dx - 1};

  // Create memory descriptors. using format::any gives the best performance
  auto inputMD = memory::desc({mInputDims}, dataType, formatAny);
  auto outputMD = memory::desc({mOutputDims}, dataType, formatAny);
  auto weightMD = memory::desc({mWeightDims}, dataType, formatWeight);
  auto biasMD = memory::desc({mBiasDims}, dataType, formatAny);

  // Choose a mode based on whether gradients are needed
  auto forwardMode =
      (input.isCalcGrad() || weights.isCalcGrad() || bias.isCalcGrad())
      ? prop_kind::forward_training
      : prop_kind::forward_inference;

  // Convolution descriptor
  std::shared_ptr<convolution_forward::desc> fwdDescriptor;
  if (hasBias) {
    fwdDescriptor = std::make_shared<convolution_forward::desc>(
        forwardMode,
        algorithm::convolution_direct,
        inputMD,
        weightMD,
        biasMD,
        outputMD,
        mStrideDims,
        mDilationDims,
        mPaddingDims,
        mPaddingDims);
  } else {
    fwdDescriptor = std::make_shared<convolution_forward::desc>(
        forwardMode,
        algorithm::convolution_direct,
        inputMD,
        weightMD,
        outputMD,
        mStrideDims,
        mDilationDims,
        mPaddingDims,
        mPaddingDims);
  }

  // Primitive descriptor
  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();
  auto fwdPrimDesc = std::make_shared<convolution_forward::primitive_desc>(
      *fwdDescriptor, dnnlEngine);

  // Create memory
  DevicePtr inputRaw(input.array());
  auto inputMemoryInit =
      memory({{{mInputDims}, dataType, formatNCHW}, dnnlEngine});
  inputMemoryInit.set_data_handle(inputRaw.get());
  DevicePtr outputRaw(output);
  auto outputMemoryInit =
      memory({{{mOutputDims}, dataType, formatNCHW}, dnnlEngine});
  outputMemoryInit.set_data_handle(outputRaw.get());
  DevicePtr weightsRaw(weights.array());
  auto weightsMemoryInit =
      memory({{{mWeightDims}, dataType, formatWeight}, dnnlEngine});
  weightsMemoryInit.set_data_handle(weightsRaw.get());

  // Network for execution
  std::vector<primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;

  // DNNL suggests checking if the layout requested for the convolution
  // is different from NCHW/OIHW (even if specified), and reordering if
  // necessary, since the convolution itself may request a different
  // ordering
  auto inputDesc = fwdPrimDesc->src_desc();
  auto weightsDesc = fwdPrimDesc->weights_desc();
  auto outputDesc = fwdPrimDesc->dst_desc();
  // Input
  auto inputMemory =
      detail::dnnlAlignOrdering(network, fwdArgs, inputMemoryInit, inputDesc);
  auto weightsMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, weightsMemoryInit, weightsDesc);
  // Output - adds a reorder after the conv if needed
  auto outputMemory = outputMemoryInit;
  if (outputMemoryInit.get_desc() != outputDesc) {
    outputMemory = memory(outputDesc, dnnlEngine);
  }

  // Create convolution
  std::shared_ptr<convolution_forward> conv;
  DevicePtr biasRaw(bias.array());
  auto formatBias = memory::format_tag::x;
  auto biasMemory = memory({{{mBiasDims}, dataType, formatBias}, dnnlEngine});
  biasMemory.set_data_handle(biasRaw.get());
  if (hasBias) {
    conv = std::make_shared<convolution_forward>(*fwdPrimDesc);
  } else {
    conv = std::make_shared<convolution_forward>(*fwdPrimDesc);
  }
  network.push_back(*conv);

  // Conv fwd args
  std::unordered_map<int, dnnl::memory> convFwdArgs = {
      {DNNL_ARG_SRC, inputMemory},
      {DNNL_ARG_WEIGHTS, weightsMemory},
      {DNNL_ARG_DST, outputMemory}};
  if (hasBias) {
    convFwdArgs[DNNL_ARG_BIAS] = biasMemory;
  }
  fwdArgs.push_back(convFwdArgs);

  // Add output reordering if needed
  if (outputMemory != outputMemoryInit) {
    network.push_back(dnnl::reorder(outputMemory, outputMemoryInit));
    fwdArgs.push_back(
        {{DNNL_ARG_FROM, outputMemory}, {DNNL_ARG_TO, outputMemoryInit}});
  }

  // Run
  detail::executeNetwork(network, fwdArgs);

  /***************************** Backward ******************************/
  auto gradFunc = [hasBias,
                   // Types
                   dataType,
                   formatNCHW,
                   formatWeight,
                   // Dims
                   mInputDims,
                   mWeightDims,
                   mOutputDims,
                   mBiasDims,
                   mStrideDims,
                   mDilationDims,
                   mPaddingDims,
                   // Memory descriptors
                   inputMD,
                   outputMD,
                   weightMD,
                   biasMD,
                   fwdPrimDesc // used for creating a bw desc
  ](std::vector<Variable>& inputs, const Variable& grad_output) {
    auto& inputRef = inputs[0];
    auto& weightRef = inputs[1];

    auto& dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();

    /*** Compute the gradient with respect to the input ***/
    if (inputRef.isCalcGrad()) {
      // Result
      auto gradInput =
          Variable(af::array(inputRef.dims(), inputRef.type()), false);

      // Backward descriptor
      auto bwdDataDesc = std::make_shared<convolution_backward_data::desc>(
          algorithm::convolution_direct,
          inputMD,
          weightMD,
          outputMD,
          mStrideDims,
          mDilationDims,
          mPaddingDims,
          mPaddingDims);
      // Primitive descriptor
      auto bwdDataPrimDesc =
          std::make_shared<convolution_backward_data::primitive_desc>(
              *bwdDataDesc, dnnlEngineBwd, *fwdPrimDesc);

      // Create memory
      DevicePtr gradOutputRaw(grad_output.array());
      auto gradOutputMemoryInit =
          memory({{{mOutputDims}, dataType, formatNCHW}, dnnlEngineBwd});
      gradOutputMemoryInit.set_data_handle(gradOutputRaw.get());
      DevicePtr gradInputRaw(gradInput.array());
      auto gradInputMemoryInit =
          memory({{{mInputDims}, dataType, formatNCHW}, dnnlEngineBwd});
      gradInputMemoryInit.set_data_handle(gradInputRaw.get());
      DevicePtr weightRaw(weightRef.array());
      auto weightsMemoryInitBackwards =
          memory({{{mWeightDims}, dataType, formatWeight}, dnnlEngineBwd});
      weightsMemoryInitBackwards.set_data_handle(weightRaw.get());

      std::vector<primitive> networkBackwards;
      std::vector<std::unordered_map<int, dnnl::memory>> bwdDataArgs;

      // Check for reorderings
      auto gradOutputDesc = bwdDataPrimDesc->diff_dst_desc();
      auto weightsDesc = bwdDataPrimDesc->weights_desc();
      auto gradInputDesc = bwdDataPrimDesc->diff_src_desc();
      auto gradOutputMemory = detail::dnnlAlignOrdering(
          networkBackwards, bwdDataArgs, gradOutputMemoryInit, gradOutputDesc);
      auto weightsMemoryBackwards = detail::dnnlAlignOrdering(
          networkBackwards,
          bwdDataArgs,
          weightsMemoryInitBackwards,
          weightsDesc);
      auto gradInputMemory = gradInputMemoryInit;
      // Don't reorder the gradient until after the conv
      if (gradInputMemoryInit.get_desc() != gradInputDesc) {
        gradInputMemory = memory(gradInputDesc, dnnlEngineBwd);
      }

      // Convolution backwards
      auto convBwdData =
          std::make_shared<convolution_backward_data>(*bwdDataPrimDesc);

      bwdDataArgs.push_back({{DNNL_ARG_DIFF_SRC, gradInputMemory},
                             {DNNL_ARG_WEIGHTS, weightsMemoryBackwards},
                             {DNNL_ARG_DIFF_DST, gradOutputMemory}});
      networkBackwards.push_back(*convBwdData);

      // Reorder the output (which is gradInput here) if necessary
      if (gradInputMemory != gradInputMemoryInit) {
        networkBackwards.push_back(
            dnnl::reorder(gradInputMemory, gradInputMemoryInit));
        bwdDataArgs.push_back({{DNNL_ARG_FROM, gradInputMemory},
                               {DNNL_ARG_TO, gradInputMemoryInit}});
      }

      detail::executeNetwork(networkBackwards, bwdDataArgs);

      inputRef.addGrad(gradInput);
    }

    /*** Compute the gradient with respect to weight and bias ***/
    if (weightRef.isCalcGrad()) {
      // Result
      auto gradWeights =
          Variable(af::array(weightRef.dims(), weightRef.type()), false);
      Variable gradBias;
      bool computeBiasGrad = hasBias && inputs[2].isCalcGrad();
      auto& biasRef = inputs[2];
      if (computeBiasGrad) {
        gradBias = Variable(af::array(biasRef.dims(), biasRef.type()), false);
      }

      // Weight backward descriptor
      std::shared_ptr<convolution_backward_weights::desc> bwdWeightDesc;
      if (hasBias) {
        bwdWeightDesc = std::make_shared<convolution_backward_weights::desc>(
            algorithm::convolution_direct,
            inputMD,
            weightMD,
            biasMD,
            outputMD,
            mStrideDims,
            mDilationDims,
            mPaddingDims,
            mPaddingDims);
      } else {
        bwdWeightDesc = std::make_shared<convolution_backward_weights::desc>(
            algorithm::convolution_direct,
            inputMD,
            weightMD,
            outputMD,
            mStrideDims,
            mDilationDims,
            mPaddingDims,
            mPaddingDims);
      }
      // Weight backward primitive descriptor
      auto bwdWeightPrimDesc =
          std::make_shared<convolution_backward_weights::primitive_desc>(
              *bwdWeightDesc, dnnlEngineBwd, *fwdPrimDesc);

      // Create memory
      DevicePtr inputRawBackwards(inputRef.array());
      auto inputMemoryInitBackwards =
          memory({{{mInputDims}, dataType, formatNCHW}, dnnlEngineBwd});
      inputMemoryInitBackwards.set_data_handle(inputRawBackwards.get());
      DevicePtr gradOutputRaw(grad_output.array());
      auto gradOutputMemoryInit =
          memory({{{mOutputDims}, dataType, formatNCHW}, dnnlEngineBwd});
      gradOutputMemoryInit.set_data_handle(gradOutputRaw.get());
      DevicePtr gradWeightsRaw(gradWeights.array());
      auto gradWeightsMemoryInit =
          memory({{{mWeightDims}, dataType, formatWeight}, dnnlEngineBwd});
      gradWeightsMemoryInit.set_data_handle(gradWeightsRaw.get());

      std::vector<primitive> networkBackwards;
      std::vector<std::unordered_map<int, dnnl::memory>> bwdWeightsArgs;

      // Check for reorderings, reorder if needed
      auto inputDesc = bwdWeightPrimDesc->src_desc();
      auto gradOutputDesc = bwdWeightPrimDesc->diff_dst_desc();
      auto gradWeightsDesc = bwdWeightPrimDesc->diff_weights_desc();
      auto inputMemoryBackwards = detail::dnnlAlignOrdering(
          networkBackwards,
          bwdWeightsArgs,
          inputMemoryInitBackwards,
          inputDesc);
      auto gradOutputMemory = detail::dnnlAlignOrdering(
          networkBackwards,
          bwdWeightsArgs,
          gradOutputMemoryInit,
          gradOutputDesc);
      // Don't reorder the grads until after the conv bwd
      auto gradWeightsMemory = gradWeightsMemoryInit;
      if (gradWeightsMemoryInit.get_desc() != gradWeightsDesc) {
        gradWeightsMemory = memory(gradWeightsDesc, dnnlEngineBwd);
      }

      // Create the convolution backward weight
      std::shared_ptr<convolution_backward_weights> bwdWeights;
      std::unordered_map<int, dnnl::memory> bwdConvWeightsArgs = {
          {DNNL_ARG_SRC, inputMemoryBackwards},
          {DNNL_ARG_DIFF_WEIGHTS, gradWeightsMemory},
          {DNNL_ARG_DIFF_DST, gradOutputMemory}};
      DevicePtr biasRawBackwards(gradBias.array());
      auto formatBias = memory::format_tag::x;
      auto gradBiasMemory =
          memory({{{mBiasDims}, dataType, formatBias}, dnnlEngineBwd});
      gradBiasMemory.set_data_handle(biasRawBackwards.get());
      if (hasBias) {
        bwdWeights =
            std::make_shared<convolution_backward_weights>(*bwdWeightPrimDesc);
        bwdConvWeightsArgs[DNNL_ARG_DIFF_BIAS] = gradBiasMemory;
      } else {
        bwdWeights =
            std::make_shared<convolution_backward_weights>(*bwdWeightPrimDesc);
      }
      networkBackwards.push_back(*bwdWeights);
      bwdWeightsArgs.push_back(bwdConvWeightsArgs);

      // Reorder weight gradients if necessary
      if (gradWeightsMemory != gradWeightsMemoryInit) {
        networkBackwards.push_back(
            dnnl::reorder(gradWeightsMemory, gradWeightsMemoryInit));
        bwdWeightsArgs.push_back({{DNNL_ARG_FROM, gradWeightsMemory},
                                  {DNNL_ARG_TO, gradWeightsMemoryInit}});
      }

      detail::executeNetwork(networkBackwards, bwdWeightsArgs);

      // Add weight and bias gradients
      weightRef.addGrad(gradWeights);
      if (computeBiasGrad) {
        biasRef.addGrad(gradBias);
      }
    }
  };

  // Return for forward
  if (hasBias) {
    return Variable(output, {input, weights, bias}, gradFunc);
  }
  return Variable(output, {input, weights}, gradFunc);
}

} // namespace fl
