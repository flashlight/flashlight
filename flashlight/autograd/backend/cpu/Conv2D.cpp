/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <memory>
#include <vector>

#include <arrayfire.h>
#include <mkldnn.hpp>

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cpu/MkldnnUtils.h"
#include "flashlight/common/DevicePtr.h"

using namespace mkldnn;

namespace fl {

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
    int groups) {
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
    int groups) {
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
  auto dataType = detail::mkldnnMapToType(input.type());
  // Use memory::format::any for memory formatting even if convolution inputs
  // are shaped a particular way.
  auto formatAny = memory::format::any;
  // MKL-DNN convention is to always shape the input and output as NHWC and
  // weights as HWIO regardless of the actual data shape. One cannot create a
  // convolution descriptor with other formatting options.
  auto formatNCHW = memory::format::nchw;
  auto formatWeight =
      (groups == 1) ? memory::format::oihw : memory::format::goihw;

  /********************************* Forward *******************************/
  // Create memory dims
  memory::dims mInputDims =
      detail::convertAfToMklDnnDims({input.dims(kIOBatchSizeIdx),
                                     input.dims(kIOChannelSizeIdx),
                                     input.dims(kHIdx),
                                     input.dims(kWIdx)});
  memory::dims mWeightDims;
  if (groups == 1) {
    mWeightDims = detail::convertAfToMklDnnDims(
        {weights.dims(kWeightOutputChannelSizeIdx),
         input.dims(kIOChannelSizeIdx),
         weights.dims(kHIdx),
         weights.dims(kWIdx)});
  } else {
    mWeightDims = detail::convertAfToMklDnnDims(
        {groups,
         weights.dims(kWeightOutputChannelSizeIdx) / groups,
         input.dims(kIOChannelSizeIdx) / groups,
         weights.dims(kHIdx),
         weights.dims(kWIdx)});
  }
  memory::dims mOutputDims =
      detail::convertAfToMklDnnDims({input.dims(kIOBatchSizeIdx),
                                     weights.dims(kWeightOutputChannelSizeIdx),
                                     output.dims(kHIdx),
                                     output.dims(kWIdx)});
  memory::dims mBiasDims = detail::convertAfToMklDnnDims(
      {weights.dims(kWeightOutputChannelSizeIdx)});
  memory::dims mStrideDims = {sy, sx};
  memory::dims mPaddingDims = {py, px};
  // NB: MKL-DNN treats a dilation of 0 as a standard convolution and indexes
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
        convolution_direct,
        inputMD,
        weightMD,
        biasMD,
        outputMD,
        mStrideDims,
        mDilationDims,
        mPaddingDims,
        mPaddingDims,
        padding_kind::zero);
  } else {
    fwdDescriptor = std::make_shared<convolution_forward::desc>(
        forwardMode,
        convolution_direct,
        inputMD,
        weightMD,
        outputMD,
        mStrideDims,
        mDilationDims,
        mPaddingDims,
        mPaddingDims,
        padding_kind::zero);
  }

  // Primitive descriptor
  auto mkldnnEngine = detail::MkldnnEngine::getInstance().getEngine();
  auto fwdPrimDesc = std::make_shared<convolution_forward::primitive_desc>(
      *fwdDescriptor, mkldnnEngine);

  // Create memory
  DevicePtr inputRaw(input.array());
  auto inputMemoryInit = memory(
      {{{mInputDims}, dataType, formatNCHW}, mkldnnEngine}, inputRaw.get());
  DevicePtr outputRaw(output);
  auto outputMemoryInit = memory(
      {{{mOutputDims}, dataType, formatNCHW}, mkldnnEngine}, outputRaw.get());
  DevicePtr weightsRaw(weights.array());
  auto weightsMemoryInit = memory(
      {{{mWeightDims}, dataType, formatWeight}, mkldnnEngine},
      weightsRaw.get());

  // Network for execution
  std::vector<primitive> network;
  // MKL-DNN suggests checking if the layout requested for the convolution
  // is different from NCHW/OIHW (even if specified), and reordering if
  // necessary, since the convolution itself may request a different
  // ordering
  auto inputPrimDesc = fwdPrimDesc->src_primitive_desc();
  auto weightsPrimDesc = fwdPrimDesc->weights_primitive_desc();
  auto outputPrimDesc = fwdPrimDesc->dst_primitive_desc();
  // Input
  auto inputMemory =
      detail::mkldnnAlignOrdering(network, inputMemoryInit, inputPrimDesc);
  auto weightsMemory =
      detail::mkldnnAlignOrdering(network, weightsMemoryInit, weightsPrimDesc);
  // Output - adds a reorder after the conv if needed
  auto outputMemory = outputMemoryInit;
  if (outputMemoryInit.get_primitive_desc() !=
      memory::primitive_desc(outputPrimDesc)) {
    outputMemory = memory(outputPrimDesc);
  }

  // Create convolution
  std::shared_ptr<convolution_forward> conv;
  DevicePtr biasRaw(bias.array());
  auto formatBias = memory::format::x;
  auto biasMemory = memory(
      {{{mBiasDims}, dataType, formatBias}, mkldnnEngine}, biasRaw.get());
  if (hasBias) {
    conv = std::make_shared<convolution_forward>(
        *fwdPrimDesc, inputMemory, weightsMemory, biasMemory, outputMemory);
  } else {
    conv = std::make_shared<convolution_forward>(
        *fwdPrimDesc, inputMemory, weightsMemory, outputMemory);
  }
  network.push_back(*conv);

  // Add output reordering if needed
  if (outputMemory != outputMemoryInit) {
    network.push_back(mkldnn::reorder(outputMemory, outputMemoryInit));
  }

  detail::MkldnnStream::getInstance().getStream().submit(network);

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

    auto mkldnnEngineBwd = detail::MkldnnEngine::getInstance().getEngine();

    /*** Compute the gradient with respect to the input ***/
    if (inputRef.isCalcGrad()) {
      // Result
      auto gradInput =
          Variable(af::array(inputRef.dims(), inputRef.type()), false);

      // Backward descriptor
      auto bwdDataDesc = std::make_shared<convolution_backward_data::desc>(
          convolution_direct,
          inputMD,
          weightMD,
          outputMD,
          mStrideDims,
          mDilationDims,
          mPaddingDims,
          mPaddingDims,
          padding_kind::zero);
      // Primitive descriptor
      auto bwdDataPrimDesc =
          std::make_shared<convolution_backward_data::primitive_desc>(
              *bwdDataDesc, mkldnnEngineBwd, *fwdPrimDesc);

      // Create memory
      DevicePtr gradOutputRaw(grad_output.array());
      auto gradOutputMemoryInit = memory(
          {{{mOutputDims}, dataType, formatNCHW}, mkldnnEngineBwd},
          gradOutputRaw.get());
      DevicePtr gradInputRaw(gradInput.array());
      auto gradInputMemoryInit = memory(
          {{{mInputDims}, dataType, formatNCHW}, mkldnnEngineBwd},
          gradInputRaw.get());
      DevicePtr weightRaw(weightRef.array());
      auto weightsMemoryInitBackwards = memory(
          {{{mWeightDims}, dataType, formatWeight}, mkldnnEngineBwd},
          weightRaw.get());

      std::vector<primitive> networkBackwards;
      // Check for reorderings
      auto gradOutputPrimitiveDesc = bwdDataPrimDesc->diff_dst_primitive_desc();
      auto weightsPrimitiveDesc = bwdDataPrimDesc->weights_primitive_desc();
      auto gradInputPrimitiveDesc = bwdDataPrimDesc->diff_src_primitive_desc();
      auto gradOutputMemory = detail::mkldnnAlignOrdering(
          networkBackwards, gradOutputMemoryInit, gradOutputPrimitiveDesc);
      auto weightsMemoryBackwards = detail::mkldnnAlignOrdering(
          networkBackwards, weightsMemoryInitBackwards, weightsPrimitiveDesc);
      auto gradInputMemory = gradInputMemoryInit;
      // Don't reorder the gradient until after the conv
      if (gradInputMemoryInit.get_primitive_desc() !=
          memory::primitive_desc(gradInputPrimitiveDesc)) {
        gradInputMemory = memory(gradInputPrimitiveDesc);
      }

      // Convolution backwards
      auto convBwdData = std::make_shared<convolution_backward_data>(
          *bwdDataPrimDesc,
          gradOutputMemory,
          weightsMemoryBackwards,
          gradInputMemory);
      networkBackwards.push_back(*convBwdData);

      detail::MkldnnStream::getInstance().getStream().submit(networkBackwards);

      // Reorder the output (which is gradInput here) if necessary
      if (gradInputMemory != gradInputMemoryInit) {
        networkBackwards.push_back(
            mkldnn::reorder(gradInputMemory, gradInputMemoryInit));
      }

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
            convolution_direct,
            inputMD,
            weightMD,
            biasMD,
            outputMD,
            mStrideDims,
            mDilationDims,
            mPaddingDims,
            mPaddingDims,
            padding_kind::zero);
      } else {
        bwdWeightDesc = std::make_shared<convolution_backward_weights::desc>(
            convolution_direct,
            inputMD,
            weightMD,
            outputMD,
            mStrideDims,
            mDilationDims,
            mPaddingDims,
            mPaddingDims,
            padding_kind::zero);
      }
      // Weight backward primitive descriptor
      auto bwdWeightPrimDesc =
          std::make_shared<convolution_backward_weights::primitive_desc>(
              *bwdWeightDesc, mkldnnEngineBwd, *fwdPrimDesc);

      // Create memory
      DevicePtr inputRawBackwards(inputRef.array());
      auto inputMemoryInitBackwards = memory(
          {{{mInputDims}, dataType, formatNCHW}, mkldnnEngineBwd},
          inputRawBackwards.get());
      DevicePtr gradOutputRaw(grad_output.array());
      auto gradOutputMemoryInit = memory(
          {{{mOutputDims}, dataType, formatNCHW}, mkldnnEngineBwd},
          gradOutputRaw.get());
      DevicePtr gradWeightsRaw(gradWeights.array());
      auto gradWeightsMemoryInit = memory(
          {{{mWeightDims}, dataType, formatWeight}, mkldnnEngineBwd},
          gradWeightsRaw.get());

      std::vector<primitive> networkBackwards;
      // Check for reorderings, reorder if needed
      auto inputPrimitiveDesc = bwdWeightPrimDesc->src_primitive_desc();
      auto gradOutputPrimitiveDesc =
          bwdWeightPrimDesc->diff_dst_primitive_desc();
      auto gradWeightsPrimitiveDesc =
          bwdWeightPrimDesc->diff_weights_primitive_desc();
      auto inputMemoryBackwards = detail::mkldnnAlignOrdering(
          networkBackwards, inputMemoryInitBackwards, inputPrimitiveDesc);
      auto gradOutputMemory = detail::mkldnnAlignOrdering(
          networkBackwards, gradOutputMemoryInit, gradOutputPrimitiveDesc);
      // Don't reorder the grads until after the conv bwd
      auto gradWeightsMemory = gradWeightsMemoryInit;
      if (gradWeightsMemoryInit.get_primitive_desc() !=
          memory::primitive_desc(gradWeightsPrimitiveDesc)) {
        gradWeightsMemory = memory(gradWeightsPrimitiveDesc);
      }

      // Create the convolution backward weight
      std::shared_ptr<convolution_backward_weights> bwdWeights;
      DevicePtr biasRawBackwards(gradBias.array());
      auto formatBias = memory::format::x;
      auto gradBiasMemory = memory(
          {{{mBiasDims}, dataType, formatBias}, mkldnnEngineBwd},
          biasRawBackwards.get());
      if (hasBias) {
        bwdWeights = std::make_shared<convolution_backward_weights>(
            *bwdWeightPrimDesc,
            inputMemoryBackwards,
            gradOutputMemory,
            gradWeightsMemory,
            gradBiasMemory);
      } else {
        bwdWeights = std::make_shared<convolution_backward_weights>(
            *bwdWeightPrimDesc,
            inputMemoryBackwards,
            gradOutputMemory,
            gradWeightsMemory);
      }
      networkBackwards.push_back(*bwdWeights);

      // Reorder weight gradients if necessary
      if (gradWeightsMemory != gradWeightsMemoryInit) {
        networkBackwards.push_back(
            mkldnn::reorder(gradWeightsMemory, gradWeightsMemoryInit));
      }

      detail::MkldnnStream::getInstance().getStream().submit(networkBackwards);

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
