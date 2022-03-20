/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/onednn/OneDnnAutogradExtension.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/tensor/backend/onednn/DnnlUtils.h"

using namespace dnnl;

namespace fl {

namespace {

// Input, output: WHCN; weights: WHIO
constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kIOChannelSizeIdx = 2;
constexpr size_t kIOBatchSizeIdx = 3;
constexpr size_t kWeightOutputChannelSizeIdx = 3;

// Use memory::format_tag::any for memory formatting even if convolution
// inputs are shaped in a particular way.
constexpr auto formatAny = memory::format_tag::any;
constexpr auto formatNCHW = memory::format_tag::nchw;
constexpr auto formatBias = memory::format_tag::x;

struct OneDnnConv2DPayload : detail::AutogradPayloadData {
  memory::dims inputDims;
  memory::dims weightDims;
  memory::dims outputDims;
  memory::dims biasDims;
  memory::dims strideDims;
  memory::dims dilationDims;
  memory::dims paddingDims;
  // Memory descriptors
  memory::desc inputMemDesc;
  memory::desc outputMemDesc;
  memory::desc weightMemDesc;
  memory::desc biasMemDesc;
  // used for creating a backward desc
  convolution_forward::primitive_desc fwdPrimDesc;
};

} // namespace

Tensor OneDnnAutogradExtension::conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (input.type() == fl::dtype::f16) {
    throw std::runtime_error("Half precision is not supported in CPU.");
  }

  const bool train = (autogradPayload != nullptr);
  auto payload = std::make_shared<OneDnnConv2DPayload>();
  if (train) {
    autogradPayload->data = payload;
  }

  // flashlight input, weight, and output shapes in column-major:
  // - Input is WHCN
  // - Weights are WHIO
  // - Output is WHCN
  // Since ArrayFire is column major, getting a raw pointer (1D
  // representation) of these shapes and viewing as if the representation is
  // row major transposes along all axis into NCHW for the input and output
  // and OIHW for the weights
  auto output = Tensor(
      {1 +
           (input.dim(kWIdx) + (2 * px) - (1 + (weights.dim(kWIdx) - 1) * dx)) /
               sx,
       1 +
           (input.dim(kHIdx) + (2 * py) - (1 + (weights.dim(kHIdx) - 1) * dy)) /
               sy,
       weights.dim(kWeightOutputChannelSizeIdx),
       input.dim(kIOBatchSizeIdx)},
      input.type());
  auto hasBias = bias.size() > 0;

  auto dataType = detail::dnnlMapToType(input.type());
  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;

  /********************************* Forward *******************************/
  // Create memory dims
  payload->inputDims = detail::convertToDnnlDims(
      {input.dim(kIOBatchSizeIdx),
       input.dim(kIOChannelSizeIdx),
       input.dim(kHIdx),
       input.dim(kWIdx)});
  if (groups == 1) {
    payload->weightDims = detail::convertToDnnlDims(
        {weights.dim(kWeightOutputChannelSizeIdx),
         input.dim(kIOChannelSizeIdx),
         weights.dim(kHIdx),
         weights.dim(kWIdx)});
  } else {
    payload->weightDims = detail::convertToDnnlDims(
        {groups,
         weights.dim(kWeightOutputChannelSizeIdx) / groups,
         input.dim(kIOChannelSizeIdx) / groups,
         weights.dim(kHIdx),
         weights.dim(kWIdx)});
  }
  payload->outputDims = detail::convertToDnnlDims(
      {input.dim(kIOBatchSizeIdx),
       weights.dim(kWeightOutputChannelSizeIdx),
       output.dim(kHIdx),
       output.dim(kWIdx)});
  payload->biasDims =
      detail::convertToDnnlDims({weights.dim(kWeightOutputChannelSizeIdx)});
  payload->strideDims = {sy, sx};
  payload->paddingDims = {py, px};
  // NB: DNNL treats a dilation of 0 as a standard convolution and indexes
  // larger dilations accordingly. See https://git.io/fhAT2 for more.
  payload->dilationDims = {dy - 1, dx - 1};

  // Create memory descriptors. using format::any gives the best performance
  payload->inputMemDesc =
      memory::desc({payload->inputDims}, dataType, formatAny);
  payload->outputMemDesc =
      memory::desc({payload->outputDims}, dataType, formatAny);
  payload->weightMemDesc =
      memory::desc({payload->weightDims}, dataType, formatWeight);
  payload->biasMemDesc = memory::desc({payload->biasDims}, dataType, formatAny);

  // Choose a mode based on whether gradients are needed
  auto forwardMode =
      train ? prop_kind::forward_training : prop_kind::forward_inference;

  // Convolution descriptor
  std::shared_ptr<convolution_forward::desc> fwdDescriptor;
  if (hasBias) {
    fwdDescriptor = std::make_shared<convolution_forward::desc>(
        forwardMode,
        algorithm::convolution_direct,
        payload->inputMemDesc,
        payload->weightMemDesc,
        payload->biasMemDesc,
        payload->outputMemDesc,
        payload->strideDims,
        payload->dilationDims,
        payload->paddingDims,
        payload->paddingDims);
  } else {
    fwdDescriptor = std::make_shared<convolution_forward::desc>(
        forwardMode,
        algorithm::convolution_direct,
        payload->inputMemDesc,
        payload->weightMemDesc,
        payload->outputMemDesc,
        payload->strideDims,
        payload->dilationDims,
        payload->paddingDims,
        payload->paddingDims);
  }

  // Primitive descriptor
  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();
  payload->fwdPrimDesc =
      convolution_forward::primitive_desc(*fwdDescriptor, dnnlEngine);

  // Create memory
  const detail::DnnlMemoryWrapper inputMemInit(
      input, {payload->inputDims}, formatNCHW);
  const detail::DnnlMemoryWrapper outputMemInit(
      output, {payload->outputDims}, formatNCHW);
  const detail::DnnlMemoryWrapper weightsMem(
      weights, {payload->weightDims}, formatWeight);

  // Network for execution
  std::vector<primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;

  // DNNL suggests checking if the layout requested for the convolution
  // is different from NCHW/OIHW (even if specified), and reordering if
  // necessary, since the convolution itself may request a different
  // ordering
  auto inputDesc = payload->fwdPrimDesc.src_desc();
  auto weightsDesc = payload->fwdPrimDesc.weights_desc();
  auto outputDesc = payload->fwdPrimDesc.dst_desc();
  // Input
  auto inputMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, inputMemInit.getMemory(), inputDesc);
  auto weightsMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, weightsMem.getMemory(), weightsDesc);
  // Output - adds a reorder after the conv if needed
  auto outputMemory = outputMemInit.getMemory();
  if (outputMemInit.getMemory().get_desc() != outputDesc) {
    outputMemory = memory(outputDesc, dnnlEngine);
  }

  // Create convolution
  std::shared_ptr<convolution_forward> conv;
  const detail::DnnlMemoryWrapper biasMemory(
      bias, payload->biasDims, formatBias);
  if (hasBias) {
    conv = std::make_shared<convolution_forward>(payload->fwdPrimDesc);
  } else {
    conv = std::make_shared<convolution_forward>(payload->fwdPrimDesc);
  }
  network.push_back(*conv);

  // Conv fwd args
  std::unordered_map<int, dnnl::memory> convFwdArgs = {
      {DNNL_ARG_SRC, inputMemory},
      {DNNL_ARG_WEIGHTS, weightsMemory},
      {DNNL_ARG_DST, outputMemory}};
  if (hasBias) {
    convFwdArgs[DNNL_ARG_BIAS] = biasMemory.getMemory();
  }
  fwdArgs.push_back(convFwdArgs);

  // Add output reordering if needed
  if (outputMemory != outputMemInit.getMemory()) {
    network.push_back(dnnl::reorder(outputMemory, outputMemInit.getMemory()));
    fwdArgs.push_back(
        {{DNNL_ARG_FROM, outputMemory},
         {DNNL_ARG_TO, outputMemInit.getMemory()}});
  }

  // Run
  detail::executeNetwork(network, fwdArgs);

  return output;
}

Tensor OneDnnAutogradExtension::conv2dBackwardData(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weights,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups,
    std::shared_ptr<DynamicBenchmark>,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (!autogradPayload) {
    throw std::invalid_argument(
        "OneDnnAutogradExtension::Tensor OneDnnAutogradExtension::conv2dBackwardData("
        "given null detail::AutogradPayload");
  }
  auto payload =
      std::static_pointer_cast<OneDnnConv2DPayload>(autogradPayload->data);

  auto gradInput = Tensor(input.shape(), input.type()); // Result

  auto dataType = detail::dnnlMapToType(input.type());
  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  auto& dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();

  // Backward descriptor
  auto bwdDataDesc = std::make_shared<convolution_backward_data::desc>(
      algorithm::convolution_direct,
      payload->inputMemDesc,
      payload->weightMemDesc,
      payload->outputMemDesc,
      payload->strideDims,
      payload->dilationDims,
      payload->paddingDims,
      payload->paddingDims);
  // Primitive descriptor
  auto bwdDataPrimDesc =
      std::make_shared<convolution_backward_data::primitive_desc>(
          *bwdDataDesc, dnnlEngineBwd, payload->fwdPrimDesc);

  // Create memory
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      gradOutput, payload->outputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradInputMemInit(
      gradInput, payload->inputDims, formatNCHW);
  const detail::DnnlMemoryWrapper weightsMemInitBwd(
      weights, payload->weightDims, formatWeight);

  std::vector<primitive> networkBackwards;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdDataArgs;

  // Check for reorderings
  auto gradOutputDesc = bwdDataPrimDesc->diff_dst_desc();
  auto weightsDesc = bwdDataPrimDesc->weights_desc();
  auto gradInputDesc = bwdDataPrimDesc->diff_src_desc();
  auto gradOutputMemory = detail::dnnlAlignOrdering(
      networkBackwards,
      bwdDataArgs,
      gradOutputMemInit.getMemory(),
      gradOutputDesc);
  auto weightsMemoryBackwards = detail::dnnlAlignOrdering(
      networkBackwards,
      bwdDataArgs,
      weightsMemInitBwd.getMemory(),
      weightsDesc);
  auto gradInputMemory = gradInputMemInit.getMemory();
  // Don't reorder the gradient until after the conv
  if (gradInputMemInit.getMemory().get_desc() != gradInputDesc) {
    gradInputMemory = memory(gradInputDesc, dnnlEngineBwd);
  }

  // Convolution backwards
  auto convBwdData =
      std::make_shared<convolution_backward_data>(*bwdDataPrimDesc);

  bwdDataArgs.push_back(
      {{DNNL_ARG_DIFF_SRC, gradInputMemory},
       {DNNL_ARG_WEIGHTS, weightsMemoryBackwards},
       {DNNL_ARG_DIFF_DST, gradOutputMemory}});
  networkBackwards.push_back(*convBwdData);

  // Reorder the output (which is gradInput here) if necessary
  if (gradInputMemory != gradInputMemInit.getMemory()) {
    networkBackwards.push_back(
        dnnl::reorder(gradInputMemory, gradInputMemInit.getMemory()));
    bwdDataArgs.push_back(
        {{DNNL_ARG_FROM, gradInputMemory},
         {DNNL_ARG_TO, gradInputMemInit.getMemory()}});
  }

  detail::executeNetwork(networkBackwards, bwdDataArgs);

  return gradInput;
}

std::pair<Tensor, Tensor> OneDnnAutogradExtension::conv2dBackwardFilterBias(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups,
    std::shared_ptr<DynamicBenchmark>,
    std::shared_ptr<DynamicBenchmark>,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (!autogradPayload) {
    throw std::invalid_argument(
        "OneDnnAutogradExtension::Tensor OneDnnAutogradExtension::conv2dBackwardData("
        "given null detail::AutogradPayload");
  }
  auto payload =
      std::static_pointer_cast<OneDnnConv2DPayload>(autogradPayload->data);

  auto gradWeights = Tensor(weights.shape(), weights.type()); // Result

  auto dataType = detail::dnnlMapToType(input.type());
  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  auto& dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();

  Tensor gradBias;
  bool computeBiasGrad = !bias.isEmpty() && !payload->biasMemDesc.is_zero();
  if (computeBiasGrad) {
    gradBias = Tensor(bias.shape(), bias.type());
  }

  // Weight backward descriptor
  std::shared_ptr<convolution_backward_weights::desc> bwdWeightDesc;
  if (computeBiasGrad) {
    bwdWeightDesc = std::make_shared<convolution_backward_weights::desc>(
        algorithm::convolution_direct,
        payload->inputMemDesc,
        payload->weightMemDesc,
        payload->biasMemDesc,
        payload->outputMemDesc,
        payload->strideDims,
        payload->dilationDims,
        payload->paddingDims,
        payload->paddingDims);
  } else {
    bwdWeightDesc = std::make_shared<convolution_backward_weights::desc>(
        algorithm::convolution_direct,
        payload->inputMemDesc,
        payload->weightMemDesc,
        payload->outputMemDesc,
        payload->strideDims,
        payload->dilationDims,
        payload->paddingDims,
        payload->paddingDims);
  }
  // Weight backward primitive descriptor
  auto bwdWeightPrimDesc =
      std::make_shared<convolution_backward_weights::primitive_desc>(
          *bwdWeightDesc, dnnlEngineBwd, payload->fwdPrimDesc);

  // Create memory
  const detail::DnnlMemoryWrapper inputRawMemInitBwd(
      input, payload->inputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      gradOutput, payload->outputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradWeightsMemInit(
      gradWeights, payload->weightDims, formatWeight);

  std::vector<primitive> networkBackwards;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdWeightsArgs;

  // Check for reorderings, reorder if needed
  auto inputDesc = bwdWeightPrimDesc->src_desc();
  auto gradOutputDesc = bwdWeightPrimDesc->diff_dst_desc();
  auto gradWeightsDesc = bwdWeightPrimDesc->diff_weights_desc();
  auto inputMemoryBackwards = detail::dnnlAlignOrdering(
      networkBackwards,
      bwdWeightsArgs,
      inputRawMemInitBwd.getMemory(),
      inputDesc);
  auto gradOutputMemory = detail::dnnlAlignOrdering(
      networkBackwards,
      bwdWeightsArgs,
      gradOutputMemInit.getMemory(),
      gradOutputDesc);
  // Don't reorder the grads until after the conv bwd
  auto gradWeightsMemory = gradWeightsMemInit.getMemory();
  if (gradWeightsMemInit.getMemory().get_desc() != gradWeightsDesc) {
    gradWeightsMemory = memory(gradWeightsDesc, dnnlEngineBwd);
  }

  // Create the convolution backward weight
  std::shared_ptr<convolution_backward_weights> bwdWeights;
  std::unordered_map<int, dnnl::memory> bwdConvWeightsArgs = {
      {DNNL_ARG_SRC, inputMemoryBackwards},
      {DNNL_ARG_DIFF_WEIGHTS, gradWeightsMemory},
      {DNNL_ARG_DIFF_DST, gradOutputMemory}};

  if (computeBiasGrad) {
    const detail::DnnlMemoryWrapper gradBiasMem(
        gradBias, payload->biasDims, formatBias);
    bwdWeights =
        std::make_shared<convolution_backward_weights>(*bwdWeightPrimDesc);
    bwdConvWeightsArgs[DNNL_ARG_DIFF_BIAS] = gradBiasMem.getMemory();
  } else {
    bwdWeights =
        std::make_shared<convolution_backward_weights>(*bwdWeightPrimDesc);
  }
  networkBackwards.push_back(*bwdWeights);
  bwdWeightsArgs.push_back(bwdConvWeightsArgs);

  // Reorder weight gradients if necessary
  if (gradWeightsMemory != gradWeightsMemInit.getMemory()) {
    networkBackwards.push_back(
        dnnl::reorder(gradWeightsMemory, gradWeightsMemInit.getMemory()));
    bwdWeightsArgs.push_back(
        {{DNNL_ARG_FROM, gradWeightsMemory},
         {DNNL_ARG_TO, gradWeightsMemInit.getMemory()}});
  }

  detail::executeNetwork(networkBackwards, bwdWeightsArgs);

  return {gradWeights, gradBias};
}

} // namespace fl
