/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/onednn/OneDnnAutogradExtension.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>

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

struct OneDnnConv2DData {
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

OneDnnConv2DData createOneDnnConv2DData(
    fl::dtype inputType,
    const Shape& inputShape,
    const Shape& weightsShape,
    const Shape& biasShape,
    const Shape& outputShape,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups) {
  const dnnl::memory::data_type dataType = detail::dnnlMapToType(inputType);
  const auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  const bool hasBias = biasShape.elements() > 0;

  OneDnnConv2DData out;
  // Create memory dims
  out.inputDims = detail::convertToDnnlDims(
      {inputShape.dim(kIOBatchSizeIdx),
       inputShape.dim(kIOChannelSizeIdx),
       inputShape.dim(kHIdx),
       inputShape.dim(kWIdx)});
  if (groups == 1) {
    out.weightDims = detail::convertToDnnlDims(
        {weightsShape.dim(kWeightOutputChannelSizeIdx),
         inputShape.dim(kIOChannelSizeIdx),
         weightsShape.dim(kHIdx),
         weightsShape.dim(kWIdx)});
  } else {
    out.weightDims = detail::convertToDnnlDims(
        {groups,
         weightsShape.dim(kWeightOutputChannelSizeIdx) / groups,
         inputShape.dim(kIOChannelSizeIdx) / groups,
         weightsShape.dim(kHIdx),
         weightsShape.dim(kWIdx)});
  }
  out.outputDims = detail::convertToDnnlDims(
      {inputShape.dim(kIOBatchSizeIdx),
       weightsShape.dim(kWeightOutputChannelSizeIdx),
       outputShape.dim(kHIdx),
       outputShape.dim(kWIdx)});
  out.biasDims = detail::convertToDnnlDims(
      {weightsShape.dim(kWeightOutputChannelSizeIdx)});
  out.strideDims = {sy, sx};
  out.paddingDims = {py, px};
  // NB: DNNL treats a dilation of 0 as a standard convolution and indexes
  // larger dilations accordingly. See https://git.io/fhAT2 for more.
  out.dilationDims = {dy - 1, dx - 1};

  // Create memory descriptors. using format::any gives the best performance
  out.inputMemDesc = memory::desc({out.inputDims}, dataType, formatAny);
  out.outputMemDesc = memory::desc({out.outputDims}, dataType, formatAny);
  out.weightMemDesc = memory::desc({out.weightDims}, dataType, formatWeight);
  out.biasMemDesc = memory::desc({out.biasDims}, dataType, formatAny);

  //
  const auto forwardMode = prop_kind::forward_training;
  // TODO: determine train mode/assess perf impact of always choosing training
  // (primitive cache storage overhead?)
  // const auto forwardMode =
  //     train ? prop_kind::forward_training : prop_kind::forward_inference;

  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();
  convolution_forward::primitive_desc fwdPrimitiveDescriptor;
  if (hasBias) {
    fwdPrimitiveDescriptor = convolution_forward::primitive_desc(
        dnnlEngine,
        forwardMode,
        algorithm::convolution_direct,
        out.inputMemDesc,
        out.weightMemDesc,
        out.biasMemDesc,
        out.outputMemDesc,
        out.strideDims,
        out.dilationDims,
        out.paddingDims,
        out.paddingDims);
  } else {
    fwdPrimitiveDescriptor = convolution_forward::primitive_desc(
        dnnlEngine,
        forwardMode,
        algorithm::convolution_direct,
        out.inputMemDesc,
        out.weightMemDesc,
        out.outputMemDesc,
        out.strideDims,
        out.dilationDims,
        out.paddingDims,
        out.paddingDims);
  }
  out.fwdPrimDesc = std::move(fwdPrimitiveDescriptor);

  return out;
}

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
    std::shared_ptr<detail::AutogradPayload>) {
  if (input.type() == fl::dtype::f16) {
    throw std::runtime_error("Half precision is not supported in CPU.");
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
  auto hasBias = bias.elements() > 0;

  auto dataType = detail::dnnlMapToType(input.type());
  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();

  /********************************* Forward *******************************/
  OneDnnConv2DData conv2DData = createOneDnnConv2DData(
      input.type(),
      input.shape(),
      weights.shape(),
      bias.shape(),
      output.shape(),
      sx,
      sy,
      px,
      py,
      dx,
      dy,
      groups);

  // Create memory
  const detail::DnnlMemoryWrapper inputMemInit(
      input, {conv2DData.inputDims}, formatNCHW);
  const detail::DnnlMemoryWrapper outputMemInit(
      output, {conv2DData.outputDims}, formatNCHW);
  const detail::DnnlMemoryWrapper weightsMem(
      weights, {conv2DData.weightDims}, formatWeight);

  // Network for execution
  std::vector<primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;

  // DNNL suggests checking if the layout requested for the convolution
  // is different from NCHW/OIHW (even if specified), and reordering if
  // necessary, since the convolution itself may request a different
  // ordering
  auto inputDesc = conv2DData.fwdPrimDesc.src_desc();
  auto weightsDesc = conv2DData.fwdPrimDesc.weights_desc();
  auto outputDesc = conv2DData.fwdPrimDesc.dst_desc();
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
      bias, conv2DData.biasDims, formatBias);
  conv = std::make_shared<convolution_forward>(conv2DData.fwdPrimDesc);

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
    std::shared_ptr<detail::AutogradPayload>) {
  auto gradInput = Tensor(input.shape(), input.type()); // Result

  auto dataType = detail::dnnlMapToType(input.type());
  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  auto& dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();

  Tensor bias; // dummy
  OneDnnConv2DData conv2DData = createOneDnnConv2DData(
      input.type(),
      input.shape(),
      weights.shape(),
      bias.shape(),
      gradOutput.shape(), // has the same shape as the Conv output
      sx,
      sy,
      px,
      py,
      dx,
      dy,
      groups);

  // Backward descriptor
  convolution_backward_data::primitive_desc bwdDataPrimitiveDesc(
      dnnlEngineBwd,
      algorithm::convolution_direct,
      conv2DData.inputMemDesc,
      conv2DData.weightMemDesc,
      conv2DData.outputMemDesc,
      conv2DData.strideDims,
      conv2DData.dilationDims,
      conv2DData.paddingDims,
      conv2DData.paddingDims,
      conv2DData.fwdPrimDesc);
  // Primitive descriptor
  auto bwdData =
      std::make_shared<convolution_backward_data>(bwdDataPrimitiveDesc);

  // Create memory
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      gradOutput, conv2DData.outputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradInputMemInit(
      gradInput, conv2DData.inputDims, formatNCHW);
  const detail::DnnlMemoryWrapper weightsMemInitBwd(
      weights, conv2DData.weightDims, formatWeight);

  std::vector<primitive> networkBackwards;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdDataArgs;

  // Check for reorderings
  auto gradOutputDesc = bwdDataPrimitiveDesc.diff_dst_desc();
  auto weightsDesc = bwdDataPrimitiveDesc.weights_desc();
  auto gradInputDesc = bwdDataPrimitiveDesc.diff_src_desc();
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
      std::make_shared<convolution_backward_data>(bwdDataPrimitiveDesc);

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
    std::shared_ptr<detail::AutogradPayload>) {
  auto gradWeights = Tensor(weights.shape(), weights.type()); // Result

  auto dataType = detail::dnnlMapToType(input.type());
  auto formatWeight =
      (groups == 1) ? memory::format_tag::oihw : memory::format_tag::goihw;
  auto& dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();
  OneDnnConv2DData conv2DData = createOneDnnConv2DData(
      input.type(),
      input.shape(),
      weights.shape(),
      bias.shape(),
      gradOutput.shape(), // has the same shape as the Conv output
      sx,
      sy,
      px,
      py,
      dx,
      dy,
      groups);

  Tensor gradBias;
  bool computeBiasGrad = !bias.isEmpty() && !conv2DData.biasMemDesc.is_zero();
  if (computeBiasGrad) {
    gradBias = Tensor(bias.shape(), bias.type());
  }

  // Weight backward descriptor
  convolution_backward_weights::primitive_desc bwdWeightPrimitiveDesc;
  if (computeBiasGrad) {
    bwdWeightPrimitiveDesc = convolution_backward_weights::primitive_desc(
        dnnlEngineBwd,
        algorithm::convolution_direct,
        conv2DData.inputMemDesc,
        conv2DData.weightMemDesc,
        conv2DData.biasMemDesc,
        conv2DData.outputMemDesc,
        conv2DData.strideDims,
        conv2DData.dilationDims,
        conv2DData.paddingDims,
        conv2DData.paddingDims,
        conv2DData.fwdPrimDesc);
  } else {
    bwdWeightPrimitiveDesc = convolution_backward_weights::primitive_desc(
        dnnlEngineBwd,
        algorithm::convolution_direct,
        conv2DData.inputMemDesc,
        conv2DData.weightMemDesc,
        conv2DData.outputMemDesc,
        conv2DData.strideDims,
        conv2DData.dilationDims,
        conv2DData.paddingDims,
        conv2DData.paddingDims,
        conv2DData.fwdPrimDesc);
  }
  // Weight backward primitive descriptor
  auto bwdWeights =
      std::make_shared<convolution_backward_weights>(bwdWeightPrimitiveDesc);

  // Create memory
  const detail::DnnlMemoryWrapper inputRawMemInitBwd(
      input, conv2DData.inputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      gradOutput, conv2DData.outputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradWeightsMemInit(
      gradWeights, conv2DData.weightDims, formatWeight);

  std::vector<primitive> networkBackwards;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdWeightsArgs;

  // Check for reorderings, reorder if needed
  auto inputDesc = bwdWeightPrimitiveDesc.src_desc();
  auto gradOutputDesc = bwdWeightPrimitiveDesc.diff_dst_desc();
  auto gradWeightsDesc = bwdWeightPrimitiveDesc.diff_weights_desc();
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
  std::unordered_map<int, dnnl::memory> bwdConvWeightsArgs = {
      {DNNL_ARG_SRC, inputMemoryBackwards},
      {DNNL_ARG_DIFF_WEIGHTS, gradWeightsMemory},
      {DNNL_ARG_DIFF_DST, gradOutputMemory}};

  if (computeBiasGrad) {
    const detail::DnnlMemoryWrapper gradBiasMem(
        gradBias, conv2DData.biasDims, formatBias);
    bwdConvWeightsArgs[DNNL_ARG_DIFF_BIAS] = gradBiasMem.getMemory();
  } else {
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
