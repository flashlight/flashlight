/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/onednn/OneDnnAutogradExtension.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <dnnl.hpp>

#include "flashlight/fl/autograd/tensor/backend/onednn/DnnlUtils.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

namespace {

// Flashlight accept HWCN order according to docs
constexpr size_t kHIdx = 0;
constexpr size_t kWIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

// Use memory::format_tag::any for memory formatting even if pool
// inputs are shaped in a particular way.
constexpr auto formatAny = dnnl::memory::format_tag::any;
constexpr auto formatNCHW = dnnl::memory::format_tag::nchw;
constexpr auto formatX = dnnl::memory::format_tag::x;
constexpr auto format2d = dnnl::memory::format_tag::nc;

int getNfeatures(const Shape& inputShape, const std::vector<int>& axes) {
  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= inputShape.dim(ax);
  }
  return nfeatures;
}

dnnl::memory::dims getInputOutputDims(
    const int minAxis,
    const int maxAxis,
    const Tensor& input,
    const int nfeatures) {
  Shape inDescDims;
  if (minAxis == 0) {
    inDescDims = Shape(
        {1,
         1,
         nfeatures,
         static_cast<long long>(input.elements() / nfeatures)});
  } else {
    int batchsz = 1;
    for (int i = maxAxis + 1; i < input.ndim(); ++i) {
      batchsz *= input.dim(i);
    }
    inDescDims = Shape(
        {1,
         static_cast<long long>(input.elements() / (nfeatures * batchsz)),
         nfeatures,
         batchsz});
  }

  dnnl::memory::dims inputOutputDims = {
      inDescDims[kBatchSizeIdx],
      inDescDims[kChannelSizeIdx],
      inDescDims[kHIdx],
      inDescDims[kWIdx]};

  return inputOutputDims;
}

struct OneDnnBatchNormPayload : detail::AutogradPayloadData {
  dnnl::batch_normalization_forward::primitive_desc fwdPrimDesc;
  Tensor weightsDnnl; // combined weight and bias
  dnnl::memory::dims weightsDnnlDims;
  dnnl::memory::desc outputMemoryDescriptor;
  dnnl::memory meanMemory;
  dnnl::memory varMemory;
  dnnl::memory weightsMemory;
};

} // namespace

Tensor OneDnnAutogradExtension::batchnorm(
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
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (momentum != 0.) {
    throw std::runtime_error("OneDNN batchnorm op doesn't support momentum.");
  }
  if (input.type() == fl::dtype::f16) {
    throw std::runtime_error("OneDNN batchnorm op - f16 inputs not supported.");
  }

  auto payload = std::make_shared<OneDnnBatchNormPayload>();
  if (train && autogradPayload) {
    autogradPayload->data = payload;
  }

  auto output = Tensor(input.shape(), input.type());
  int nfeatures = getNfeatures(input.shape(), axes);

  if (runningVar.isEmpty()) {
    runningVar = fl::full({nfeatures}, 1., input.type());
  }

  if (runningMean.isEmpty()) {
    runningMean = fl::full({nfeatures}, 0., input.type());
  }

  // Check if axes are valid
  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (maxAxis - minAxis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  auto dType = detail::dnnlMapToType(input.type());
  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();

  // Prepare combined weights
  // If empty, user specifies affine to false. Both not trainable.
  auto weightNonempty =
      weight.isEmpty() ? fl::full({nfeatures}, 1., fl::dtype::f32) : weight;
  auto biasNonempty =
      bias.isEmpty() ? fl::full({nfeatures}, 0., fl::dtype::f32) : bias;

  // DNNL only accepts weight and bias as a combined input.
  // https://git.io/JLn9X
  payload->weightsDnnl = fl::concatenate(0, weightNonempty, biasNonempty);

  auto inputOutputDims = getInputOutputDims(minAxis, maxAxis, input, nfeatures);

  auto inputOutputMemDesc =
      dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
  payload->weightsDnnlDims = detail::convertToDnnlDims({2, nfeatures});

  // Memory for forward
  const detail::DnnlMemoryWrapper inputMemory(
      input, inputOutputDims, formatNCHW);
  const detail::DnnlMemoryWrapper outputMemory(
      output, inputOutputDims, formatNCHW);
  const detail::DnnlMemoryWrapper meanMemory(
      runningMean, {runningMean.dim(0)}, formatX);
  const detail::DnnlMemoryWrapper varMemory(
      runningVar, {runningVar.dim(0)}, formatX);
  // combined scale and shift (weight and bias)
  const detail::DnnlMemoryWrapper weightsMemory(
      payload->weightsDnnl, payload->weightsDnnlDims, format2d);
  payload->meanMemory = meanMemory.getMemory();
  payload->varMemory = varMemory.getMemory();
  payload->weightsMemory = weightsMemory.getMemory();
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
  payload->fwdPrimDesc =
      dnnl::batch_normalization_forward::primitive_desc(fwdDesc, dnnlEngine);
  payload->outputMemoryDescriptor = outputMemory.getDescriptor();
  auto bn = dnnl::batch_normalization_forward(payload->fwdPrimDesc);
  std::unordered_map<int, dnnl::memory> bnFwdArgs = {
      {DNNL_ARG_SRC, inputMemory.getMemory()},
      {DNNL_ARG_MEAN, meanMemory.getMemory()},
      {DNNL_ARG_VARIANCE, varMemory.getMemory()},
      {DNNL_ARG_DST, outputMemory.getMemory()},
      {DNNL_ARG_SCALE_SHIFT, weightsMemory.getMemory()}};

  // Execute
  std::vector<dnnl::primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs = {bnFwdArgs};
  network.push_back(bn);
  detail::executeNetwork(network, fwdArgs);

  return output;
}

std::tuple<Tensor, Tensor, Tensor> OneDnnAutogradExtension::batchnormBackward(
    const Tensor& gradOutput,
    const Tensor& saveMean,
    const Tensor& saveVar,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<int>& axes,
    const bool train,
    const float epsilon,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (!autogradPayload) {
    throw std::invalid_argument(
        "OneDnnAutogradExtension::pool2dBackward given null detail::AutogradPayload");
  }
  auto payload =
      std::static_pointer_cast<OneDnnBatchNormPayload>(autogradPayload->data);

  auto dType = detail::dnnlMapToType(input.type());
  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();

  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (maxAxis - minAxis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  int nfeatures = getNfeatures(input.shape(), axes);
  auto inputOutputDims = getInputOutputDims(minAxis, maxAxis, input, nfeatures);

  auto gradInput = Tensor(input.shape(), input.type());
  auto gradWeightsDNNL =
      Tensor(payload->weightsDnnl.shape(), payload->weightsDnnl.type());

  const detail::DnnlMemoryWrapper inputMemory(
      input, inputOutputDims, formatNCHW);

  // Memory for gradient computation
  const detail::DnnlMemoryWrapper gradOutputMem(
      gradOutput, inputOutputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradInputMem(
      gradInput, inputOutputDims, formatNCHW);
  const detail::DnnlMemoryWrapper gradWeightsMem(
      gradWeightsDNNL, payload->weightsDnnlDims, format2d);

  // Primitives and descriptors
  auto bwdDesc = dnnl::batch_normalization_backward::desc(
      dnnl::prop_kind::backward,
      gradOutputMem.getDescriptor(),
      payload->outputMemoryDescriptor,
      epsilon,
      dnnl::normalization_flags::use_scale_shift);
  auto bwdPrimDesc = dnnl::batch_normalization_backward::primitive_desc(
      bwdDesc, dnnlEngine, payload->fwdPrimDesc);
  auto bwdPrim =
      std::make_shared<dnnl::batch_normalization_backward>(bwdPrimDesc);

  // Execute
  std::vector<dnnl::primitive> networkBackwards;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdArgs = {
      {{DNNL_ARG_SRC, inputMemory.getMemory()},
       {DNNL_ARG_MEAN, payload->meanMemory},
       {DNNL_ARG_VARIANCE, payload->varMemory},
       {DNNL_ARG_SCALE_SHIFT, payload->weightsMemory},
       {DNNL_ARG_DIFF_SRC, gradInputMem.getMemory()},
       {DNNL_ARG_DIFF_DST, gradOutputMem.getMemory()},
       {DNNL_ARG_DIFF_SCALE_SHIFT, gradWeightsMem.getMemory()}}};
  networkBackwards.push_back(*bwdPrim);
  detail::executeNetwork(networkBackwards, bwdArgs);

  return {
      gradInput,
      gradWeightsDNNL(fl::range(0, nfeatures)), // weights grad
      gradWeightsDNNL(fl::range(nfeatures, 2 * nfeatures)) // bias grad
  };
};

} // namespace fl
