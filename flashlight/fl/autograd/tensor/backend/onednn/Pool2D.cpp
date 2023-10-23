/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/onednn/OneDnnAutogradExtension.h"

#include <unordered_map>
#include <vector>

#include <dnnl.h>

#include "flashlight/fl/autograd/tensor/backend/onednn/DnnlUtils.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace dnnl;

namespace fl {

namespace {

constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

// Use memory::format_tag::any for memory formatting even if pool
// inputs are shaped in a particular way.
constexpr auto formatAny = memory::format_tag::any;
constexpr auto formatNCHW = memory::format_tag::nchw;

struct DimsData {
  memory::dims inputDims;
  memory::dims outputDims;
  memory::dims windowDims;
  memory::dims strideDims;
  std::vector<int64_t> paddingDims;
};

DimsData getDimsData(
    const Shape& input,
    const Shape& output,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py) {
  DimsData d;
  d.inputDims = detail::convertToDnnlDims(
      {input.dim(kBatchSizeIdx),
       input.dim(kChannelSizeIdx),
       input.dim(kHIdx),
       input.dim(kWIdx)});
  d.outputDims = detail::convertToDnnlDims(
      {input.dim(kBatchSizeIdx),
       input.dim(kChannelSizeIdx),
       output.dim(kHIdx),
       output.dim(kWIdx)});
  d.windowDims = {wy, wx};
  d.strideDims = {sy, sx};
  d.paddingDims = {py, px};
  return d;
}

} // namespace

struct OneDnnPool2DPayload : detail::AutogradPayloadData {
  memory workspace;
  memory outputMemory;
  DimsData dimsData;
  pooling_forward::primitive_desc poolingFwdPrimDesc;
};

Tensor OneDnnAutogradExtension::pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  const bool train = (autogradPayload != nullptr);
  auto payload = std::make_shared<OneDnnPool2DPayload>();
  if (train) {
    autogradPayload->data = payload;
  }

  // inputX x inputY x channels x batch
  auto ix = input.dim(kWIdx);
  auto iy = input.ndim() > kHIdx ? input.dim(kHIdx) : 1;
  auto c = input.ndim() > kChannelSizeIdx ? input.dim(kChannelSizeIdx) : 1;
  auto b = input.ndim() > kBatchSizeIdx ? input.dim(kBatchSizeIdx) : 1;

  auto output = Tensor(
      {1 + (ix + 2 * px - wx) / sx, 1 + (iy + 2 * py - wy) / sy, c, b},
      input.type());

  payload->dimsData =
      getDimsData({ix, iy, c, b}, output.shape(), wx, wy, sx, sy, px, py);
  auto& d = payload->dimsData;
  auto dataType = detail::dnnlMapToType(input.type());

  // Memory desc
  auto inputMD = memory::desc({d.inputDims}, dataType, formatNCHW);
  auto outputMD = memory::desc({d.outputDims}, dataType, formatAny);

  // Memory
  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();
  const detail::DnnlMemoryWrapper inputMemInit(
      input, {d.inputDims}, formatNCHW);
  const detail::DnnlMemoryWrapper outputMemInit(
      output, {d.outputDims}, formatNCHW);

  // Choose a mode based on whether gradients are needed
  auto forwardMode = train ? prop_kind::forward : prop_kind::forward_inference;

  // Descriptors
  auto poolingMode = detail::dnnlMapToPoolingMode(mode);
  payload->poolingFwdPrimDesc = pooling_forward::primitive_desc(
      dnnlEngine,
      forwardMode,
      poolingMode,
      inputMD,
      outputMD,
      d.strideDims,
      d.windowDims,
      memory::dims{1, 1}, // dilation -- TODO: add to API
      d.paddingDims,
      d.paddingDims);
  auto& primDesc = payload->poolingFwdPrimDesc;

  // Network
  std::vector<primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;
  // Reorder if needed
  auto inputDesc = primDesc.src_desc();
  auto outputDesc = primDesc.dst_desc();
  auto inputMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, inputMemInit.getMemory(), inputDesc);
  payload->outputMemory = outputMemInit.getMemory();
  if (outputMemInit.getMemory().get_desc() != outputDesc) {
    payload->outputMemory = memory(outputDesc, dnnlEngine);
  }
  // Workspace and layer (only training mode requires a workspace)
  std::shared_ptr<pooling_forward> pooling;
  std::unordered_map<int, dnnl::memory> fwdPoolingArgs;
  fwdPoolingArgs[DNNL_ARG_SRC] = inputMemory;
  fwdPoolingArgs[DNNL_ARG_DST] = payload->outputMemory;
  if (train) {
    payload->workspace = memory(primDesc.workspace_desc(), dnnlEngine);
    pooling = std::make_shared<pooling_forward>(primDesc);
    fwdPoolingArgs[DNNL_ARG_WORKSPACE] = payload->workspace;
  } else {
    pooling = std::make_shared<pooling_forward>(primDesc);
  }
  network.push_back(*pooling);
  fwdArgs.push_back(fwdPoolingArgs);

  // Add output reordering if needed
  if (payload->outputMemory != outputMemInit.getMemory()) {
    network.push_back(
        dnnl::reorder(payload->outputMemory, outputMemInit.getMemory()));
    fwdArgs.push_back(
        {{DNNL_ARG_FROM, payload->outputMemory},
         {DNNL_ARG_TO, outputMemInit.getMemory()}});
  }

  detail::executeNetwork(network, fwdArgs);
  return output;
}

Tensor OneDnnAutogradExtension::pool2dBackward(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& poolOutput,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (!autogradPayload) {
    throw std::invalid_argument(
        "OneDnnAutogradExtension::pool2dBackward given null detail::AutogradPayload");
  }
  auto payload =
      std::static_pointer_cast<OneDnnPool2DPayload>(autogradPayload->data);

  auto gradInput = Tensor(input.shape(), fl::dtype::f32);
  auto& dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();

  DimsData& d = payload->dimsData;
  auto poolingMode = detail::dnnlMapToPoolingMode(mode);
  auto dataType = detail::dnnlMapToType(input.type());

  // Memory
  const detail::DnnlMemoryWrapper gradInputMemInit(
      gradInput, {d.inputDims}, formatNCHW);
  const detail::DnnlMemoryWrapper gradOutputMemInit(
      gradOutput, {d.outputDims}, formatNCHW);

  // Descriptors
  // Memory descriptors from initialized memory must be used since
  // pooling_backward descriptors require an ordering
  auto gradInputMD = gradInputMemInit.getMemory().get_desc();
  auto gradOutputMD = gradOutputMemInit.getMemory().get_desc();
  auto bwdPrimitiveDesc = pooling_backward::primitive_desc(
      dnnlEngineBwd,
      poolingMode,
      gradInputMD,
      gradOutputMD,
      d.strideDims,
      d.windowDims,
      memory::dims{1, 1}, // dilation - TODO: add to API
      d.paddingDims,
      d.paddingDims,
      payload->poolingFwdPrimDesc // hint
  );

  std::vector<primitive> networkBackward;
  std::vector<std::unordered_map<int, dnnl::memory>> bwdArgs;
  // Reorder output memory if required
  auto gradOutputMemory = detail::dnnlAlignOrdering(
      networkBackward,
      bwdArgs,
      gradOutputMemInit.getMemory(),
      payload->outputMemory.get_desc());

  auto poolBwd = pooling_backward(bwdPrimitiveDesc);
  std::unordered_map<int, dnnl::memory> bwdPoolingArgs = {
      {DNNL_ARG_DIFF_SRC, gradInputMemInit.getMemory()},
      {DNNL_ARG_DIFF_DST, gradOutputMemory},
      {DNNL_ARG_WORKSPACE, payload->workspace}};
  bwdArgs.push_back(bwdPoolingArgs);
  networkBackward.push_back(poolBwd);

  detail::executeNetwork(networkBackward, bwdArgs);

  return gradInput;
}

} // namespace fl
