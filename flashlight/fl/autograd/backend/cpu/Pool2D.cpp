/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unordered_map>
#include <vector>

#include <arrayfire.h>
#include <dnnl.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Utils.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cpu/DnnlUtils.h"

using namespace dnnl;

namespace {

constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

} // namespace

namespace fl {

Variable pool2d(
    const Variable& input,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode) {
  auto inputDimsRaw = input.dims();
  auto output = af::array(
      1 + (input.dims(kWIdx) + 2 * px - wx) / sx,
      1 + (input.dims(kHIdx) + 2 * py - wy) / sy,
      input.dims(kChannelSizeIdx),
      input.dims(kBatchSizeIdx));

  // Dims
  memory::dims inputDims = detail::convertAfToDnnlDims(
      {input.dims(kBatchSizeIdx),
       input.dims(kChannelSizeIdx),
       input.dims(kHIdx),
       input.dims(kWIdx)});
  memory::dims outputDims = detail::convertAfToDnnlDims(
      {input.dims(kBatchSizeIdx),
       input.dims(kChannelSizeIdx),
       output.dims(kHIdx),
       output.dims(kWIdx)});
  memory::dims windowDims = {wy, wx};
  memory::dims strideDims = {sy, sx};
  std::vector<int64_t> paddingDims = {py, px};

  auto dataType = detail::dnnlMapToType(input.type());
  auto formatNCHW = memory::format_tag::nchw;
  auto formatAny = memory::format_tag::any;

  // Memory desc
  auto inputMD = memory::desc({inputDims}, dataType, formatNCHW);
  auto outputMD = memory::desc({outputDims}, dataType, formatAny);

  // Memory
  auto& dnnlEngine = detail::DnnlEngine::getInstance().getEngine();
  const detail::DnnlMemoryWrapper inputMemInit(
      input.array(), {inputDims}, formatNCHW);
  const detail::DnnlMemoryWrapper outputMemInit(
      output, {outputDims}, formatNCHW);

  // Choose a mode based on whether gradients are needed
  auto forwardMode =
      input.isCalcGrad() ? prop_kind::forward : prop_kind::forward_inference;

  // Descriptors
  auto poolingMode = detail::dnnlMapToPoolingMode(mode);
  auto desc = pooling_forward::desc(
      forwardMode,
      poolingMode,
      inputMD,
      outputMD,
      strideDims,
      windowDims,
      paddingDims,
      paddingDims);
  auto primDesc = pooling_forward::primitive_desc(desc, dnnlEngine);

  // Network
  std::vector<primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;
  // Reorder if needed
  auto inputDesc = primDesc.src_desc();
  auto outputDesc = primDesc.dst_desc();
  auto inputMemory = detail::dnnlAlignOrdering(
      network, fwdArgs, inputMemInit.getMemory(), inputDesc);
  auto outputMemory = outputMemInit.getMemory();
  if (outputMemInit.getMemory().get_desc() != outputDesc) {
    outputMemory = memory(outputDesc, dnnlEngine);
  }
  // Workspace and layer (only training mode requires a workspace)
  std::shared_ptr<memory> workspaceMemory; // no default ctors
  std::shared_ptr<pooling_forward> pooling;
  std::unordered_map<int, dnnl::memory> fwdPoolingArgs;
  fwdPoolingArgs[DNNL_ARG_SRC] = inputMemory;
  fwdPoolingArgs[DNNL_ARG_DST] = outputMemory;
  if (input.isCalcGrad()) {
    workspaceMemory =
        std::make_shared<memory>(primDesc.workspace_desc(), dnnlEngine);
    pooling = std::make_shared<pooling_forward>(primDesc);
    fwdPoolingArgs[DNNL_ARG_WORKSPACE] = *workspaceMemory;
  } else {
    pooling = std::make_shared<pooling_forward>(primDesc);
  }
  network.push_back(*pooling);
  fwdArgs.push_back(fwdPoolingArgs);

  // Add output reordering if needed
  if (outputMemory != outputMemInit.getMemory()) {
    network.push_back(dnnl::reorder(outputMemory, outputMemInit.getMemory()));
    fwdArgs.push_back(
        {{DNNL_ARG_FROM, outputMemory},
         {DNNL_ARG_TO, outputMemInit.getMemory()}});
  }

  detail::executeNetwork(network, fwdArgs);

  auto gradFunc =
      [dataType,
       formatNCHW,
       inputDimsRaw, // need to pass if inputs are empty
       primDescFwd = std::move(primDesc), // forward desc
       poolingMode,
       // needed for backwards pass. null in inference mode
       workspaceMemory,
       // dims
       inputDims,
       outputDims,
       windowDims,
       strideDims,
       paddingDims,
       // capture the output memory primitive desc for reordering, since it
       // can't be retrieved from the pooling primitive descriptor
       outputMemory](
          std::vector<Variable>& inputs, const Variable& grad_output) {
        auto& in = inputs[0];
        if (!in.isCalcGrad()) {
          return;
        }

        auto gradInput =
            Variable(af::array(inputDimsRaw, af::dtype::f32), false);
        auto& dnnlEngineBwd = detail::DnnlEngine::getInstance().getEngine();

        // Memory
        const detail::DnnlMemoryWrapper gradInputMemInit(
            gradInput.array(), {inputDims}, formatNCHW);
        const detail::DnnlMemoryWrapper gradOutputMemInit(
            grad_output.array(), {outputDims}, formatNCHW);

        // Descriptors
        // Memory descriptors from initialized memory must be used since
        // pooling_backward descriptors require an ordering
        auto gradInputMD = gradInputMemInit.getMemory().get_desc();
        auto gradOutputMD = gradOutputMemInit.getMemory().get_desc();
        auto bwdDesc = pooling_backward::desc(
            poolingMode,
            gradInputMD,
            gradOutputMD,
            strideDims,
            windowDims,
            paddingDims,
            paddingDims);
        // Pass forward descriptor as a hint
        auto bwdPrimDesc = pooling_backward::primitive_desc(
            bwdDesc, dnnlEngineBwd, primDescFwd);

        std::vector<primitive> networkBackward;
        std::vector<std::unordered_map<int, dnnl::memory>> bwdArgs;
        // Reorder output memory if required
        auto gradOutputMemory = detail::dnnlAlignOrdering(
            networkBackward,
            bwdArgs,
            gradOutputMemInit.getMemory(),
            outputMemory.get_desc());

        auto poolBwd = pooling_backward(bwdPrimDesc);
        std::unordered_map<int, dnnl::memory> bwdPoolingArgs = {
            {DNNL_ARG_DIFF_SRC, gradInputMemInit.getMemory()},
            {DNNL_ARG_DIFF_DST, gradOutputMemory},
            {DNNL_ARG_WORKSPACE, *workspaceMemory}};
        bwdArgs.push_back(bwdPoolingArgs);
        networkBackward.push_back(poolBwd);

        detail::executeNetwork(networkBackward, bwdArgs);

        in.addGrad(gradInput);
      };

  return Variable(output, {input.withoutData()}, gradFunc);
}

} // namespace fl
