/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include <arrayfire.h>
#include <mkldnn.h>

#include "flashlight/flashlight/autograd/Functions.h"
#include "flashlight/flashlight/autograd/Utils.h"
#include "flashlight/flashlight/autograd/Variable.h"
#include "flashlight/flashlight/autograd/backend/cpu/MkldnnUtils.h"
#include "flashlight/flashlight/common/DevicePtr.h"

using namespace mkldnn;

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
  memory::dims inputDims =
      detail::convertAfToMklDnnDims({input.dims(kBatchSizeIdx),
                                     input.dims(kChannelSizeIdx),
                                     input.dims(kHIdx),
                                     input.dims(kWIdx)});
  memory::dims outputDims =
      detail::convertAfToMklDnnDims({input.dims(kBatchSizeIdx),
                                     input.dims(kChannelSizeIdx),
                                     output.dims(kHIdx),
                                     output.dims(kWIdx)});
  memory::dims windowDims = {wy, wx};
  memory::dims strideDims = {sy, sx};
  std::vector<int> paddingDims = {py, px};

  auto dataType = detail::mkldnnMapToType(input.type());
  auto formatNCHW = memory::format::nchw;
  auto formatAny = memory::format::any;

  // Memory desc
  auto inputMD = memory::desc({inputDims}, dataType, formatNCHW);
  auto outputMD = memory::desc({outputDims}, dataType, formatAny);

  // Memory
  auto mkldnnEngine = detail::MkldnnEngine::getInstance().getEngine();
  DevicePtr inputRaw(input.array());
  auto inputMemoryInit = memory(
      {{{inputDims}, dataType, formatNCHW}, mkldnnEngine}, inputRaw.get());
  DevicePtr outputRaw(output);
  auto outputMemoryInit = memory(
      {{{outputDims}, dataType, formatNCHW}, mkldnnEngine}, outputRaw.get());

  // Choose a mode based on whether gradients are needed
  auto forwardMode =
      input.isCalcGrad() ? prop_kind::forward : prop_kind::forward_inference;

  // Descriptors
  auto poolingMode = detail::mkldnnMapToPoolingMode(mode);
  auto desc = pooling_forward::desc(
      forwardMode,
      poolingMode,
      inputMD,
      outputMD,
      strideDims,
      windowDims,
      paddingDims,
      paddingDims,
      padding_kind::zero);
  auto primDesc = pooling_forward::primitive_desc(desc, mkldnnEngine);

  // Network
  std::vector<primitive> network;
  // Reorder if needed
  auto inputPrimDesc = primDesc.src_primitive_desc();
  auto outputPrimDesc = primDesc.dst_primitive_desc();
  auto inputMemory =
      detail::mkldnnAlignOrdering(network, inputMemoryInit, inputPrimDesc);
  auto outputMemory = outputMemoryInit;
  if (outputMemoryInit.get_primitive_desc() !=
      memory::primitive_desc(outputPrimDesc)) {
    outputMemory = memory(outputPrimDesc);
  }
  // Workspace and layer (only training mode requires a workspace)
  std::shared_ptr<memory> workspaceMemory; // no default ctors
  std::shared_ptr<pooling_forward> pooling;
  if (input.isCalcGrad()) {
    workspaceMemory =
        std::make_shared<memory>(primDesc.workspace_primitive_desc());
    pooling = std::make_shared<pooling_forward>(
        primDesc, inputMemory, outputMemory, *workspaceMemory);
  } else {
    pooling =
        std::make_shared<pooling_forward>(primDesc, inputMemory, outputMemory);
  }
  network.push_back(*pooling);

  // Add output reordering if needed
  if (outputMemory != outputMemoryInit) {
    network.push_back(mkldnn::reorder(outputMemory, outputMemoryInit));
  }

  detail::MkldnnStream::getInstance().getStream().submit(network);

  auto gradFunc =
      [dataType,
       formatNCHW,
       inputDimsRaw, // need to pass if inputs are empty
       primDesc, // forward desc
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
        auto mkldnnEngineBwd = detail::MkldnnEngine::getInstance().getEngine();

        // Memory
        DevicePtr gradInputRaw(gradInput.array());
        auto gradInputMemoryInit = memory(
            {{{inputDims}, dataType, formatNCHW}, mkldnnEngineBwd},
            gradInputRaw.get());
        DevicePtr gradOutputRaw(grad_output.array());
        auto gradOutputMemoryInit = memory(
            {{{outputDims}, dataType, formatNCHW}, mkldnnEngineBwd},
            gradOutputRaw.get());

        // Descriptors
        // Memory descriptors from initialized memory must be used since
        // pooling_backward descriptors require an ordering
        auto gradInputMD = gradInputMemoryInit.get_primitive_desc().desc();
        auto gradOutputMD = gradOutputMemoryInit.get_primitive_desc().desc();
        auto bwdDesc = pooling_backward::desc(
            poolingMode,
            gradInputMD,
            gradOutputMD,
            strideDims,
            windowDims,
            paddingDims,
            paddingDims,
            padding_kind::zero);
        // Pass forward descriptor as a hint
        auto bwdPrimDesc = pooling_backward::primitive_desc(
            bwdDesc, mkldnnEngineBwd, primDesc);

        std::vector<primitive> networkBackward;
        // Reorder output memory if required
        auto gradOutputMemory = detail::mkldnnAlignOrdering(
            networkBackward,
            gradOutputMemoryInit,
            outputMemory.get_primitive_desc());

        auto poolBwd = pooling_backward(
            bwdPrimDesc,
            gradOutputMemory,
            *workspaceMemory, // workspace memory from forward
            gradInputMemoryInit);
        networkBackward.push_back(poolBwd);

        detail::MkldnnStream::getInstance().getStream().submit(networkBackward);

        in.addGrad(gradInput);
      };

  return Variable(output, {input.withoutData()}, gradFunc);
}

} // namespace fl
