/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <arrayfire.h>
#include <dnnl.hpp>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cpu/DnnlUtils.h"

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
  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (maxAxis - minAxis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  auto dType = detail::dnnlMapToType(input.type());
  auto dnnlEngine = detail::DnnlEngine::getInstance().getEngine();
  auto formatX = dnnl::memory::format_tag::x;
  auto format2d = dnnl::memory::format_tag::nc;
  // DNNL requires NCHW order and expects row-major tensors.
  // The input tensor is in WHCN and column-major as per AF, which is
  // equivalent; no reorder needed
  auto formatNCHW = dnnl::memory::format_tag::nchw;

  // Prepare combined weights
  // If empty, user specifies affine to false. Both not trainable.
  auto weightNonempty = weight.isempty()
      ? Variable(af::constant(1.0, nfeatures, af::dtype::f32), false)
      : weight;
  auto biasNonempty = bias.isempty()
      ? Variable(af::constant(0.0, nfeatures, af::dtype::f32), false)
      : bias;

  // DNNL only accepts weight and bias as a combined input.
  // https://git.io/JLn9X
  auto weightsDnnl = af::join(0, weightNonempty.array(), biasNonempty.array());
  af::dim4 inDescDims;
  if (minAxis == 0) {
    inDescDims = af::dim4(1, 1, nfeatures, input.elements() / nfeatures);
  } else {
    int batchsz = 1;
    for (int i = maxAxis + 1; i < 4; ++i) {
      batchsz *= input.dims(i);
    }
    inDescDims = af::dim4(
        1, input.elements() / (nfeatures * batchsz), nfeatures, batchsz);
  }

  dnnl::memory::dims inputOutputDims = {
      inDescDims[kBatchSizeIdx],
      inDescDims[kChannelSizeIdx],
      inDescDims[kHIdx],
      inDescDims[kWIdx]};
  auto inputOutputMemDesc =
      dnnl::memory::desc({inputOutputDims}, dType, formatNCHW);
  dnnl::memory::dims weightsDnnlDims =
      detail::convertAfToDnnlDims({2, nfeatures});

  // Memory for forward
  const detail::DnnlMemoryWrapper inputMemory(
      input.array(), inputOutputDims, formatNCHW);
  const detail::DnnlMemoryWrapper outputMemory(
      output, inputOutputDims, formatNCHW);
  const detail::DnnlMemoryWrapper meanMemory(
      runningMean.array(), {runningMean.dims(0)}, formatX);
  const detail::DnnlMemoryWrapper varMemory(
      runningVar.array(), {runningVar.dims(0)}, formatX);
  // combined scale and shift (weight and bias)
  const detail::DnnlMemoryWrapper weightsMemory(
      weightsDnnl, weightsDnnlDims, format2d);
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

  /****************************************************************************/
  // Setup backward func

  auto gradFunc = [train,
                   epsilon,
                   nfeatures,
                   fwdPrimDesc,
                   outputMemDesc = outputMemory.getDescriptor(),
                   inputOutputDims,
                   formatNCHW,
                   format2d,
                   dType,
                   weightsDnnl,
                   weightsDnnlDims,
                   inputMemoryBwd = inputMemory.getMemory(),
                   meanMemInit = meanMemory.getMemory(),
                   varMemInit = varMemory.getMemory(),
                   weightsDnnlMemInit = weightsMemory.getMemory()](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) mutable {
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
    const detail::DnnlMemoryWrapper gradOutputMem(
        grad_output.array(), inputOutputDims, formatNCHW);
    const detail::DnnlMemoryWrapper gradInputMem(
        grad_input.array(), inputOutputDims, formatNCHW);
    const detail::DnnlMemoryWrapper gradWeightsMem(
        grad_weightsDNNL.array(), weightsDnnlDims, format2d);

    // Primitives and descriptors
    auto bwdDesc = dnnl::batch_normalization_backward::desc(
        dnnl::prop_kind::backward,
        gradOutputMem.getDescriptor(),
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
        {{DNNL_ARG_SRC, inputMemoryBwd},
         {DNNL_ARG_MEAN, meanMemInit},
         {DNNL_ARG_VARIANCE, varMemInit},
         {DNNL_ARG_SCALE_SHIFT, weightsDnnlMemInit},
         {DNNL_ARG_DIFF_SRC, gradInputMem.getMemory()},
         {DNNL_ARG_DIFF_DST, gradOutputMem.getMemory()},
         {DNNL_ARG_DIFF_SCALE_SHIFT, gradWeightsMem.getMemory()}}};
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
  // CPU backend DNNL doesn't support a momentum factor.
  // If momentum is enabled, throw.
  if (momentum == 0.0) {
    return batchnorm(
        input, weight, bias, runningMean, runningVar, axes, train, epsilon);
  } else {
    throw std::runtime_error("BatchNorm CPU backend doesn't support momentum.");
  }
}

} // namespace fl
