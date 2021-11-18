/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/Functions.h"

#include <cudnn.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

namespace {
struct RNNGradData {
  fl::Tensor dy;
  fl::Tensor dhy;
  fl::Tensor dcy;
};
} // namespace

namespace fl {
void rnnBackward(
    std::vector<Variable>& inputs,
    const std::shared_ptr<struct RNNGradData> gradData,
    const Tensor& y,
    size_t workspaceSize,
    size_t reserveSize,
    Tensor reserveSpace,
    int numLayers,
    int hiddenSize,
    RnnMode mode,
    bool bidirectional,
    float dropProb) {
  if (inputs.size() != 4) {
    throw std::invalid_argument("wrong # of inputs for RNN");
  }

  auto input = inputs[0];
  auto hx = inputs[1];
  auto cx = inputs[2];
  auto weights = inputs[3];

  Tensor hxTensor = hx.tensor();
  Tensor cxTensor = cx.tensor();
  Tensor weightsTensor = weights.tensor();

  if (!(input.isCalcGrad() || hx.isCalcGrad() || cx.isCalcGrad() ||
        weights.isCalcGrad())) {
    return;
  }

  auto handle = getCudnnHandle();

  auto& x = input.tensor();
  auto dims = x.shape();
  int inputSize = dims[0];
  int batchSize = dims.ndim() < 2 ? 1 : dims[1];
  int seqLength = dims.ndim() < 3 ? 1 : dims[2];
  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  DropoutDescriptor dropout(dropProb);
  RNNDescriptor rnnDesc(
      input.type(), hiddenSize, numLayers, mode, bidirectional, dropout);
  if (input.type() == fl::dtype::f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
  }

  TensorDescriptorArray yDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  TensorDescriptorArray dyDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  Shape hDims = {1, hiddenSize, batchSize, totalLayers};
  TensorDescriptor dhyDesc(x.type(), hDims);
  TensorDescriptor dcyDesc(x.type(), hDims);
  TensorDescriptor hxDesc(x.type(), hDims);
  TensorDescriptor cxDesc(x.type(), hDims);

  Variable dhx(Tensor(hxTensor.shape(), hxTensor.type()), false);
  Variable dcx(Tensor(cxTensor.shape(), cxTensor.type()), false);
  TensorDescriptor dhxDesc(x.type(), hDims);
  TensorDescriptor dcxDesc(x.type(), hDims);

  FilterDescriptor wDesc(weightsTensor);

  Variable dx(Tensor(input.dims(), input.type()), false);
  TensorDescriptorArray dxDescs(
      seqLength, dx.type(), {1, 1, inputSize, batchSize});

  Tensor workspace({static_cast<long long>(workspaceSize)}, fl::dtype::b8);

  auto& dy = gradData->dy;
  if (dy.isEmpty()) {
    dy = fl::full(y.shape(), 0.0, y.type());
  }
  auto& dhy = gradData->dhy;
  auto& dcy = gradData->dcy;

  DevicePtr yRaw(y);
  DevicePtr workspaceRaw(workspace);
  DevicePtr reserveSpaceRaw(reserveSpace);

  {
    DevicePtr dyRaw(dy); // Has to be set to 0 if empty
    DevicePtr dhyRaw(dhy);
    DevicePtr dcyRaw(dcy);

    DevicePtr wRaw(weightsTensor);

    DevicePtr hxRaw(hxTensor);
    DevicePtr cxRaw(cxTensor);

    DevicePtr dxRaw(dx.tensor());
    DevicePtr dhxRaw(dhx.tensor());
    DevicePtr dcxRaw(dcx.tensor());

    /* We need to update reserveSpace even if we just want the
     * weight gradients. */
    CUDNN_CHECK_ERR(cudnnRNNBackwardData(
        handle,
        rnnDesc.descriptor,
        seqLength,
        yDesc.descriptors,
        yRaw.get(),
        dyDesc.descriptors,
        dyRaw.get(),
        dhyDesc.descriptor,
        dhyRaw.get(),
        dcyDesc.descriptor,
        dcyRaw.get(),
        wDesc.descriptor,
        wRaw.get(),
        hxDesc.descriptor,
        hxRaw.get(),
        cxDesc.descriptor,
        cxRaw.get(),
        dxDescs.descriptors,
        dxRaw.get(),
        dhxDesc.descriptor,
        dhxRaw.get(),
        dcxDesc.descriptor,
        dcxRaw.get(),
        workspaceRaw.get(),
        workspaceSize,
        reserveSpaceRaw.get(),
        reserveSize));
  }
  if (input.isCalcGrad()) {
    input.addGrad(dx);
  }

  if (hx.isCalcGrad() && !hx.isempty()) {
    hx.addGrad(dhx.as(hx.type()));
  }

  if (cx.isCalcGrad() && !cx.isempty()) {
    cx.addGrad(dcx.as(cx.type()));
  }

  if (weights.isCalcGrad()) {
    if (input.type() == fl::dtype::f16) {
      CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
          rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else {
      CUDNN_CHECK_ERR(
          cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
    }
    TensorDescriptorArray xDescs(
        seqLength, x.type(), {1, 1, inputSize, batchSize});
    Variable dw(
        fl::full(weightsTensor.shape(), 0, weightsTensor.type()), false);

    FilterDescriptor dwDesc(dw);

    {
      DevicePtr xRaw(x.asContiguousTensor());
      DevicePtr dwRaw(dw.tensor());
      DevicePtr hxRaw(hxTensor);

      CUDNN_CHECK_ERR(cudnnRNNBackwardWeights(
          handle,
          rnnDesc.descriptor,
          seqLength,
          xDescs.descriptors,
          xRaw.get(),
          hxDesc.descriptor,
          hxRaw.get(),
          yDesc.descriptors,
          yRaw.get(),
          workspaceRaw.get(),
          workspaceSize,
          dwDesc.descriptor,
          dwRaw.get(),
          reserveSpaceRaw.get(),
          reserveSize));
    }
    weights.addGrad(dw);
  }
} // namespace fl

std::tuple<Variable, Variable, Variable> rnn(
    const Variable& input,
    const Variable& hiddenState,
    const Variable& cellState,
    const Variable& weights,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropProb) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(input, hiddenState, cellState, weights);

  Tensor x = input.tensor().asContiguousTensor();

  Tensor hxTensor = hiddenState.tensor().asContiguousTensor();
  Tensor cxTensor = cellState.tensor().asContiguousTensor();

  DropoutDescriptor dropout(dropProb);
  RNNDescriptor rnnDesc(
      input.type(), hiddenSize, numLayers, mode, bidirectional, dropout);
  if (input.type() == fl::dtype::f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
  }

  auto dims = x.shape();
  int inputSize = dims[0];
  int batchSize = dims.ndim() < 2 ? 1 : dims[1];
  int seqLength = dims.ndim() < 3 ? 1 : dims[2];

  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  TensorDescriptorArray xDescs(
      seqLength, x.type(), {1, 1, inputSize, batchSize});

  if (!hxTensor.isEmpty()) {
    auto hxDims = hxTensor.shape();
    int hxHiddenSize = hxDims[0];
    int hxBatchSize = hxTensor.ndim() < 2 ? 1 : hxDims[1];
    int hxTotalLayers = hxTensor.ndim() < 3 ? 1 : hxDims[2];

    if (!(hxHiddenSize == hiddenSize && hxBatchSize == batchSize &&
          hxTotalLayers == totalLayers)) {
      throw std::invalid_argument("invalid hidden state dims for RNN");
    }
  }

  if (!cxTensor.isEmpty() &&
      !(mode == RnnMode::LSTM && cxTensor.dim(0) == hiddenSize &&
        cxTensor.dim(1) == batchSize && cxTensor.dim(2) == totalLayers)) {
    throw std::invalid_argument("invalid cell state dims for RNN");
  }

  Shape hDims = {1, hiddenSize, batchSize, totalLayers};
  TensorDescriptor hxDesc(x.type(), hDims);
  TensorDescriptor cxDesc(x.type(), hDims);

  auto handle = getCudnnHandle();

  size_t paramSize;
  CUDNN_CHECK_ERR(cudnnGetRNNParamsSize(
      handle,
      rnnDesc.descriptor,
      xDescs.descriptors[0],
      &paramSize,
      cudnnMapToType(weights.tensor().type())));
  if (paramSize != weights.array().bytes()) {
    throw std::invalid_argument(
        "invalid # of parameters or wrong input shape for RNN");
  }
  FilterDescriptor wDesc(weights);

  Tensor y({outSize, batchSize, seqLength}, input.type());
  TensorDescriptorArray yDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  Tensor hy({hiddenSize, batchSize, totalLayers}, x.type());
  TensorDescriptor hyDesc(x.type(), hDims);

  Tensor cy;
  if (mode == RnnMode::LSTM) {
    cy = Tensor(hy.shape(), x.type());
  }

  TensorDescriptor cyDesc(x.type(), hDims);

  size_t workspaceSize;
  CUDNN_CHECK_ERR(cudnnGetRNNWorkspaceSize(
      handle,
      rnnDesc.descriptor,
      seqLength,
      xDescs.descriptors,
      &workspaceSize));
  Tensor workspace({static_cast<long long>(workspaceSize)}, fl::dtype::b8);

  size_t reserveSize;
  CUDNN_CHECK_ERR(cudnnGetRNNTrainingReserveSize(
      handle, rnnDesc.descriptor, seqLength, xDescs.descriptors, &reserveSize));
  Tensor reserveSpace({static_cast<long long>(reserveSize)}, fl::dtype::b8);
  {
    DevicePtr xRaw(x);
    DevicePtr hxRaw(hxTensor);
    DevicePtr cxRaw(cxTensor);
    DevicePtr wRaw(weights.tensor().asContiguousTensor());
    DevicePtr yRaw(y);
    DevicePtr hyRaw(hy);
    DevicePtr cyRaw(cy);
    DevicePtr workspaceRaw(workspace);
    DevicePtr reserveSpaceRaw(reserveSpace);

    CUDNN_CHECK_ERR(cudnnRNNForwardTraining(
        handle,
        rnnDesc.descriptor,
        seqLength,
        xDescs.descriptors,
        xRaw.get(),
        hxDesc.descriptor,
        hxRaw.get(),
        cxDesc.descriptor,
        cxRaw.get(),
        wDesc.descriptor,
        wRaw.get(),
        yDesc.descriptors,
        yRaw.get(),
        hyDesc.descriptor,
        hyRaw.get(),
        cyDesc.descriptor,
        cyRaw.get(),
        workspaceRaw.get(),
        workspaceSize,
        reserveSpaceRaw.get(),
        reserveSize));
  }
  auto gradData = std::make_shared<RNNGradData>();

  auto gradFunc = [y,
                   workspaceSize,
                   reserveSize,
                   reserveSpace,
                   numLayers,
                   hiddenSize,
                   mode,
                   bidirectional,
                   dropProb,
                   gradData](
                      std::vector<Variable>& inputs,
                      const Variable& /* gradOutput */) {
    rnnBackward(
        inputs,
        gradData,
        y,
        workspaceSize,
        reserveSize,
        reserveSpace,
        numLayers,
        hiddenSize,
        mode,
        bidirectional,
        dropProb);
  };

  Variable dummy(Tensor(), {input, hiddenState, cellState, weights}, gradFunc);

  auto dyGradFunc =
      [gradData](std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(Tensor(), false));
        }
        gradData->dy = gradOutput.tensor().asContiguousTensor();
      };

  auto dhyGradFunc =
      [gradData](std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(Tensor(), false));
        }
        gradData->dhy = gradOutput.tensor().asContiguousTensor();
      };

  auto dcyGradFunc =
      [gradData](std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(Tensor(), false));
        }
        gradData->dcy = gradOutput.tensor().asContiguousTensor();
      };

  Variable yv(y, {dummy}, dyGradFunc);
  Variable hyv(hy, {dummy}, dhyGradFunc);
  Variable cyv(cy, {dummy}, dcyGradFunc);
  return std::make_tuple(yv, hyv, cyv);
}

} // namespace fl
