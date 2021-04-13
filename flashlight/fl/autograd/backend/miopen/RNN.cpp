/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/Functions.h"

#include <cudnn.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

namespace {
struct RNNGradData {
  af::array dy;
  af::array dhy;
  af::array dcy;
};
} // namespace

namespace fl {
void rnnBackward(
    std::vector<Variable>& inputs,
    const std::shared_ptr<struct RNNGradData> gradData,
    const af::array& y,
    size_t workspaceSize,
    size_t reserveSize,
    af::array reserveSpace,
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

  af::array hxArray = hx.array();
  af::array cxArray = cx.array();
  af::array weightsArray = weights.array();

  if (!(input.isCalcGrad() || hx.isCalcGrad() || cx.isCalcGrad() ||
        weights.isCalcGrad())) {
    return;
  }

  auto handle = getCudnnHandle();

  auto& x = input.array();
  auto dims = x.dims();
  int inputSize = dims[0];
  int batchSize = dims[1];
  int seqLength = dims[2];
  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  DropoutDescriptor dropout(dropProb);
  RNNDescriptor rnnDesc(
      input.type(), hiddenSize, numLayers, mode, bidirectional, dropout);
  if (input.type() == f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
  }

  TensorDescriptorArray yDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  TensorDescriptorArray dyDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  af::dim4 hDims = {1, hiddenSize, batchSize, totalLayers};
  TensorDescriptor dhyDesc(x.type(), hDims);
  TensorDescriptor dcyDesc(x.type(), hDims);
  TensorDescriptor hxDesc(x.type(), hDims);
  TensorDescriptor cxDesc(x.type(), hDims);

  Variable dhx(af::array(hxArray.dims(), hxArray.type()), false);
  Variable dcx(af::array(cxArray.dims(), cxArray.type()), false);
  TensorDescriptor dhxDesc(x.type(), hDims);
  TensorDescriptor dcxDesc(x.type(), hDims);

  FilterDescriptor wDesc(weightsArray);

  Variable dx(af::array(input.dims(), input.type()), false);
  TensorDescriptorArray dxDescs(
      seqLength, dx.type(), {1, 1, inputSize, batchSize});

  af::array workspace(workspaceSize, af::dtype::b8);

  auto& dy = gradData->dy;
  if (dy.isempty()) {
    dy = af::constant(0.0, y.dims(), y.type());
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

    DevicePtr wRaw(weightsArray);

    DevicePtr hxRaw(hxArray);
    DevicePtr cxRaw(cxArray);

    DevicePtr dxRaw(dx.array());
    DevicePtr dhxRaw(dhx.array());
    DevicePtr dcxRaw(dcx.array());

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
    if (input.type() == f16) {
      CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
          rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else {
      CUDNN_CHECK_ERR(
          cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
    }
    TensorDescriptorArray xDescs(
        seqLength, x.type(), {1, 1, inputSize, batchSize});
    Variable dw(
        af::constant(0, weightsArray.dims(), weightsArray.type()), false);

    FilterDescriptor dwDesc(dw);

    {
      DevicePtr xRaw(x);
      DevicePtr dwRaw(dw.array());
      DevicePtr hxRaw(hxArray);

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

  auto& x = input.array();

  af::array hxArray = hiddenState.array();
  af::array cxArray = cellState.array();

  DropoutDescriptor dropout(dropProb);
  RNNDescriptor rnnDesc(
      input.type(), hiddenSize, numLayers, mode, bidirectional, dropout);
  if (input.type() == f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
  }

  auto dims = x.dims();

  int inputSize = dims[0];
  int batchSize = dims[1];
  int seqLength = dims[2];

  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  TensorDescriptorArray xDescs(
      seqLength, x.type(), {1, 1, inputSize, batchSize});

  if (!hxArray.isempty() &&
      !(hxArray.dims(0) == hiddenSize && hxArray.dims(1) == batchSize &&
        hxArray.dims(2) == totalLayers)) {
    throw std::invalid_argument("invalid hidden state dims for RNN");
  }

  if (!cxArray.isempty() &&
      !(mode == RnnMode::LSTM && cxArray.dims(0) == hiddenSize &&
        cxArray.dims(1) == batchSize && cxArray.dims(2) == totalLayers)) {
    throw std::invalid_argument("invalid cell state dims for RNN");
  }

  af::dim4 hDims = {1, hiddenSize, batchSize, totalLayers};
  TensorDescriptor hxDesc(x.type(), hDims);
  TensorDescriptor cxDesc(x.type(), hDims);

  auto handle = getCudnnHandle();

  size_t paramSize;
  CUDNN_CHECK_ERR(cudnnGetRNNParamsSize(
      handle,
      rnnDesc.descriptor,
      xDescs.descriptors[0],
      &paramSize,
      cudnnMapToType(weights.array().type())));
  if (paramSize != weights.array().bytes()) {
    throw std::invalid_argument(
        "invalid # of parameters or wrong input shape for RNN");
  }
  FilterDescriptor wDesc(weights);

  af::array y(outSize, batchSize, seqLength, input.type());
  TensorDescriptorArray yDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  af::array hy({hiddenSize, batchSize, totalLayers}, x.type());
  TensorDescriptor hyDesc(x.type(), hDims);

  af::array cy;
  if (mode == RnnMode::LSTM) {
    cy = af::array(hy.dims(), x.type());
  }

  TensorDescriptor cyDesc(x.type(), hDims);

  size_t workspaceSize;
  CUDNN_CHECK_ERR(cudnnGetRNNWorkspaceSize(
      handle,
      rnnDesc.descriptor,
      seqLength,
      xDescs.descriptors,
      &workspaceSize));
  af::array workspace(workspaceSize, af::dtype::b8);

  size_t reserveSize;
  CUDNN_CHECK_ERR(cudnnGetRNNTrainingReserveSize(
      handle, rnnDesc.descriptor, seqLength, xDescs.descriptors, &reserveSize));
  af::array reserveSpace(reserveSize, af::dtype::b8);
  {
    DevicePtr xRaw(x);
    DevicePtr hxRaw(hxArray);
    DevicePtr cxRaw(cxArray);
    DevicePtr wRaw(weights.array());
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

  Variable dummy(
      af::array(), {input, hiddenState, cellState, weights}, gradFunc);

  auto dyGradFunc =
      [gradData](std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(af::array(), false));
        }
        gradData->dy = gradOutput.array();
      };

  auto dhyGradFunc =
      [gradData](std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(af::array(), false));
        }
        gradData->dhy = gradOutput.array();
      };

  auto dcyGradFunc =
      [gradData](std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(af::array(), false));
        }
        gradData->dcy = gradOutput.array();
      };

  Variable yv(y, {dummy}, dyGradFunc);
  Variable hyv(hy, {dummy}, dhyGradFunc);
  Variable cyv(cy, {dummy}, dcyGradFunc);
  return std::make_tuple(yv, hyv, cyv);
}

} // namespace fl
