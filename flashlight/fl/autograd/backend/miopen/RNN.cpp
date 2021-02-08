/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <vector>

#include <miopen/miopen.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/autograd/backend/miopen/MiOpenUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/MiOpenUtils.h"

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

  auto handle = fl::getMiOpenHandle();

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

    // MIOPEN_CHECK_ERR(miopenRNNBackwardData(
    //     miopen.handle() /*handle*/, rnn_desc.handle() /*rnnDesc*/,
    //     model_dims.seq_length /*seqLength*/, output_desc.handles() /*yDesc*/,
    //     output_data.opaque() /*y*/, output_desc.handles() /*dyDesc*/,
    //     output_backprop_data.opaque() /*dy*/, output_h_desc.handle()
    //     /*dhyDesc*/, output_h_backprop_data.opaque() /*dhy*/,
    //     output_c_desc.handle() /*dcyDesc*/,
    //     output_c_backprop_data.opaque() /*dcy*/,
    //     rnn_desc.params_handle() /*wDesc*/, params.opaque() /*w*/,
    //     input_h_desc.handle() /*hxDesc*/, input_h_data.opaque() /*hx*/,
    //     input_c_desc.handle() /*cxDesc*/, input_c_data.opaque() /*cx*/,
    //     input_desc.handles() /*dxDesc*/, input_backprop_data->opaque()
    //     /*dx*/, input_h_desc.handle() /*dhxDesc*/,
    //     input_h_backprop_data->opaque() /*dhx*/,
    //     input_c_desc.handle() /*dcxDesc*/,
    //     input_c_backprop_data->opaque() /*dcx*/, workspace.opaque()
    //     /*workspace*/, workspace.size() /*workSpaceSizeInBytes*/,
    //     reserve_space_data->opaque() /*reserveSpace*/,
    //     reserve_space_data->size() /*reserveSpaceSizeInBytes*/));

    MIOPEN_CHECK_ERR(miopenRNNBackwardData(
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
    TensorDescriptorArray xDescs(
        seqLength, x.type(), {1, 1, inputSize, batchSize});
    Variable dw(
        af::constant(0, weightsArray.dims(), weightsArray.type()), false);

    FilterDescriptor dwDesc(dw);

    {
      DevicePtr xRaw(x);
      DevicePtr dwRaw(dw.array());
      DevicePtr hxRaw(hxArray);

      MIOPEN_CHECK_ERR(miopenRNNBackwardWeights(
          handle,
          rnnDesc.descriptor,
          seqLength,
          xDescs.descriptors,
          xRaw.get(),
          hxDesc.descriptor,
          hxRaw.get(),
          yDesc.descriptors,
          yRaw.get(),
          dwDesc.descriptor,
          dwRaw.get(),
          workspaceRaw.get(),
          workspaceSize,
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
  if (input.type() != f32) {
    throw std::invalid_argument("MiOpen RNN supports only f32 input type");
  }

  FL_VARIABLE_DTYPES_MATCH_CHECK(input, hiddenState, cellState, weights);

  auto& x = input.array();

  af::array hxArray = hiddenState.array();
  af::array cxArray = cellState.array();

  DropoutDescriptor dropout(dropProb);
  RNNDescriptor rnnDesc(
      input.type(), hiddenSize, numLayers, mode, bidirectional, dropout);

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

  auto handle = fl::getMiOpenHandle();

  // miopenGetRNNLayerParamSize(miopenHandle_t handle, miopenRNNDescriptor_t
  // rnnDesc, const int layer, miopenTensorDescriptor_t xDesc, const int
  // paramID, size_t *numBytes)

  // size_t paramSize;
  // MIOPEN_CHECK_ERR(cudnnGetRNNParamsSize(
  //     handle,
  //     rnnDesc.descriptor,
  //     xDescs.descriptors[0],
  //     &paramSize,
  //     cudnnMapToType(weights.array().type())));
  // if (paramSize != weights.array().bytes()) {
  //   throw std::invalid_argument(
  //       "invalid # of parameters or wrong input shape for RNN");
  // }
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
  MIOPEN_CHECK_ERR(miopenGetRNNWorkspaceSize(
      handle,
      rnnDesc.descriptor,
      seqLength,
      xDescs.descriptors,
      &workspaceSize));
  af::array workspace(workspaceSize, af::dtype::b8);

  size_t reserveSize;
  MIOPEN_CHECK_ERR(miopenGetRNNTrainingReserveSize(
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

    MIOPEN_CHECK_ERR(miopenRNNForwardTraining(
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
