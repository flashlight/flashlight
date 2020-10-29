/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/flashlight/autograd/Functions.h"

#include <cudnn.h>

#include "flashlight/flashlight/autograd/Variable.h"
#include "flashlight/flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/flashlight/common/DevicePtr.h"

namespace {
struct RNNGradData {
  af::array dy;
  af::array dhy;
  af::array dcy;
};
} // namespace

namespace fl {
void rnn_backward(
    std::vector<Variable>& inputs,
    const std::shared_ptr<struct RNNGradData> grad_data,
    const af::array& y,
    size_t workspace_size,
    size_t reserve_size,
    af::array reserve_space,
    int num_layers,
    int hidden_size,
    RnnMode mode,
    bool bidirectional,
    float drop_prob) {
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
  int input_size = dims[0];
  int batch_size = dims[1];
  int seq_length = dims[2];
  int total_layers = num_layers * (bidirectional ? 2 : 1);
  int out_size = hidden_size * (bidirectional ? 2 : 1);

  DropoutDescriptor dropout(drop_prob);
  RNNDescriptor rnn_desc(
      input.type(), hidden_size, num_layers, mode, bidirectional, dropout);
  if (input.type() == f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnn_desc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnn_desc.descriptor, CUDNN_DEFAULT_MATH));
  }

  TensorDescriptorArray y_descs(
      seq_length, y.type(), {1, 1, out_size, batch_size});

  TensorDescriptorArray dy_descs(
      seq_length, y.type(), {1, 1, out_size, batch_size});

  af::dim4 h_dims = {1, hidden_size, batch_size, total_layers};
  TensorDescriptor dhy_desc(x.type(), h_dims);
  TensorDescriptor dcy_desc(x.type(), h_dims);
  TensorDescriptor hx_desc(x.type(), h_dims);
  TensorDescriptor cx_desc(x.type(), h_dims);

  Variable dhx(af::array(hxArray.dims(), hxArray.type()), false);
  Variable dcx(af::array(cxArray.dims(), cxArray.type()), false);
  TensorDescriptor dhx_desc(x.type(), h_dims);
  TensorDescriptor dcx_desc(x.type(), h_dims);

  FilterDescriptor w_desc(weightsArray);

  Variable dx(af::array(input.dims(), input.type()), false);
  TensorDescriptorArray dx_descs(
      seq_length, dx.type(), {1, 1, input_size, batch_size});

  af::array workspace(workspace_size, af::dtype::b8);

  auto& dy = grad_data->dy;
  if (dy.isempty()) {
    dy = af::constant(0.0, y.dims(), y.type());
  }
  auto& dhy = grad_data->dhy;
  auto& dcy = grad_data->dcy;

  DevicePtr y_raw(y);
  DevicePtr workspace_raw(workspace);
  DevicePtr reserve_space_raw(reserve_space);

  {
    DevicePtr dy_raw(dy); // Has to be set to 0 if empty
    DevicePtr dhy_raw(dhy);
    DevicePtr dcy_raw(dcy);

    DevicePtr w_raw(weightsArray);

    DevicePtr hx_raw(hxArray);
    DevicePtr cx_raw(cxArray);

    DevicePtr dx_raw(dx.array());
    DevicePtr dhx_raw(dhx.array());
    DevicePtr dcx_raw(dcx.array());

    /* We need to update reserve_space even if we just want the
     * weight gradients. */
    CUDNN_CHECK_ERR(cudnnRNNBackwardData(
        handle,
        rnn_desc.descriptor,
        seq_length,
        y_descs.descriptors,
        y_raw.get(),
        dy_descs.descriptors,
        dy_raw.get(),
        dhy_desc.descriptor,
        dhy_raw.get(),
        dcy_desc.descriptor,
        dcy_raw.get(),
        w_desc.descriptor,
        w_raw.get(),
        hx_desc.descriptor,
        hx_raw.get(),
        cx_desc.descriptor,
        cx_raw.get(),
        dx_descs.descriptors,
        dx_raw.get(),
        dhx_desc.descriptor,
        dhx_raw.get(),
        dcx_desc.descriptor,
        dcx_raw.get(),
        workspace_raw.get(),
        workspace_size,
        reserve_space_raw.get(),
        reserve_size));
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
          rnn_desc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else {
      CUDNN_CHECK_ERR(
          cudnnSetRNNMatrixMathType(rnn_desc.descriptor, CUDNN_DEFAULT_MATH));
    }
    TensorDescriptorArray x_descs(
        seq_length, x.type(), {1, 1, input_size, batch_size});
    Variable dw(
        af::constant(0, weightsArray.dims(), weightsArray.type()), false);

    FilterDescriptor dw_desc(dw);

    {
      DevicePtr x_raw(x);
      DevicePtr dw_raw(dw.array());
      DevicePtr hx_raw(hxArray);

      CUDNN_CHECK_ERR(cudnnRNNBackwardWeights(
          handle,
          rnn_desc.descriptor,
          seq_length,
          x_descs.descriptors,
          x_raw.get(),
          hx_desc.descriptor,
          hx_raw.get(),
          y_descs.descriptors,
          y_raw.get(),
          workspace_raw.get(),
          workspace_size,
          dw_desc.descriptor,
          dw_raw.get(),
          reserve_space_raw.get(),
          reserve_size));
    }
    weights.addGrad(dw);
  }
} // namespace fl

std::tuple<Variable, Variable, Variable> rnn(
    const Variable& input,
    const Variable& hidden_state,
    const Variable& cell_state,
    const Variable& weights,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional,
    float drop_prob) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(input, hidden_state, cell_state, weights);

  auto& x = input.array();

  af::array hxArray = hidden_state.array();
  af::array cxArray = cell_state.array();

  DropoutDescriptor dropout(drop_prob);
  RNNDescriptor rnn_desc(
      input.type(), hidden_size, num_layers, mode, bidirectional, dropout);
  if (input.type() == f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnn_desc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnn_desc.descriptor, CUDNN_DEFAULT_MATH));
  }

  auto dims = x.dims();

  int input_size = dims[0];
  int batch_size = dims[1];
  int seq_length = dims[2];

  int total_layers = num_layers * (bidirectional ? 2 : 1);
  int out_size = hidden_size * (bidirectional ? 2 : 1);

  TensorDescriptorArray x_descs(
      seq_length, x.type(), {1, 1, input_size, batch_size});

  if (!hxArray.isempty() &&
      !(hxArray.dims(0) == hidden_size && hxArray.dims(1) == batch_size &&
        hxArray.dims(2) == total_layers)) {
    throw std::invalid_argument("invalid hidden state dims for RNN");
  }

  if (!cxArray.isempty() &&
      !(mode == RnnMode::LSTM && cxArray.dims(0) == hidden_size &&
        cxArray.dims(1) == batch_size && cxArray.dims(2) == total_layers)) {
    throw std::invalid_argument("invalid cell state dims for RNN");
  }

  af::dim4 h_dims = {1, hidden_size, batch_size, total_layers};
  TensorDescriptor hx_desc(x.type(), h_dims);
  TensorDescriptor cx_desc(x.type(), h_dims);

  auto handle = getCudnnHandle();

  size_t param_size;
  CUDNN_CHECK_ERR(cudnnGetRNNParamsSize(
      handle,
      rnn_desc.descriptor,
      x_descs.descriptors[0],
      &param_size,
      cudnnMapToType(weights.array().type())));
  if (param_size != weights.array().bytes()) {
    throw std::invalid_argument(
        "invalid # of parameters or wrong input shape for RNN");
  }
  FilterDescriptor w_desc(weights);

  af::array y(out_size, batch_size, seq_length, input.type());
  TensorDescriptorArray y_descs(
      seq_length, y.type(), {1, 1, out_size, batch_size});

  af::array hy({hidden_size, batch_size, total_layers}, x.type());
  TensorDescriptor hy_desc(x.type(), h_dims);

  af::array cy;
  if (mode == RnnMode::LSTM) {
    cy = af::array(hy.dims(), x.type());
  }

  TensorDescriptor cy_desc(x.type(), h_dims);

  size_t workspace_size;
  CUDNN_CHECK_ERR(cudnnGetRNNWorkspaceSize(
      handle,
      rnn_desc.descriptor,
      seq_length,
      x_descs.descriptors,
      &workspace_size));
  af::array workspace(workspace_size, af::dtype::b8);

  size_t reserve_size;
  CUDNN_CHECK_ERR(cudnnGetRNNTrainingReserveSize(
      handle,
      rnn_desc.descriptor,
      seq_length,
      x_descs.descriptors,
      &reserve_size));
  af::array reserve_space(reserve_size, af::dtype::b8);
  {
    DevicePtr x_raw(x);
    DevicePtr hx_raw(hxArray);
    DevicePtr cx_raw(cxArray);
    DevicePtr w_raw(weights.array());
    DevicePtr y_raw(y);
    DevicePtr hy_raw(hy);
    DevicePtr cy_raw(cy);
    DevicePtr workspace_raw(workspace);
    DevicePtr reserve_space_raw(reserve_space);

    CUDNN_CHECK_ERR(cudnnRNNForwardTraining(
        handle,
        rnn_desc.descriptor,
        seq_length,
        x_descs.descriptors,
        x_raw.get(),
        hx_desc.descriptor,
        hx_raw.get(),
        cx_desc.descriptor,
        cx_raw.get(),
        w_desc.descriptor,
        w_raw.get(),
        y_descs.descriptors,
        y_raw.get(),
        hy_desc.descriptor,
        hy_raw.get(),
        cy_desc.descriptor,
        cy_raw.get(),
        workspace_raw.get(),
        workspace_size,
        reserve_space_raw.get(),
        reserve_size));
  }
  auto grad_data = std::make_shared<RNNGradData>();

  auto gradFunc = [y,
                   workspace_size,
                   reserve_size,
                   reserve_space,
                   num_layers,
                   hidden_size,
                   mode,
                   bidirectional,
                   drop_prob,
                   grad_data](
                      std::vector<Variable>& inputs,
                      const Variable& /* grad_output */) {
    rnn_backward(
        inputs,
        grad_data,
        y,
        workspace_size,
        reserve_size,
        reserve_space,
        num_layers,
        hidden_size,
        mode,
        bidirectional,
        drop_prob);
  };

  Variable dummy(
      af::array(), {input, hidden_state, cell_state, weights}, gradFunc);

  auto dy_gradFunc =
      [grad_data](std::vector<Variable>& inputs, const Variable& grad_output) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(af::array(), false));
        }
        grad_data->dy = grad_output.array();
      };

  auto dhy_gradFunc =
      [grad_data](std::vector<Variable>& inputs, const Variable& grad_output) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(af::array(), false));
        }
        grad_data->dhy = grad_output.array();
      };

  auto dcy_gradFunc =
      [grad_data](std::vector<Variable>& inputs, const Variable& grad_output) {
        if (!inputs[0].isGradAvailable()) {
          inputs[0].addGrad(Variable(af::array(), false));
        }
        grad_data->dcy = grad_output.array();
      };

  Variable yv(y, {dummy}, dy_gradFunc);
  Variable hyv(hy, {dummy}, dhy_gradFunc);
  Variable cyv(cy, {dummy}, dcy_gradFunc);
  return std::make_tuple(yv, hyv, cyv);
}

} // namespace fl
