/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/autograd/Functions.h"

#include <cudnn.h>

#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/common/DevicePtr.h"

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

  TensorDescriptorArray y_descs(
      seq_length, y.type(), {1, 1, out_size, batch_size});

  TensorDescriptorArray dy_descs(
      seq_length, y.type(), {1, 1, out_size, batch_size});

  af::dim4 h_dims = {1, hidden_size, batch_size, total_layers};
  TensorDescriptor dhy_desc(x.type(), h_dims);
  TensorDescriptor dcy_desc(x.type(), h_dims);
  TensorDescriptor hx_desc(x.type(), h_dims);
  TensorDescriptor cx_desc(x.type(), h_dims);

  Variable dhx(af::array(hx.dims(), hx.type()), false);
  Variable dcx(af::array(cx.dims(), cx.type()), false);
  TensorDescriptor dhx_desc(x.type(), h_dims);
  TensorDescriptor dcx_desc(x.type(), h_dims);

  FilterDescriptor w_desc(weights);

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

    DevicePtr w_raw(weights.array());

    DevicePtr hx_raw(hx.array());
    DevicePtr cx_raw(cx.array());

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
    hx.addGrad(dhx);
  }

  if (cx.isCalcGrad() && !cx.isempty()) {
    cx.addGrad(dcx);
  }

  if (weights.isCalcGrad()) {
    TensorDescriptorArray x_descs(
        seq_length, x.type(), {1, 1, input_size, batch_size});
    Variable dw(af::constant(0, weights.dims(), weights.type()), false);

    FilterDescriptor dw_desc(dw);

    {
      DevicePtr x_raw(x);
      DevicePtr dw_raw(dw.array());
      DevicePtr hx_raw(hx.array());

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
}

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
  auto& x = input.array();
  auto& hx = hidden_state.array();
  auto& cx = cell_state.array();

  DropoutDescriptor dropout(drop_prob);
  RNNDescriptor rnn_desc(
      input.type(), hidden_size, num_layers, mode, bidirectional, dropout);

  auto dims = x.dims();

  int input_size = dims[0];
  int batch_size = dims[1];
  int seq_length = dims[2];

  int total_layers = num_layers * (bidirectional ? 2 : 1);
  int out_size = hidden_size * (bidirectional ? 2 : 1);

  TensorDescriptorArray x_descs(
      seq_length, x.type(), {1, 1, input_size, batch_size});

  if (!hx.isempty() &&
      !(hx.dims(0) == hidden_size && hx.dims(1) == batch_size &&
        hx.dims(2) == total_layers)) {
    throw std::invalid_argument("invalid hidden state dims for RNN");
  }

  if (!cx.isempty() &&
      !(mode == RnnMode::LSTM && cx.dims(0) == hidden_size &&
        cx.dims(1) == batch_size && cx.dims(2) == total_layers)) {
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
    DevicePtr hx_raw(hx);
    DevicePtr cx_raw(cx);
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
