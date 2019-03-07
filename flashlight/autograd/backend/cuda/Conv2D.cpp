/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <cudnn.h>

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/common/DevicePtr.h"

namespace {

const cudnnConvolutionFwdPreference_t fwd_pref =
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
const cudnnConvolutionBwdDataPreference_t bwd_data_pref =
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
const cudnnConvolutionBwdFilterPreference_t bwd_filter_pref =
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;

const size_t fwd_mem_limit = 0;
const size_t bwd_data_mem_limit = 0;
const size_t bwd_filter_mem_limit = 0;

} // namespace

namespace fl {

Variable conv2d(
    const Variable& input,
    const Variable& weights,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups) {
  auto dummy_bias = Variable(af::array(), false);
  return conv2d(input, weights, dummy_bias, sx, sy, px, py, dx, dy, groups);
}

Variable conv2d(
    const Variable& input,
    const Variable& weights,
    const Variable& bias,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups) {
  auto has_bias = bias.elements() > 0;

  auto in_desc = TensorDescriptor(input);
  auto wt_desc = FilterDescriptor(weights);
  auto conv_desc = ConvDescriptor(input.type(), px, py, sx, sy, dx, dy, groups);

  std::array<int, 4> odims;
  CUDNN_CHECK_ERR(cudnnGetConvolutionNdForwardOutputDim(
      conv_desc.descriptor,
      in_desc.descriptor,
      wt_desc.descriptor,
      4,
      odims.data()));
  auto output = af::array(odims[3], odims[2], odims[1], odims[0], input.type());
  auto out_desc = TensorDescriptor(output);

  auto handle = getCudnnHandle();

  cudnnConvolutionFwdAlgo_t fwd_algo;
  CUDNN_CHECK_ERR(cudnnGetConvolutionForwardAlgorithm(
      handle,
      in_desc.descriptor,
      wt_desc.descriptor,
      conv_desc.descriptor,
      out_desc.descriptor,
      fwd_pref,
      fwd_mem_limit,
      &fwd_algo));

  size_t wspace_fwd_bytes = 0;
  CUDNN_CHECK_ERR(cudnnGetConvolutionForwardWorkspaceSize(
      handle,
      in_desc.descriptor,
      wt_desc.descriptor,
      conv_desc.descriptor,
      out_desc.descriptor,
      fwd_algo,
      &wspace_fwd_bytes));
  auto wspace = af::array(wspace_fwd_bytes, af::dtype::b8);
  {
    DevicePtr in_raw(input.array());
    DevicePtr wt_raw(weights.array());
    DevicePtr out_raw(output);
    DevicePtr wspace_raw(wspace);

    const void* one = kOne(input.type());
    const void* zero = kZero(input.type());

    CUDNN_CHECK_ERR(cudnnConvolutionForward(
        handle,
        one,
        in_desc.descriptor,
        in_raw.get(),
        wt_desc.descriptor,
        wt_raw.get(),
        conv_desc.descriptor,
        fwd_algo,
        wspace_raw.get(),
        wspace_fwd_bytes,
        zero,
        out_desc.descriptor,
        out_raw.get()));
    if (has_bias) {
      auto bs_desc = TensorDescriptor(bias);
      DevicePtr bs_raw(bias.array());

      CUDNN_CHECK_ERR(cudnnAddTensor(
          handle,
          one,
          bs_desc.descriptor,
          bs_raw.get(),
          one,
          out_desc.descriptor,
          out_raw.get()));
    }
  }
  auto gradFunc = [sx, sy, px, py, dx, dy, has_bias, groups](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto& in = inputs[0];
    auto& wt = inputs[1];

    auto i_desc = TensorDescriptor(in);
    auto o_desc = TensorDescriptor(grad_output);

    auto hndl = getCudnnHandle();

    const void* oneg = kOne(in.type());
    const void* zerog = kZero(in.type());
    {
      DevicePtr grad_result_raw(grad_output.array());

      if (has_bias && inputs[2].isCalcGrad()) {
        auto& bs = inputs[2];
        auto grad_bias = Variable(af::array(bs.dims(), bs.type()), false);
        {
          DevicePtr gradbiasraw(grad_bias.array());
          auto b_desc = TensorDescriptor(bs);
          CUDNN_CHECK_ERR(cudnnConvolutionBackwardBias(
              hndl,
              oneg,
              o_desc.descriptor,
              grad_result_raw.get(),
              zerog,
              b_desc.descriptor,
              gradbiasraw.get()));
        }
        bs.addGrad(grad_bias);
      }
    }
    auto w_desc = FilterDescriptor(wt);
    auto c_desc = ConvDescriptor(in.type(), px, py, sx, sy, dx, dy, groups);

    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataAlgorithm(
        hndl,
        w_desc.descriptor,
        o_desc.descriptor,
        c_desc.descriptor,
        i_desc.descriptor,
        bwd_data_pref,
        bwd_data_mem_limit,
        &bwd_data_algo));

    size_t wspace_bwd_data_bytes = 0;
    if (in.isCalcGrad()) {
      CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataWorkspaceSize(
          hndl,
          w_desc.descriptor,
          o_desc.descriptor,
          c_desc.descriptor,
          i_desc.descriptor,
          bwd_data_algo,
          &wspace_bwd_data_bytes));
    }
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterAlgorithm(
        hndl,
        i_desc.descriptor,
        o_desc.descriptor,
        c_desc.descriptor,
        w_desc.descriptor,
        bwd_filter_pref,
        bwd_filter_mem_limit,
        &bwd_filter_algo));

    size_t wspace_bwd_filter_bytes = 0;
    if (wt.isCalcGrad()) {
      CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          hndl,
          i_desc.descriptor,
          o_desc.descriptor,
          c_desc.descriptor,
          w_desc.descriptor,
          bwd_filter_algo,
          &wspace_bwd_filter_bytes));
    }
    auto wspace_bwd_bytes =
        std::max(wspace_bwd_filter_bytes, wspace_bwd_data_bytes);

    auto ws = af::array(wspace_bwd_bytes, af::dtype::b8);

    DevicePtr i_raw(in.array());
    DevicePtr w_raw(wt.array());
    DevicePtr ws_raw(ws);
    if (in.isCalcGrad()) {
      auto grad_input = Variable(af::array(in.dims(), in.type()), false);
      {
        DevicePtr grad_input_raw(grad_input.array());
        DevicePtr grad_result_raw(grad_output.array());
        CUDNN_CHECK_ERR(cudnnConvolutionBackwardData(
            hndl,
            oneg,
            w_desc.descriptor,
            w_raw.get(),
            o_desc.descriptor,
            grad_result_raw.get(),
            c_desc.descriptor,
            bwd_data_algo,
            ws_raw.get(),
            wspace_bwd_bytes,
            zerog,
            i_desc.descriptor,
            grad_input_raw.get()));
      }
      in.addGrad(grad_input);
    }
    if (wt.isCalcGrad()) {
      auto grad_weight = Variable(af::array(wt.dims(), wt.type()), false);
      {
        DevicePtr grad_weight_raw(grad_weight.array());
        DevicePtr grad_result_raw(grad_output.array());
        CUDNN_CHECK_ERR(cudnnConvolutionBackwardFilter(
            hndl,
            oneg,
            i_desc.descriptor,
            i_raw.get(),
            o_desc.descriptor,
            grad_result_raw.get(),
            c_desc.descriptor,
            bwd_filter_algo,
            ws_raw.get(),
            wspace_bwd_bytes,
            zerog,
            w_desc.descriptor,
            grad_weight_raw.get()));
      }
      wt.addGrad(grad_weight);
    }
  };

  if (has_bias) {
    return Variable(output, {input, weights, bias}, gradFunc);
  }
  return Variable(output, {input, weights}, gradFunc);
}

} // namespace fl
