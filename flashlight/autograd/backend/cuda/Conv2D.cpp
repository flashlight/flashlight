/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <stdexcept>

#include <cudnn.h>

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/common/DevicePtr.h"

namespace {
constexpr size_t kWorkspaceSizeLimitBytes = 512 * 1024 * 1024; // 512 MB
constexpr cudnnConvolutionFwdAlgo_t kFwdDefaultAlgo =
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
constexpr cudnnConvolutionBwdDataAlgo_t kBwdDataDefaultAlgo =
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
constexpr cudnnConvolutionBwdFilterAlgo_t kBwdFilterDefaultAlgo =
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

// Get the algorithm which gives best performance.
// Since cuDNN doesn't support memory limits, we manually choose an algorithm
// which requires less than a specific workspace size.
template <typename T>
T getBestAlgorithm(const std::vector<T>& algoPerfs) {
  for (const auto& algoPerf : algoPerfs) {
    if (algoPerf.status == CUDNN_STATUS_SUCCESS &&
        algoPerf.memory < kWorkspaceSizeLimitBytes) {
      return algoPerf;
    }
  }
  throw std::runtime_error("Error while finding cuDNN Conv Algorithm.");
}

cudnnConvolutionFwdAlgoPerf_t getFwdAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const cudnnFilterDescriptor_t& wDesc,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& yDesc) {
  int numFwdAlgoRequested, numFwdAlgoReturned;

  CUDNN_CHECK_ERR(cudnnGetConvolutionForwardAlgorithmMaxCount(
      fl::getCudnnHandle(), &numFwdAlgoRequested));

  std::vector<cudnnConvolutionFwdAlgoPerf_t> fwdAlgoPerfs(numFwdAlgoRequested);
  CUDNN_CHECK_ERR(cudnnGetConvolutionForwardAlgorithm_v7(
      fl::getCudnnHandle(),
      xDesc,
      wDesc,
      convDesc,
      yDesc,
      numFwdAlgoRequested,
      &numFwdAlgoReturned,
      fwdAlgoPerfs.data()));

  return getBestAlgorithm(fwdAlgoPerfs);
}

cudnnConvolutionBwdDataAlgoPerf_t getBwdDataAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const cudnnFilterDescriptor_t& wDesc,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& yDesc,
    bool isStrided) {
  int numBwdDataAlgoRequested, numBwdDataAlgoReturned;

  CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
      fl::getCudnnHandle(), &numBwdDataAlgoRequested));

  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwdDataAlgoPerfs(
      numBwdDataAlgoRequested);
  CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataAlgorithm_v7(
      fl::getCudnnHandle(),
      wDesc,
      yDesc,
      convDesc,
      xDesc,
      numBwdDataAlgoRequested,
      &numBwdDataAlgoReturned,
      bwdDataAlgoPerfs.data()));

  auto bestAlgo = getBestAlgorithm(bwdDataAlgoPerfs);

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_DATA_ALGO_1
  // Seems to produce erroneous results on Tesla P100 GPUs.
  // 2: blacklist FFT algorithms for strided dgrad -
  // https://github.com/pytorch/pytorch/issues/16610
  bool isAlgoBlacklisted = false;
#ifndef FL_CUDNN_ALLOW_ALGO_1
  if (bestAlgo.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1) {
    isAlgoBlacklisted = true;
  }
#endif
#if CUDNN_VERSION < 7500
  if (isStrided &&
      (bestAlgo.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
       bestAlgo.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
    isAlgoBlacklisted = true;
  }
#endif
  if (isAlgoBlacklisted) {
    for (const auto& algoPerf : bwdDataAlgoPerfs) {
      if (algoPerf.algo == kBwdDataDefaultAlgo) {
        return algoPerf;
      }
    }
  }
  return bestAlgo;
}

cudnnConvolutionBwdFilterAlgoPerf_t getBwdFilterAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const cudnnFilterDescriptor_t& wDesc,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& yDesc) {
  int numBwdFilterAlgoRequested, numBwdFilterAlgoReturned;

  CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
      fl::getCudnnHandle(), &numBwdFilterAlgoRequested));

  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwdFilterAlgoPerfs(
      numBwdFilterAlgoRequested);
  CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
      fl::getCudnnHandle(),
      xDesc,
      yDesc,
      convDesc,
      wDesc,
      numBwdFilterAlgoRequested,
      &numBwdFilterAlgoReturned,
      bwdFilterAlgoPerfs.data()));
  auto bestAlgo = getBestAlgorithm(bwdFilterAlgoPerfs);

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_FILTER_ALGO_1
  // We do the blacklist here just to be safe as we did in BWD_DATA_ALGO_1
  bool isAlgoBlacklisted = false;
#ifndef FL_CUDNN_ALLOW_ALGO_1
  if (bestAlgo.algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) {
    isAlgoBlacklisted = true;
  }
#endif
  if (isAlgoBlacklisted) {
    for (const auto& algoPerf : bwdFilterAlgoPerfs) {
      if (algoPerf.algo == kBwdFilterDefaultAlgo) {
        return algoPerf;
      }
    }
  }
  return bestAlgo;
}
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
  auto hasBias = bias.elements() > 0;

  auto inDesc = TensorDescriptor(input);
  auto wtDesc = FilterDescriptor(weights);
  auto convDesc = ConvDescriptor(input.type(), px, py, sx, sy, dx, dy, groups);

  std::array<int, 4> odims;
  CUDNN_CHECK_ERR(cudnnGetConvolutionNdForwardOutputDim(
      convDesc.descriptor,
      inDesc.descriptor,
      wtDesc.descriptor,
      4,
      odims.data()));
  auto output = af::array(odims[3], odims[2], odims[1], odims[0], input.type());
  auto outDesc = TensorDescriptor(output);

  auto handle = getCudnnHandle();

  auto fwdAlgoBestPerf = getFwdAlgo(
      inDesc.descriptor,
      wtDesc.descriptor,
      convDesc.descriptor,
      outDesc.descriptor);

  af::array wspace;

  try {
    wspace = af::array(fwdAlgoBestPerf.memory, af::dtype::b8);
  } catch (const std::exception& e) {
    fwdAlgoBestPerf.algo = kFwdDefaultAlgo;
    CUDNN_CHECK_ERR(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        inDesc.descriptor,
        wtDesc.descriptor,
        convDesc.descriptor,
        outDesc.descriptor,
        fwdAlgoBestPerf.algo,
        &fwdAlgoBestPerf.memory));
    wspace = af::array(fwdAlgoBestPerf.memory, af::dtype::b8);
  }
  {
    DevicePtr inPtr(input.array());
    DevicePtr wtPtr(weights.array());
    DevicePtr outPtr(output);
    DevicePtr wspacePtr(wspace);

    const void* one = kOne(input.type());
    const void* zero = kZero(input.type());

    CUDNN_CHECK_ERR(cudnnConvolutionForward(
        handle,
        one,
        inDesc.descriptor,
        inPtr.get(),
        wtDesc.descriptor,
        wtPtr.get(),
        convDesc.descriptor,
        fwdAlgoBestPerf.algo,
        wspacePtr.get(),
        fwdAlgoBestPerf.memory,
        zero,
        outDesc.descriptor,
        outPtr.get()));

    if (hasBias) {
      auto bsDesc = TensorDescriptor(bias);
      DevicePtr bsPtr(bias.array());

      CUDNN_CHECK_ERR(cudnnAddTensor(
          handle,
          one,
          bsDesc.descriptor,
          bsPtr.get(),
          one,
          outDesc.descriptor,
          outPtr.get()));
    }
  }
  auto gradFunc = [sx, sy, px, py, dx, dy, hasBias, groups](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto& in = inputs[0];
    auto& wt = inputs[1];

    auto iDesc = TensorDescriptor(in);
    auto oDesc = TensorDescriptor(gradOutput);

    auto hndl = getCudnnHandle();

    const void* oneg = kOne(in.type());
    const void* zerog = kZero(in.type());
    {
      DevicePtr gradResultPtr(gradOutput.array());

      if (hasBias && inputs[2].isCalcGrad()) {
        auto& bs = inputs[2];
        auto gradBias = Variable(af::array(bs.dims(), bs.type()), false);
        {
          DevicePtr gradBiasPtr(gradBias.array());
          auto bDesc = TensorDescriptor(bs);
          CUDNN_CHECK_ERR(cudnnConvolutionBackwardBias(
              hndl,
              oneg,
              oDesc.descriptor,
              gradResultPtr.get(),
              zerog,
              bDesc.descriptor,
              gradBiasPtr.get()));
        }
        bs.addGrad(gradBias);
      }
    }
    auto wDesc = FilterDescriptor(wt);
    auto cDesc = ConvDescriptor(in.type(), px, py, sx, sy, dx, dy, groups);

    DevicePtr iPtr(in.array());
    DevicePtr wPtr(wt.array());

    if (in.isCalcGrad()) {
      bool isStrided = (dx * dy) > 1;
      auto bwdDataAlgoBestPerf = getBwdDataAlgo(
          iDesc.descriptor,
          wDesc.descriptor,
          cDesc.descriptor,
          oDesc.descriptor,
          isStrided);
      af::array ws;

      try {
        ws = af::array(bwdDataAlgoBestPerf.memory, af::dtype::b8);
      } catch (const std::exception& e) {
        bwdDataAlgoBestPerf.algo = kBwdDataDefaultAlgo;
        CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataWorkspaceSize(
            hndl,
            wDesc.descriptor,
            oDesc.descriptor,
            cDesc.descriptor,
            iDesc.descriptor,
            bwdDataAlgoBestPerf.algo,
            &bwdDataAlgoBestPerf.memory));
        ws = af::array(bwdDataAlgoBestPerf.memory, af::dtype::b8);
      }
      auto gradInput = Variable(af::array(in.dims(), in.type()), false);
      {
        DevicePtr gradInputPtr(gradInput.array());
        DevicePtr gradResultPtr(gradOutput.array());
        DevicePtr wsPtr(ws);
        CUDNN_CHECK_ERR(cudnnConvolutionBackwardData(
            hndl,
            oneg,
            wDesc.descriptor,
            wPtr.get(),
            oDesc.descriptor,
            gradResultPtr.get(),
            cDesc.descriptor,
            bwdDataAlgoBestPerf.algo,
            wsPtr.get(),
            bwdDataAlgoBestPerf.memory,
            zerog,
            iDesc.descriptor,
            gradInputPtr.get()));
      }
      in.addGrad(gradInput);
    }
    if (wt.isCalcGrad()) {
      auto bwdFilterAlgoBestPerf = getBwdFilterAlgo(
          iDesc.descriptor,
          wDesc.descriptor,
          cDesc.descriptor,
          oDesc.descriptor);
      af::array ws;

      try {
        ws = af::array(bwdFilterAlgoBestPerf.memory, af::dtype::b8);
      } catch (const std::exception& e) {
        bwdFilterAlgoBestPerf.algo = kBwdFilterDefaultAlgo;
        CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            hndl,
            iDesc.descriptor,
            oDesc.descriptor,
            cDesc.descriptor,
            wDesc.descriptor,
            bwdFilterAlgoBestPerf.algo,
            &bwdFilterAlgoBestPerf.memory));
        ws = af::array(bwdFilterAlgoBestPerf.memory, af::dtype::b8);
      }
      auto gradWeight = Variable(af::array(wt.dims(), wt.type()), false);
      {
        DevicePtr gradWeightPtr(gradWeight.array());
        DevicePtr gradResultPtr(gradOutput.array());
        DevicePtr wsPtr(ws);
        CUDNN_CHECK_ERR(cudnnConvolutionBackwardFilter(
            hndl,
            oneg,
            iDesc.descriptor,
            iPtr.get(),
            oDesc.descriptor,
            gradResultPtr.get(),
            cDesc.descriptor,
            bwdFilterAlgoBestPerf.algo,
            wsPtr.get(),
            bwdFilterAlgoBestPerf.memory,
            zerog,
            wDesc.descriptor,
            gradWeightPtr.get()));
      }
      wt.addGrad(gradWeight);
    }
  };

  if (hasBias) {
    return Variable(output, {input, weights, bias}, gradFunc);
  }
  return Variable(output, {input, weights}, gradFunc);
}

} // namespace fl
