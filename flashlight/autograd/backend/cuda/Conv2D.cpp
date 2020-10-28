/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <stdexcept>

#include <cudnn.h>

#include "flashlight/flashlight/autograd/Functions.h"
#include "flashlight/flashlight/autograd/Variable.h"
#include "flashlight/flashlight/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/flashlight/common/DevicePtr.h"

namespace {
const fl::cpp::enum_unordered_set<cudnnConvolutionFwdAlgo_t> fwdPreferredAlgos =
    {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
     CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

const fl::cpp::enum_unordered_set<cudnnConvolutionBwdDataAlgo_t>
    bwdDataPreferredAlgos = {CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};

const fl::cpp::enum_unordered_set<cudnnConvolutionBwdFilterAlgo_t>
    bwdFilterPreferredAlgos = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED};

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
template <typename T, typename ALGO_TYPE>
T getBestAlgorithm(
    const std::vector<T>& algoPerfs,
    const fl::cpp::enum_unordered_set<ALGO_TYPE>& preferredAlgos,
    const bool algoMustSupportF16) {
  T reserved;
  bool algoFound = false;
  for (const auto& algoPerf : algoPerfs) {
    if (algoPerf.status == CUDNN_STATUS_SUCCESS &&
        algoPerf.memory < kWorkspaceSizeLimitBytes) {
      if (!algoMustSupportF16 ||
          (preferredAlgos.find(algoPerf.algo) != preferredAlgos.end())) {
        return algoPerf;
      } else if (!algoFound) {
        reserved = algoPerf;
        algoFound = true;
      }
    }
  }
  if (algoFound) {
    return reserved;
  } else {
    throw std::runtime_error("Error while finding cuDNN Conv Algorithm.");
  }
}

cudnnConvolutionFwdAlgoPerf_t getFwdAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const cudnnFilterDescriptor_t& wDesc,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& yDesc,
    const af::dtype arithmeticPrecision) {
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

  return getBestAlgorithm(
      fwdAlgoPerfs, fwdPreferredAlgos, arithmeticPrecision == f16);
}

cudnnConvolutionBwdDataAlgoPerf_t getBwdDataAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const cudnnFilterDescriptor_t& wDesc,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& yDesc,
    bool isStrided,
    const af::dtype arithmeticPrecision) {
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

  auto bestAlgo = getBestAlgorithm(
      bwdDataAlgoPerfs, bwdDataPreferredAlgos, arithmeticPrecision == f16);

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_DATA_ALGO_1
  // Seems to produce erroneous results on Tesla P100 GPUs.
  // 2: blacklist FFT algorithms for strided dgrad -
  // https://github.com/pytorch/pytorch/issues/16610
  bool isAlgoBlacklisted = false;
#ifndef FL_CUDNN_ALLOW_ALGO_1
  if (arithmeticPrecision != f16 &&
      bestAlgo.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1) {
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
    const cudnnTensorDescriptor_t& yDesc,
    const af::dtype arithmeticPrecision) {
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
  auto bestAlgo = getBestAlgorithm(
      bwdFilterAlgoPerfs, bwdFilterPreferredAlgos, arithmeticPrecision == f16);

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_FILTER_ALGO_1
  // We do the blacklist here just to be safe as we did in BWD_DATA_ALGO_1
  bool isAlgoBlacklisted = false;
#ifndef FL_CUDNN_ALLOW_ALGO_1
  if (arithmeticPrecision != f16 &&
      bestAlgo.algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1) {
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

  af::array weightsArray;
  if (weights.type() == input.type()) {
    weightsArray = weights.array();
  } else {
    weightsArray = weights.array().as(input.type());
  }

  af::array biasArray;
  if (bias.type() == input.type()) {
    biasArray = bias.array();
  } else {
    biasArray = bias.array().as(input.type());
  }

  auto inDesc = TensorDescriptor(input);
  auto wtDesc = FilterDescriptor(weightsArray);
  auto convDesc = ConvDescriptor(input.type(), px, py, sx, sy, dx, dy, groups);
  if (input.type() == f16) {
    CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
        convDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetConvolutionMathType(convDesc.descriptor, CUDNN_DEFAULT_MATH));
  }

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
      outDesc.descriptor,
      input.type());

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
    DevicePtr wtPtr(weightsArray);
    DevicePtr outPtr(output);
    DevicePtr wspacePtr(wspace);

    auto scalarsType = input.type() == f16 ? f32 : input.type();
    const void* one = kOne(scalarsType);
    const void* zero = kZero(scalarsType);

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
      auto bsDesc = TensorDescriptor(biasArray);
      DevicePtr bsPtr(biasArray);

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

    af::array wtArray;
    if (wt.type() == in.type()) {
      wtArray = wt.array();
    } else {
      wtArray = wt.array().as(in.type());
    }

    af::array gradOutputArray;
    if (gradOutput.type() == in.type()) {
      gradOutputArray = gradOutput.array();
    } else {
      gradOutputArray = gradOutput.array().as(in.type());
    }

    auto iDesc = TensorDescriptor(in);
    auto oDesc = TensorDescriptor(gradOutputArray);

    auto hndl = getCudnnHandle();

    auto scalarsType = in.type() == f16 ? f32 : in.type();
    const void* oneg = kOne(scalarsType);
    const void* zerog = kZero(scalarsType);
    {
      DevicePtr gradResultPtr(gradOutputArray);

      if (hasBias && inputs[2].isCalcGrad()) {
        auto& bs = inputs[2];
        af::array bsArray;
        if (bs.type() == in.type()) {
          bsArray = bs.array();
        } else {
          bsArray = bs.array().as(in.type());
        }
        auto gradBias =
            Variable(af::array(bsArray.dims(), bsArray.type()), false);
        {
          DevicePtr gradBiasPtr(gradBias.array());
          auto bDesc = TensorDescriptor(bsArray);
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
    auto wDesc = FilterDescriptor(wtArray);
    auto cDesc = ConvDescriptor(in.type(), px, py, sx, sy, dx, dy, groups);
    if (in.type() == f16) {
      CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
          cDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else {
      CUDNN_CHECK_ERR(
          cudnnSetConvolutionMathType(cDesc.descriptor, CUDNN_DEFAULT_MATH));
    }

    DevicePtr iPtr(in.array());
    DevicePtr wPtr(wtArray);

    if (in.isCalcGrad()) {
      bool isStrided = (dx * dy) > 1;
      auto bwdDataAlgoBestPerf = getBwdDataAlgo(
          iDesc.descriptor,
          wDesc.descriptor,
          cDesc.descriptor,
          oDesc.descriptor,
          isStrided,
          in.type());
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
        DevicePtr gradResultPtr(gradOutputArray);
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
          oDesc.descriptor,
          in.type());
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
      auto gradWeight =
          Variable(af::array(wtArray.dims(), wtArray.type()), false);
      {
        DevicePtr gradWeightPtr(gradWeight.array());
        DevicePtr gradResultPtr(gradOutputArray);
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
