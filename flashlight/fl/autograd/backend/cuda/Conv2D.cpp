/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <stdexcept>

#include <cudnn.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/backend/cuda/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/common/DynamicBenchmark.h"

namespace {
const fl::cpp::fl_unordered_set<cudnnConvolutionFwdAlgo_t> kFwdPreferredAlgos =
    {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
     CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

const fl::cpp::fl_unordered_set<cudnnConvolutionBwdDataAlgo_t>
    kBwdDataPreferredAlgos = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};

const fl::cpp::fl_unordered_set<cudnnConvolutionBwdFilterAlgo_t>
    kBwdFilterPreferredAlgos = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED};

constexpr size_t kWorkspaceSizeLimitBytes = 512 * 1024 * 1024; // 512 MB
constexpr cudnnConvolutionFwdAlgo_t kFwdDefaultAlgo =
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
constexpr cudnnConvolutionBwdDataAlgo_t kBwdDataDefaultAlgo =
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
constexpr cudnnConvolutionBwdFilterAlgo_t kBwdFilterDefaultAlgo =
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

cudnnConvolutionFwdAlgoPerf_t getFwdAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const fl::DevicePtr& xPtr,
    const cudnnFilterDescriptor_t& wDesc,
    const fl::DevicePtr& wtPtr,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& yDesc,
    fl::DevicePtr& outPtr) {
  static const cudnnConvolutionFwdAlgo_t algos[] = {
      CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_FFT,
      CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
      CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
  };
  static_assert(
      sizeof(algos) / sizeof(algos[0]) == CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
      "The number of forward algorithms in Flashlight "
      "must be equal to that of cuDNN.");

  size_t maxWorkspaceSize = 0;
  for (const auto& algo : algos) {
    size_t sz;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
        fl::getCudnnHandle(), xDesc, wDesc, convDesc, yDesc, algo, &sz);

    if (status == CUDNN_STATUS_SUCCESS && sz > maxWorkspaceSize) {
      maxWorkspaceSize = sz;
      if (maxWorkspaceSize >= kWorkspaceSizeLimitBytes) {
        maxWorkspaceSize = kWorkspaceSizeLimitBytes;
        break;
      }
    }
  }

  auto workspace = af::array(maxWorkspaceSize, af::dtype::b8);
  fl::DevicePtr workspacePtr(workspace);

  int numFwdAlgoRequested, numFwdAlgoReturned;

  CUDNN_CHECK_ERR(cudnnGetConvolutionForwardAlgorithmMaxCount(
      fl::getCudnnHandle(), &numFwdAlgoRequested));

  std::vector<cudnnConvolutionFwdAlgoPerf_t> fwdAlgoPerfs(numFwdAlgoRequested);

  CUDNN_CHECK_ERR(cudnnFindConvolutionForwardAlgorithmEx(
      fl::getCudnnHandle(),
      xDesc,
      xPtr.get(),
      wDesc,
      wtPtr.get(),
      convDesc,
      yDesc,
      outPtr.get(),
      numFwdAlgoRequested,
      &numFwdAlgoReturned,
      fwdAlgoPerfs.data(),
      workspacePtr.get(),
      maxWorkspaceSize));

  for (const auto& algoPerf : fwdAlgoPerfs) {
    if (algoPerf.status == CUDNN_STATUS_SUCCESS) {
      return algoPerf;
    }
  }
  throw std::runtime_error(
      "Cannot find a compatible CUDNN convolution algorithm.");
}

cudnnConvolutionBwdDataAlgoPerf_t getBwdDataAlgo(
    const cudnnTensorDescriptor_t& dxDesc,
    fl::DevicePtr& dxPtr,
    const cudnnFilterDescriptor_t& wDesc,
    const fl::DevicePtr& wPtr,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& dyDesc,
    const fl::DevicePtr& dyPtr,
    const bool isStrided) {
  static const cudnnConvolutionBwdDataAlgo_t algos[] = {
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
  static_assert(
      sizeof(algos) / sizeof(algos[0]) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
      "The number of backward data algorithms in Flashlight "
      "must be equal to that of cuDNN.");

  size_t maxWorkspaceSize = 0;
  for (const auto& algo : algos) {
    size_t sz;
    auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        fl::getCudnnHandle(), wDesc, dyDesc, convDesc, dxDesc, algo, &sz);

    if (status == CUDNN_STATUS_SUCCESS && sz > maxWorkspaceSize) {
      maxWorkspaceSize = sz;
    }
    if (maxWorkspaceSize >= kWorkspaceSizeLimitBytes) {
      maxWorkspaceSize = kWorkspaceSizeLimitBytes;
    }
  }

  auto workspace = af::array(maxWorkspaceSize, af::dtype::b8);
  fl::DevicePtr workspacePtr(workspace);

  int numBwdDataAlgoRequested, numBwdDataAlgoReturned;

  CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
      fl::getCudnnHandle(), &numBwdDataAlgoRequested));

  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwdDataAlgoPerfs(
      numBwdDataAlgoRequested);
  CUDNN_CHECK_ERR(cudnnFindConvolutionBackwardDataAlgorithmEx(
      fl::getCudnnHandle(),
      wDesc,
      wPtr.get(),
      dyDesc,
      dyPtr.get(),
      convDesc,
      dxDesc,
      dxPtr.get(),
      numBwdDataAlgoRequested,
      &numBwdDataAlgoReturned,
      bwdDataAlgoPerfs.data(),
      workspacePtr.get(),
      maxWorkspaceSize));

  cudnnConvolutionBwdDataAlgoPerf_t bestAlgoPerf;
  for (const auto& algoPerf : bwdDataAlgoPerfs) {
    if (algoPerf.status == CUDNN_STATUS_SUCCESS) {
      bestAlgoPerf = algoPerf;
      break;
    }
  }

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_DATA_ALGO_1
  // Seems to produce erroneous results on Tesla P100 GPUs.
  // 2: blacklist FFT algorithms for strided dgrad -
  // https://github.com/pytorch/pytorch/issues/16610
  bool algoBlacklisted = false;
#ifdef FL_CUDNN_BLOCK_ALGO_1
  if (bestAlgoPerf.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 &&
      (bestAlgoPerf.mathType == CUDNN_DEFAULT_MATH ||
       bestAlgoPerf.mathType == CUDNN_FMA_MATH)) {
    algoBlacklisted = true;
  }
#endif
#if CUDNN_VERSION < 7500
  if (isStrided &&
      (bestAlgo.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
       bestAlgo.algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
    algoBlacklisted = true;
  }
#endif
  if (algoBlacklisted) {
    std::cerr
        << "WARNING: The optimal algorithm for computing the gradients w.r.t "
           "data is blocked. Falling back on a default algorithm. This can "
           "increase the execution time."
        << std::endl;

    for (const auto& algoPerf : bwdDataAlgoPerfs) {
      if (algoPerf.algo == kBwdDataDefaultAlgo) {
        return algoPerf;
      }
    }
  } else {
    return bestAlgoPerf;
  }
  throw std::runtime_error(
      "Cannot find a compatible CUDNN convolution bwd data algorithm.");
}

cudnnConvolutionBwdFilterAlgoPerf_t getBwdFilterAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const fl::DevicePtr& xPtr,
    const cudnnFilterDescriptor_t& dwDesc,
    fl::DevicePtr& dwPtr,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& dyDesc,
    const fl::DevicePtr& dyPtr) {
  static const cudnnConvolutionBwdFilterAlgo_t algos[] = {
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
      // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD, // NOT Implemented Yet.
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
  };

  // TODO: When cuDNN starts supporting _ALGO_WINOGRAD, remove this -1 and
  // uncomment it from the above list.
  static_assert(
      sizeof(algos) / sizeof(algos[0]) ==
          CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1,
      "The number of backward filter algorithms in Flashlight "
      "must be equal to that of cuDNN.");

  size_t maxWorkspaceSize = 0;
  for (const auto& algo : algos) {
    size_t sz;
    auto status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
        fl::getCudnnHandle(), xDesc, dyDesc, convDesc, dwDesc, algo, &sz);

    if (status == CUDNN_STATUS_SUCCESS && sz > maxWorkspaceSize) {
      maxWorkspaceSize = sz;
    }
    if (maxWorkspaceSize >= kWorkspaceSizeLimitBytes) {
      maxWorkspaceSize = kWorkspaceSizeLimitBytes;
    }
  }

  auto workspace = af::array(maxWorkspaceSize, af::dtype::b8);
  fl::DevicePtr workspacePtr(workspace);

  int numBwdFilterAlgoRequested, numBwdFilterAlgoReturned;

  CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
      fl::getCudnnHandle(), &numBwdFilterAlgoRequested));

  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwdFilterAlgoPerfs(
      numBwdFilterAlgoRequested);

  CUDNN_CHECK_ERR(cudnnFindConvolutionBackwardFilterAlgorithmEx(
      fl::getCudnnHandle(),
      xDesc,
      xPtr.get(),
      dyDesc,
      dyPtr.get(),
      convDesc,
      dwDesc,
      dwPtr.get(),
      numBwdFilterAlgoRequested,
      &numBwdFilterAlgoReturned,
      bwdFilterAlgoPerfs.data(),
      workspacePtr.get(),
      maxWorkspaceSize));

  cudnnConvolutionBwdFilterAlgoPerf_t bestAlgoPerf;
  for (const auto& algoPerf : bwdFilterAlgoPerfs) {
    if (algoPerf.status == CUDNN_STATUS_SUCCESS) {
      bestAlgoPerf = algoPerf;
      break;
    }
  }

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_FILTER_ALGO_1
  // We do the blacklist here just to be safe as we did in BWD_DATA_ALGO_1
  bool algoBlacklisted = false;
#ifdef FL_CUDNN_BLOCK_ALGO_1
  if (bestAlgo.algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 &&
      (bestAlgoPerf.mathType == CUDNN_DEFAULT_MATH ||
       bestAlgoPerf.mathType == CUDNN_FMA_MATH)) {
    algoBlacklisted = true;
  }
#endif
  if (algoBlacklisted) {
    std::cerr
        << "WARNING: The optimal algorithm for computing the gradients w.r.t "
           "filter is blocked. Falling back on a default algorithm. This can "
           "increase the execution time."
        << std::endl;

    for (const auto& algoPerf : bwdFilterAlgoPerfs) {
      if (algoPerf.algo == kBwdFilterDefaultAlgo) {
        return algoPerf;
      }
    }
  } else {
    return bestAlgoPerf;
  }
  throw std::runtime_error(
      "Cannot find a compatible CUDNN convolution bwd filer algorithm.");
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
    int groups,
    detail::ConvAlgoConfigsPtr convAlgoConfigs) {
  auto dummyBias =
      Variable(af::array(af::dim4(0, 1, 1, 1), input.type()), false);
  return conv2d(
      input,
      weights,
      dummyBias,
      sx,
      sy,
      px,
      py,
      dx,
      dy,
      groups,
      convAlgoConfigs);
}

Variable conv2d(
    const Variable& in,
    const Variable& wt,
    const Variable& bs,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    detail::ConvAlgoConfigsPtr convAlgoConfigs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(in, wt, bs);

  auto hasBias = bs.elements() > 0;

  bool fp16Math = in.type() == f16;
  // Only performing upcast in case it is required. If a downcast is required,
  // that can be handled on the fly within cuDNN.
  if (!fp16Math) {
    if (in.type() == af::dtype::f16) {
      in.array() = in.array().as(af::dtype::f32);
    }

    if (wt.type() == af::dtype::f16) {
      wt.array() = wt.array().as(af::dtype::f32);
    }

    if (bs.type() == af::dtype::f16) {
      bs.array() = bs.array().as(af::dtype::f32);
    }
  }

  auto inDesc = TensorDescriptor(in);
  auto wtDesc = FilterDescriptor(wt.array());
  auto convDesc = ConvDescriptor(in.type(), px, py, sx, sy, dx, dy, groups);

  if (fp16Math) {
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
  auto output = af::array(odims[3], odims[2], odims[1], odims[0], in.type());
  auto outDesc = TensorDescriptor(output);

  auto handle = getCudnnHandle();

  af::array wspace;
  DevicePtr inPtr(in.array());
  DevicePtr wtPtr(wt.array());
  DevicePtr outPtr(output);

  int inputX = in.dims()[0];
  int batchSize = in.dims()[3];
  if (!convAlgoConfigs->fwd.find(inputX, batchSize, fp16Math)) {
    auto tmpWorkspace = af::array(kWorkspaceSizeLimitBytes, af::dtype::b8);
    DevicePtr tmpWsPtr(tmpWorkspace);

    cudnnConvolutionFwdAlgoPerf_t fwdAlgoBestPerf = getFwdAlgo(
        inDesc.descriptor,
        inPtr,
        wtDesc.descriptor,
        wtPtr,
        convDesc.descriptor,
        outDesc.descriptor,
        outPtr);
    auto fwdAlgoConfigs = fl::detail::AlgoConfigs(
        fwdAlgoBestPerf.algo, fwdAlgoBestPerf.memory, fwdAlgoBestPerf.mathType);
    convAlgoConfigs->fwd.set(inputX, batchSize, fp16Math, fwdAlgoConfigs);
  }

  auto fwdAlgoConfigs = convAlgoConfigs->fwd.get(inputX, batchSize, fp16Math);
  CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
      convDesc.descriptor, (cudnnMathType_t)fwdAlgoConfigs.mathType));

  try {
    wspace = af::array(fwdAlgoConfigs.memory, af::dtype::b8);
  } catch (const std::exception& e) {
    std::cerr
        << "WARNING: Could not allocate GPU memory for the most performant "
           "convolution algorithm. Falling back on a default algorithm. This "
           "increases the execution time."
        << std::endl;
    fwdAlgoConfigs.algo = kFwdDefaultAlgo;
    CUDNN_CHECK_ERR(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        inDesc.descriptor,
        wtDesc.descriptor,
        convDesc.descriptor,
        outDesc.descriptor,
        (cudnnConvolutionFwdAlgo_t)fwdAlgoConfigs.algo,
        &fwdAlgoConfigs.memory));
    wspace = af::array(fwdAlgoConfigs.memory, af::dtype::b8);
    convAlgoConfigs->fwd.set(inputX, batchSize, fp16Math, fwdAlgoConfigs);
  }

  DevicePtr wspacePtr(wspace);

  auto scalarsType = in.type() == f16 ? f32 : in.type();
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
      (cudnnConvolutionFwdAlgo_t)fwdAlgoConfigs.algo,
      wspacePtr.get(),
      fwdAlgoConfigs.memory,
      zero,
      outDesc.descriptor,
      outPtr.get()));

  if (hasBias) {
    auto bsDesc = TensorDescriptor(bs.array());
    DevicePtr bsPtr(bs.array());

    CUDNN_CHECK_ERR(cudnnAddTensor(
        handle,
        one,
        bsDesc.descriptor,
        bsPtr.get(),
        one,
        outDesc.descriptor,
        outPtr.get()));
  }
  auto gradFunc = [sx,
                   sy,
                   px,
                   py,
                   dx,
                   dy,
                   hasBias,
                   groups,
                   convAlgoConfigs,
                   fp16Math](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto& in = inputs[0];
    auto& wt = inputs[1];
    int inputX = in.dims()[0];
    int batchSize = in.dims()[3];

    auto iDesc = TensorDescriptor(in);
    auto wDesc = FilterDescriptor(wt.array());
    auto cDesc = ConvDescriptor(in.type(), px, py, sx, sy, dx, dy, groups);
    auto oDesc = TensorDescriptor(gradOutput.array());

    auto hndl = getCudnnHandle();

    auto scalarsType = in.type() == f16 ? f32 : in.type();
    const void* oneg = kOne(scalarsType);
    const void* zerog = kZero(scalarsType);

    // Bias gradients
    if (hasBias && inputs.size() > 2 && inputs[2].isCalcGrad()) {
      auto& bias = inputs[2];
      auto convolutionBackwardBias = [&bias, &hndl, oneg, zerog](
                                         const af::array& bsArray,
                                         const af::array& gradOutput,
                                         const TensorDescriptor& oDesc) {
        DevicePtr gradResultPtr(gradOutput);

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
        bias.addGrad(gradBias);
      };
      convolutionBackwardBias(bias.array(), gradOutput.array(), oDesc);
    }

    // Gradients with respect to the input
    auto convolutionBackwardData = [&hndl,
                                    &in,
                                    oneg,
                                    zerog,
                                    dx,
                                    dy,
                                    inputX,
                                    batchSize,
                                    fp16Math,
                                    convAlgoConfigs](
                                       const af::array& inArray,
                                       const af::array& wtArray,
                                       const af::array& gradOutputArray,
                                       TensorDescriptor& iDesc,
                                       FilterDescriptor& wDesc,
                                       ConvDescriptor& cDesc,
                                       TensorDescriptor& oDesc) {
      if (fp16Math) {
        CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
            cDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
      } else {
        CUDNN_CHECK_ERR(
            cudnnSetConvolutionMathType(cDesc.descriptor, CUDNN_DEFAULT_MATH));
      }
      DevicePtr inPtr(inArray);
      auto gradInput =
          Variable(af::array(inArray.dims(), inArray.type()), false);
      DevicePtr gradInputPtr(gradInput.array());

      DevicePtr wPtr(wtArray);
      DevicePtr gradResultPtr(gradOutputArray);
      if (in.isCalcGrad()) {
        if (!convAlgoConfigs->bwdData.find(inputX, batchSize, fp16Math)) {
          bool isStrided = (dx * dy) > 1;
          auto bwdDataAlgoBestPerf = getBwdDataAlgo(
              iDesc.descriptor,
              gradInputPtr,
              wDesc.descriptor,
              wPtr,
              cDesc.descriptor,
              oDesc.descriptor,
              gradResultPtr,
              isStrided);
          auto bwdDataAlgoConfigs = fl::detail::AlgoConfigs(
              bwdDataAlgoBestPerf.algo,
              bwdDataAlgoBestPerf.memory,
              bwdDataAlgoBestPerf.mathType);
          convAlgoConfigs->bwdData.set(
              inputX, batchSize, fp16Math, bwdDataAlgoConfigs);
        }
        auto bwdDataAlgoConfigs =
            convAlgoConfigs->bwdData.get(inputX, batchSize, fp16Math);
        CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
            cDesc.descriptor, (cudnnMathType_t)bwdDataAlgoConfigs.mathType));

        af::array ws;
        try {
          ws = af::array(bwdDataAlgoConfigs.memory, af::dtype::b8);
        } catch (const std::exception& e) {
          std::cerr
              << "WARNING: Could not allocate GPU memory for the most "
                 "performant bwd data convolution algorithm. Falling back "
                 "on a default algorithm."
              << std::endl;

          bwdDataAlgoConfigs.algo = kBwdDataDefaultAlgo;
          CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataWorkspaceSize(
              hndl,
              wDesc.descriptor,
              oDesc.descriptor,
              cDesc.descriptor,
              iDesc.descriptor,
              (cudnnConvolutionBwdDataAlgo_t)bwdDataAlgoConfigs.algo,
              &bwdDataAlgoConfigs.memory));
          ws = af::array(bwdDataAlgoConfigs.memory, af::dtype::b8);
          convAlgoConfigs->bwdData.set(
              inputX, batchSize, fp16Math, bwdDataAlgoConfigs);
        }

        {
          DevicePtr wsPtr(ws);
          CUDNN_CHECK_ERR(cudnnConvolutionBackwardData(
              hndl,
              oneg,
              wDesc.descriptor,
              wPtr.get(),
              oDesc.descriptor,
              gradResultPtr.get(),
              cDesc.descriptor,
              (cudnnConvolutionBwdDataAlgo_t)bwdDataAlgoConfigs.algo,
              wsPtr.get(),
              bwdDataAlgoConfigs.memory,
              zerog,
              iDesc.descriptor,
              gradInputPtr.get()));
        }
        in.addGrad(gradInput);
      }
    };

    convolutionBackwardData(
        in.array(), wt.array(), gradOutput.array(), iDesc, wDesc, cDesc, oDesc);

    // Gradients with respect to the filter
    auto convolutionBackwardFilter = [&hndl,
                                      &wt,
                                      oneg,
                                      zerog,
                                      fp16Math,
                                      inputX,
                                      batchSize,
                                      convAlgoConfigs](
                                         const af::array& inArray,
                                         const af::array& wtArray,
                                         const af::array& gradOutputArray,
                                         TensorDescriptor& iDesc,
                                         FilterDescriptor& wDesc,
                                         ConvDescriptor& cDesc,
                                         TensorDescriptor& oDesc) {
      if (fp16Math) {
        CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
            cDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
      } else {
        CUDNN_CHECK_ERR(
            cudnnSetConvolutionMathType(cDesc.descriptor, CUDNN_DEFAULT_MATH));
      }
      auto gradWeight =
          Variable(af::array(wtArray.dims(), wtArray.type()), false);
      DevicePtr gradWeightPtr(gradWeight.array());
      DevicePtr iPtr(inArray);
      DevicePtr gradResultPtr(gradOutputArray);
      if (wt.isCalcGrad()) {
        if (!convAlgoConfigs->bwdFilter.find(inputX, batchSize, fp16Math)) {
          auto bwdFilterAlgoBestPerf = getBwdFilterAlgo(
              iDesc.descriptor,
              iPtr,
              wDesc.descriptor,
              gradWeightPtr,
              cDesc.descriptor,
              oDesc.descriptor,
              gradResultPtr);
          auto bwdFilterAlgoConfigs = fl::detail::AlgoConfigs(
              bwdFilterAlgoBestPerf.algo,
              bwdFilterAlgoBestPerf.memory,
              bwdFilterAlgoBestPerf.mathType);
          convAlgoConfigs->bwdFilter.set(
              inputX, batchSize, fp16Math, bwdFilterAlgoConfigs);
        }
        auto bwdFilterAlgoConfigs =
            convAlgoConfigs->bwdFilter.get(inputX, batchSize, fp16Math);
        CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
            cDesc.descriptor, (cudnnMathType_t)bwdFilterAlgoConfigs.mathType));

        af::array ws;
        try {
          ws = af::array(bwdFilterAlgoConfigs.memory, af::dtype::b8);
        } catch (const std::exception& e) {
          std::cerr
              << "WARNING: Could not allocate GPU memory for the most "
                 "performant bwd filter convolution algorithm. Falling back "
                 "on a default algorithm."
              << std::endl;

          bwdFilterAlgoConfigs.algo = kBwdFilterDefaultAlgo;
          CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
              hndl,
              iDesc.descriptor,
              oDesc.descriptor,
              cDesc.descriptor,
              wDesc.descriptor,
              (cudnnConvolutionBwdFilterAlgo_t)bwdFilterAlgoConfigs.algo,
              &bwdFilterAlgoConfigs.memory));
          ws = af::array(bwdFilterAlgoConfigs.memory, af::dtype::b8);
          convAlgoConfigs->bwdFilter.set(
              inputX, batchSize, fp16Math, bwdFilterAlgoConfigs);
        }

        {
          DevicePtr wsPtr(ws);
          CUDNN_CHECK_ERR(cudnnConvolutionBackwardFilter(
              hndl,
              oneg,
              iDesc.descriptor,
              iPtr.get(),
              oDesc.descriptor,
              gradResultPtr.get(),
              cDesc.descriptor,
              (cudnnConvolutionBwdFilterAlgo_t)bwdFilterAlgoConfigs.algo,
              wsPtr.get(),
              bwdFilterAlgoConfigs.memory,
              zerog,
              wDesc.descriptor,
              gradWeightPtr.get()));
        }
        wt.addGrad(gradWeight);
      }
    };

    convolutionBackwardFilter(
        in.array(), wt.array(), gradOutput.array(), iDesc, wDesc, cDesc, oDesc);
  };

  if (hasBias) {
    return Variable(output, {in, wt, bs}, gradFunc);
  }
  return Variable(output, {in, wt}, gradFunc);
}

} // namespace fl
