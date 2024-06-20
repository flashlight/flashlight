/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <cudnn.h>

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/common/DynamicBenchmark.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {

namespace {

std::unordered_map<fl::CudnnAutogradExtension::KernelMode, cudnnMathType_t>
    kKernelModesToCudnnMathType = {
        {fl::CudnnAutogradExtension::KernelMode::F32, CUDNN_DEFAULT_MATH},
        {fl::CudnnAutogradExtension::KernelMode::F32_ALLOW_CONVERSION,
         CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION},
        {fl::CudnnAutogradExtension::KernelMode::F16,
         CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION}};

const std::unordered_set<cudnnConvolutionFwdAlgo_t> kFwdPreferredAlgos = {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

const std::unordered_set<cudnnConvolutionBwdDataAlgo_t> kBwdDataPreferredAlgos =
    {CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
     CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};

const std::unordered_set<cudnnConvolutionBwdFilterAlgo_t>
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

// Get the algorithm which gives best performance.
// Since cuDNN doesn't support memory limits, we manually choose an algorithm
// which requires less than a specific workspace size.
template <typename T, typename ALGO_TYPE>
T getBestAlgorithm(
    const std::vector<T>& algoPerfs,
    const std::unordered_set<ALGO_TYPE>& preferredAlgos,
    const fl::dtype arithmeticPrecision) {
  T reserved;
  bool algoFound = false;
  for (const auto& algoPerf : algoPerfs) {
    if (algoPerf.status == CUDNN_STATUS_SUCCESS &&
        algoPerf.memory < kWorkspaceSizeLimitBytes) {
      if (!(arithmeticPrecision == fl::dtype::f16) ||
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
    const fl::dtype arithmeticPrecision) {
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
      fwdAlgoPerfs, kFwdPreferredAlgos, arithmeticPrecision);
}

cudnnConvolutionBwdDataAlgoPerf_t getBwdDataAlgo(
    const cudnnTensorDescriptor_t& xDesc,
    const cudnnFilterDescriptor_t& wDesc,
    const cudnnConvolutionDescriptor_t& convDesc,
    const cudnnTensorDescriptor_t& yDesc,
    bool /* isStrided */,
    const fl::dtype arithmeticPrecision) {
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
      bwdDataAlgoPerfs, kBwdDataPreferredAlgos, arithmeticPrecision);

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_DATA_ALGO_1
  // Seems to produce erroneous results on Tesla P100 GPUs.
  // 2: blacklist FFT algorithms for strided dgrad -
  // https://github.com/pytorch/pytorch/issues/16610
  bool isAlgoBlacklisted = false;
#ifndef FL_CUDNN_ALLOW_ALGO_1
  if (arithmeticPrecision != fl::dtype::f16 &&
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
    const fl::dtype arithmeticPrecision) {
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
      bwdFilterAlgoPerfs, kBwdFilterPreferredAlgos, arithmeticPrecision);

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_FILTER_ALGO_1
  // We do the blacklist here just to be safe as we did in BWD_DATA_ALGO_1
  bool isAlgoBlacklisted = false;
#ifndef FL_CUDNN_ALLOW_ALGO_1
  if (arithmeticPrecision != fl::dtype::f16 &&
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

/**
 * Sets the cudnnMathType according to a `KernelMode` value.
 *
 * @param[in] cDesc a reference to a `ConvDescriptor` for which the math type
 will
 * be set.
 * @param[in] kernelOptions a pointer to the DynamicBenchmarkOptions for the
 possible kernel modes.
 */
void setCudnnConvMathType(
    ConvDescriptor& cDesc,
    const std::shared_ptr<
        fl::DynamicBenchmarkOptions<fl::CudnnAutogradExtension::KernelMode>>&
        kernelOptions) {
  CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
      cDesc.descriptor,
      kKernelModesToCudnnMathType.at(kernelOptions->currentOption())));
}

void setDefaultMathType(ConvDescriptor& cDesc, const Tensor& input) {
  if (input.type() == fl::dtype::f16) {
    CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
        cDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetConvolutionMathType(cDesc.descriptor, CUDNN_DEFAULT_MATH));
  }
}

} // namespace

Tensor CudnnAutogradExtension::conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups,
    std::shared_ptr<detail::AutogradPayload>) {
  if (input.ndim() != 4) {
    throw std::invalid_argument(
        "conv2d: expects input tensor to be 4 dimensions: "
        "in WHCN ordering. Given tensor has " +
        std::to_string(input.ndim()) + " dimensions.");
  }

  auto hasBias = bias.elements() > 0;

  auto inDesc = TensorDescriptor(input);
  auto wtDesc = FilterDescriptor(weights);
  auto convDesc = ConvDescriptor(input.type(), px, py, sx, sy, dx, dy, groups);
  if (input.type() == fl::dtype::f16) {
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
  auto output = Tensor({odims[3], odims[2], odims[1], odims[0]}, input.type());
  auto outDesc = TensorDescriptor(output);

  auto handle = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  auto fwdAlgoBestPerf = getFwdAlgo(
      inDesc.descriptor,
      wtDesc.descriptor,
      convDesc.descriptor,
      outDesc.descriptor,
      input.type());

  Tensor wspace;

  try {
    wspace =
        Tensor({static_cast<long long>(fwdAlgoBestPerf.memory)}, fl::dtype::b8);
  } catch (const std::exception&) {
    fwdAlgoBestPerf.algo = kFwdDefaultAlgo;
    CUDNN_CHECK_ERR(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        inDesc.descriptor,
        wtDesc.descriptor,
        convDesc.descriptor,
        outDesc.descriptor,
        fwdAlgoBestPerf.algo,
        &fwdAlgoBestPerf.memory));
    wspace =
        Tensor({static_cast<long long>(fwdAlgoBestPerf.memory)}, fl::dtype::b8);
  }
  {
    DevicePtr inPtr(input);
    DevicePtr wtPtr(weights);
    DevicePtr outPtr(output);
    DevicePtr wspacePtr(wspace);
    // ensure cudnn compute stream waits on streams of input/output tensors
    relativeSync(cudnnStream, {input, weights, wspace, output});

    auto scalarsType =
        input.type() == fl::dtype::f16 ? fl::dtype::f32 : input.type();
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
      auto bsDesc = TensorDescriptor(bias);
      DevicePtr bsPtr(bias);
      // ensure cudnn compute stream waits on stream of bias tensor
      relativeSync(cudnnStream, {bias});
      CUDNN_CHECK_ERR(cudnnAddTensor(
          handle,
          one,
          bsDesc.descriptor,
          bsPtr.get(),
          one,
          outDesc.descriptor,
          outPtr.get()));
    }
    // ensure output stream waits on cudnn compute stream
    relativeSync({output}, cudnnStream);
  }
  return output;
}

Tensor CudnnAutogradExtension::conv2dBackwardData(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weight,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups,
    std::shared_ptr<DynamicBenchmark> dataGradBenchmark,
    std::shared_ptr<detail::AutogradPayload>) {
  auto hndl = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  auto scalarsType =
      input.type() == fl::dtype::f16 ? fl::dtype::f32 : input.type();
  const void* oneg = kOne(scalarsType);
  const void* zerog = kZero(scalarsType);

  // Create default descriptors assuming no casts. If dynamic
  // benchmarking suggests input or weight casting should occur, these
  // descriptors may not be used/new ones with the correct types will be
  // used instead.
  auto iDesc = TensorDescriptor(input);
  auto wDesc = FilterDescriptor(weight);
  auto cDesc = ConvDescriptor(input.type(), px, py, sx, sy, dx, dy, groups);
  auto oDesc = TensorDescriptor(gradOutput);

  setDefaultMathType(cDesc, input);

  // Gradients with respect to the input
  auto convolutionBackwardData =
      [&hndl, &cudnnStream, &dataGradBenchmark, oneg, zerog, dx, dy](
          const Tensor& inTensor,
          const Tensor& wtTensor,
          const Tensor& gradOutputTensor,
          TensorDescriptor& iDesc,
          FilterDescriptor& wDesc,
          ConvDescriptor& cDesc,
          TensorDescriptor& oDesc) -> Tensor {
    if (dataGradBenchmark && DynamicBenchmark::getBenchmarkMode()) {
      setCudnnConvMathType(
          cDesc,
          dataGradBenchmark->getOptions<DynamicBenchmarkOptions<KernelMode>>());
    }

    DevicePtr wPtr(wtTensor);
    // ensure cudnn compute stream waits on stream of weight tensor
    relativeSync(cudnnStream, {wtTensor});
    bool isStrided = (dx * dy) > 1;
    auto bwdDataAlgoBestPerf = getBwdDataAlgo(
        iDesc.descriptor,
        wDesc.descriptor,
        cDesc.descriptor,
        oDesc.descriptor,
        isStrided,
        inTensor.type());

    Tensor ws;
    try {
      ws = Tensor(
          {static_cast<long long>(bwdDataAlgoBestPerf.memory)}, fl::dtype::b8);
    } catch (const std::exception&) {
      bwdDataAlgoBestPerf.algo = kBwdDataDefaultAlgo;
      CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardDataWorkspaceSize(
          hndl,
          wDesc.descriptor,
          oDesc.descriptor,
          cDesc.descriptor,
          iDesc.descriptor,
          bwdDataAlgoBestPerf.algo,
          &bwdDataAlgoBestPerf.memory));
      ws = Tensor(
          {static_cast<long long>(bwdDataAlgoBestPerf.memory)}, fl::dtype::b8);
    }

    auto gradInput = Tensor(inTensor.shape(), inTensor.type());
    {
      DevicePtr gradInputPtr(gradInput);
      DevicePtr gradResultPtr(gradOutputTensor);
      DevicePtr wsPtr(ws);
      // ensure cudnn compute stream waits on streams of input/output tensors
      relativeSync(cudnnStream, {gradOutputTensor, ws, gradInput});
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
    // ensure stream of gradient waits on the cudnn compute stream
    relativeSync({gradInput}, cudnnStream);
    return gradInput;
  };

  Tensor dataGradOut;

  if (dataGradBenchmark && DynamicBenchmark::getBenchmarkMode()) {
    KernelMode dataBwdOption =
        dataGradBenchmark->getOptions<DynamicBenchmarkOptions<KernelMode>>()
            ->currentOption();

    if (input.type() == fl::dtype::f16 &&
        dataBwdOption == CudnnAutogradExtension::KernelMode::F32 &&
        dataBwdOption ==
            CudnnAutogradExtension::KernelMode::F32_ALLOW_CONVERSION) {
      // The input type of fp16, but the result of the dynamic benchmark
      // is that using fp32 kernels is faster for computing bwd with fp16
      // kernels, including the cast
      Tensor inTensorF32;
      Tensor wtTensorF32;
      Tensor gradOutputTensorF32;
      dataGradBenchmark->audit(
          [&input,
           &inTensorF32,
           &weight,
           &wtTensorF32,
           &gradOutput,
           &gradOutputTensorF32]() {
            inTensorF32 = input.astype(fl::dtype::f32);
            wtTensorF32 = weight.astype(fl::dtype::f32);
            gradOutputTensorF32 = gradOutput.astype(fl::dtype::f32);
          },
          /* incrementCount = */ false);

      auto iDescF32 = TensorDescriptor(inTensorF32);
      auto wDescF32 = FilterDescriptor(wtTensorF32);
      auto cDescF32 =
          ConvDescriptor(fl::dtype::f32, px, py, sx, sy, dx, dy, groups);
      auto oDescF32 = TensorDescriptor(gradOutputTensorF32);
      // core bwd data computation
      dataGradBenchmark->audit([&dataGradOut,
                                &convolutionBackwardData,
                                &inTensorF32,
                                &wtTensorF32,
                                &gradOutputTensorF32,
                                &iDescF32,
                                &wDescF32,
                                &cDescF32,
                                &oDescF32]() {
        dataGradOut = convolutionBackwardData(
            inTensorF32,
            wtTensorF32,
            gradOutputTensorF32,
            iDescF32,
            wDescF32,
            cDescF32,
            oDescF32);
      });

    } else {
      dataGradBenchmark->audit([&dataGradOut,
                                &convolutionBackwardData,
                                &input,
                                &weight,
                                &gradOutput,
                                &iDesc,
                                &wDesc,
                                &cDesc,
                                &oDesc]() {
        dataGradOut = convolutionBackwardData(
            input, weight, gradOutput, iDesc, wDesc, cDesc, oDesc);
      });
    }

  } else {
    // No benchmarking - proceed normally
    dataGradOut = convolutionBackwardData(
        input, weight, gradOutput, iDesc, wDesc, cDesc, oDesc);
  }

  return dataGradOut;
}

std::pair<Tensor, Tensor> CudnnAutogradExtension::conv2dBackwardFilterBias(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups,
    std::shared_ptr<DynamicBenchmark> filterGradBenchmark,
    std::shared_ptr<DynamicBenchmark> biasGradBenchmark,
    std::shared_ptr<detail::AutogradPayload>) {
  auto hndl = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  auto scalarsType =
      input.type() == fl::dtype::f16 ? fl::dtype::f32 : input.type();
  const void* oneg = kOne(scalarsType);
  const void* zerog = kZero(scalarsType);

  // Create default descriptors assuming no casts. If dynamic
  // benchmarking suggests input or weight casting should occur, these
  // descriptors may not be used/new ones with the correct types will be
  // used instead.
  auto iDesc = TensorDescriptor(input);
  auto wDesc = FilterDescriptor(weight);
  auto cDesc = ConvDescriptor(input.type(), px, py, sx, sy, dx, dy, groups);
  auto oDesc = TensorDescriptor(gradOutput);

  setDefaultMathType(cDesc, input);

  // Gradients with respect to the filter
  auto convolutionBackwardFilter =
      [&hndl, &cudnnStream, &filterGradBenchmark, oneg, zerog](
          const Tensor& inTensor,
          const Tensor& wtTensor,
          const Tensor& gradOutputTensor,
          TensorDescriptor& iDesc,
          FilterDescriptor& wDesc,
          ConvDescriptor& cDesc,
          TensorDescriptor& oDesc) -> Tensor {
    if (filterGradBenchmark && DynamicBenchmark::getBenchmarkMode()) {
      setCudnnConvMathType(
          cDesc,
          filterGradBenchmark
              ->getOptions<DynamicBenchmarkOptions<KernelMode>>());
    }

    DevicePtr iPtr(inTensor);
    // ensure cudnn compute stream waits on stream of input tensor
    relativeSync(cudnnStream, {inTensor});
    auto bwdFilterAlgoBestPerf = getBwdFilterAlgo(
        iDesc.descriptor,
        wDesc.descriptor,
        cDesc.descriptor,
        oDesc.descriptor,
        inTensor.type());

    Tensor ws;
    try {
      ws = Tensor(
          {static_cast<long long>(bwdFilterAlgoBestPerf.memory)},
          fl::dtype::b8);
    } catch (const std::exception&) {
      bwdFilterAlgoBestPerf.algo = kBwdFilterDefaultAlgo;
      CUDNN_CHECK_ERR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          hndl,
          iDesc.descriptor,
          oDesc.descriptor,
          cDesc.descriptor,
          wDesc.descriptor,
          bwdFilterAlgoBestPerf.algo,
          &bwdFilterAlgoBestPerf.memory));
      ws = Tensor(
          {static_cast<long long>(bwdFilterAlgoBestPerf.memory)},
          fl::dtype::b8);
    }

    auto gradWeight = Tensor(wtTensor.shape(), wtTensor.type());
    {
      DevicePtr gradWeightPtr(gradWeight);
      DevicePtr gradResultPtr(gradOutputTensor);
      DevicePtr wsPtr(ws);
      // ensure cudnn compute stream waits on streams of input/output tensors
      relativeSync(cudnnStream, {gradOutputTensor, ws, gradWeight});
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
    // ensure gradient tensor stream waits on cudnn compute stream
    relativeSync({gradWeight}, cudnnStream);
    return gradWeight;
  };

  Tensor filterGradOut;

  if (filterGradBenchmark && DynamicBenchmark::getBenchmarkMode()) {
    KernelMode dataBwdOption =
        filterGradBenchmark->getOptions<DynamicBenchmarkOptions<KernelMode>>()
            ->currentOption();

    if (input.type() == fl::dtype::f16 &&
        dataBwdOption == CudnnAutogradExtension::KernelMode::F32 &&
        dataBwdOption ==
            CudnnAutogradExtension::KernelMode::F32_ALLOW_CONVERSION) {
      // The input type of fp16, but the result of the dynamic benchmark is
      // that using fp32 kernels is faster for computing bwd with fp16
      // kernels, including the cast
      Tensor inTensorF32;
      Tensor wtTensorF32;
      Tensor gradOutputTensorF32;
      filterGradBenchmark->audit(
          [&input,
           &inTensorF32,
           &weight,
           &wtTensorF32,
           &gradOutput,
           &gradOutputTensorF32]() {
            inTensorF32 = input.astype(fl::dtype::f32);
            wtTensorF32 = weight.astype(fl::dtype::f32);
            gradOutputTensorF32 = gradOutput.astype(fl::dtype::f32);
          },
          /* incrementCount = */ false);

      auto iDescF32 = TensorDescriptor(inTensorF32);
      auto wDescF32 = FilterDescriptor(wtTensorF32);
      auto cDescF32 =
          ConvDescriptor(fl::dtype::f32, px, py, sx, sy, dx, dy, groups);
      auto oDescF32 = TensorDescriptor(gradOutputTensorF32);
      // core bwd data computation
      filterGradBenchmark->audit([&filterGradOut,
                                  &convolutionBackwardFilter,
                                  &inTensorF32,
                                  &wtTensorF32,
                                  &gradOutputTensorF32,
                                  &iDescF32,
                                  &wDescF32,
                                  &cDescF32,
                                  &oDescF32]() {
        filterGradOut = convolutionBackwardFilter(
            inTensorF32,
            wtTensorF32,
            gradOutputTensorF32,
            iDescF32,
            wDescF32,
            cDescF32,
            oDescF32);
      });

    } else {
      filterGradBenchmark->audit([&filterGradOut,
                                  &convolutionBackwardFilter,
                                  &input,
                                  &weight,
                                  &gradOutput,
                                  &iDesc,
                                  &wDesc,
                                  &cDesc,
                                  &oDesc]() {
        filterGradOut = convolutionBackwardFilter(
            input, weight, gradOutput, iDesc, wDesc, cDesc, oDesc);
      });
    }

  } else {
    filterGradOut = convolutionBackwardFilter(
        input, weight, gradOutput, iDesc, wDesc, cDesc, oDesc);
  }

  auto convolutionBackwardBias = [&hndl, &cudnnStream, oneg, zerog](
                                     const Tensor& bsTensor,
                                     const Tensor& gradOutput,
                                     const TensorDescriptor& oDesc) -> Tensor {
    auto gradBias = Tensor(bsTensor.shape(), bsTensor.type());
    {
      DevicePtr gradBiasPtr(gradBias);
      DevicePtr gradResultPtr(gradOutput);
      // ensure cudnn compute stream waits on gradient tensor streams
      relativeSync(cudnnStream, {gradOutput, gradBias});
      auto bDesc = TensorDescriptor(bsTensor);
      CUDNN_CHECK_ERR(cudnnConvolutionBackwardBias(
          hndl,
          oneg,
          oDesc.descriptor,
          gradResultPtr.get(),
          zerog,
          bDesc.descriptor,
          gradBiasPtr.get()));
    }
    // ensure gradient bias tensor stream waits on cudnn compute stream
    relativeSync({gradBias}, cudnnStream);
    return gradBias;
  };

  Tensor biasGradOut;

  if (!bias.isEmpty()) {
    if (biasGradBenchmark && DynamicBenchmark::getBenchmarkMode()) {
      KernelMode biasBwdOption =
          biasGradBenchmark->getOptions<DynamicBenchmarkOptions<KernelMode>>()
              ->currentOption();

      if (bias.type() == fl::dtype::f16 &&
          biasBwdOption == CudnnAutogradExtension::KernelMode::F32 &&
          biasBwdOption ==
              CudnnAutogradExtension::KernelMode::F32_ALLOW_CONVERSION) {
        // The input type of fp16, but the result of the dynamic benchmark is
        // that using fp32 kernels is faster for computing bwd with fp16
        // kernels, including the cast
        Tensor biasF32;
        Tensor gradOutputF32;
        // Time cast bias and grad output if benchmarking
        biasGradBenchmark->audit(
            [&bias, &gradOutput, &biasF32, &gradOutputF32]() {
              biasF32 = bias.astype(fl::dtype::f32);
              gradOutputF32 = gradOutput.astype(fl::dtype::f32);
            },
            /* incrementCount = */ false);
        auto oDescF32 = TensorDescriptor(gradOutputF32);
        // Perform bias gradient computation
        biasGradBenchmark->audit([&biasGradOut,
                                  &convolutionBackwardBias,
                                  &biasF32,
                                  &gradOutputF32,
                                  &oDescF32]() {
          biasGradOut =
              convolutionBackwardBias(biasF32, gradOutputF32, oDescF32);
        });
      } else {
        // Grad output and bias types are already the same, so perform the
        // computation using whatever input type is given
        biasGradBenchmark->audit([&biasGradOut,
                                  &convolutionBackwardBias,
                                  &bias,
                                  &gradOutput,
                                  &oDesc]() {
          biasGradOut = convolutionBackwardBias(bias, gradOutput, oDesc);
        });
      }
    } else {
      // No benchmark; proceed normally
      biasGradOut = convolutionBackwardBias(bias, gradOutput, oDesc);
    }
  }

  return {filterGradOut, biasGradOut};
}

} // namespace fl
