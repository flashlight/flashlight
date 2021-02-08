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

// Get the algorithm which gives best performance.
// Since cuDNN doesn't support memory limits, we manually choose an algorithm
// which requires less than a specific workspace size.
template <typename T, typename ALGO_TYPE>
T getBestAlgorithm(
    const std::vector<T>& algoPerfs,
    const fl::cpp::fl_unordered_set<ALGO_TYPE>& preferredAlgos,
    const af::dtype arithmeticPrecision) {
  T reserved;
  bool algoFound = false;
  for (const auto& algoPerf : algoPerfs) {
    if (algoPerf.status == CUDNN_STATUS_SUCCESS &&
        algoPerf.memory < kWorkspaceSizeLimitBytes) {
      if (!(arithmeticPrecision == af::dtype::f16) ||
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
      &fwdAlgoPerfs,
      workSpace,
      workSpaceSize,
      /*exhaustiveSearch=*/false));

  return getBestAlgorithm(
      fwdAlgoPerfs, kFwdPreferredAlgos, arithmeticPrecision);
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
      bwdDataAlgoPerfs, kBwdDataPreferredAlgos, arithmeticPrecision);

  // We use a few hacks here to resolve some cuDNN bugs
  // 1: blacklist BWD_DATA_ALGO_1
  // Seems to produce erroneous results on Tesla P100 GPUs.
  // 2: blacklist FFT algorithms for strided dgrad -
  // https://github.com/pytorch/pytorch/issues/16610
  bool isAlgoBlacklisted = false;
#ifndef FL_CUDNN_ALLOW_ALGO_1
  if (arithmeticPrecision != af::dtype::f16 &&
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
      bwdFilterAlgoPerfs, kBwdFilterPreferredAlgos, arithmeticPrecision);

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

enum class KernelMode { F32 = 0, F32_ALLOW_CONVERSION = 1, F16 = 2 };
const fl::cpp::fl_unordered_map<KernelMode, cudnnMathType_t>
    kKernelModesToCudnnMathType = {
        {KernelMode::F32, CUDNN_DEFAULT_MATH},
        {KernelMode::F32_ALLOW_CONVERSION,
         CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION},
        {KernelMode::F16, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION}};

std::shared_ptr<fl::DynamicBenchmark> createBenchmarkOptions() {
  return std::make_shared<fl::DynamicBenchmark>(
      std::make_shared<fl::DynamicBenchmarkOptions<KernelMode>>(
          std::vector<KernelMode>({KernelMode::F32,
                                   KernelMode::F32_ALLOW_CONVERSION,
                                   KernelMode::F16}),
          fl::kDynamicBenchmarkDefaultCount));
}

/**
 * Sets the cudnnMathType according to a `KernelMode` value.
 *
 * @param[in] cDesc a pointer to a `ConvDescriptor` for which the math type will
 * be set.
 * @param[in] kernelOptions a pointer to the DynamicBenchmarkOptions for the
 possible kernel modes.
 */
void setCudnnMathType(
    fl::ConvDescriptor& cDesc,
    const std::shared_ptr<fl::DynamicBenchmarkOptions<KernelMode>>&
        kernelOptions) {
  CUDNN_CHECK_ERR(cudnnSetConvolutionMathType(
      cDesc.descriptor,
      kKernelModesToCudnnMathType.at(kernelOptions->currentOption())));
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
    std::shared_ptr<detail::ConvBenchmarks> benchmarks) {
  auto dummy_bias = Variable(af::array(input.type()), false);
  return conv2d(
      input, weights, dummy_bias, sx, sy, px, py, dx, dy, groups, benchmarks);
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
    std::shared_ptr<detail::ConvBenchmarks> benchmarks) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(in, wt, bs);

  auto input = FL_ADJUST_INPUT_TYPE(in);
  auto weights = FL_ADJUST_INPUT_TYPE(wt);
  auto bias = FL_ADJUST_INPUT_TYPE(bs);

  auto hasBias = bias.elements() > 0;

  auto inDesc = TensorDescriptor(input);
  auto wtDesc = FilterDescriptor(weights.array());
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
    DevicePtr wtPtr(weights.array());
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
      auto bsDesc = TensorDescriptor(bias.array());
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
  auto gradFunc = [sx, sy, px, py, dx, dy, hasBias, groups, benchmarks](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    // Create benchmarks if needed
    if (benchmarks && DynamicBenchmark::getBenchmarkMode()) {
      if (!benchmarks->bwdFilterBenchmark) {
        benchmarks->bwdFilterBenchmark = createBenchmarkOptions();
      }
      if (!benchmarks->bwdDataBenchmark) {
        benchmarks->bwdDataBenchmark = createBenchmarkOptions();
      }
      if (!benchmarks->bwdBiasBenchmark) {
        benchmarks->bwdBiasBenchmark = createBenchmarkOptions();
      }
    }

    auto& in = inputs[0];
    auto& wt = inputs[1];

    // Create default descriptors assuming no casts. If dynamic
    // benchmarking suggests input or weight casting should occur, these
    // descriptors may not be used/new ones with the correct types will be used
    // instead.
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

      if (benchmarks && DynamicBenchmark::getBenchmarkMode()) {
        KernelMode biasBwdOption =
            benchmarks->bwdBiasBenchmark
                ->getOptions<DynamicBenchmarkOptions<KernelMode>>()
                ->currentOption();

        if (in.type() == af::dtype::f16 && biasBwdOption == KernelMode::F32 &&
            biasBwdOption == KernelMode::F32_ALLOW_CONVERSION) {
          // The input type of fp16, but the result of the dynamic benchmark is
          // that using fp32 kernels is faster for computing bwd with fp16
          // kernels, including the cast
          af::array biasF32;
          af::array gradOutputF32;
          // Time cast bias and grad output if benchmarking
          benchmarks->bwdBiasBenchmark->audit(
              [&bias, &gradOutput, &biasF32, &gradOutputF32]() {
                biasF32 = bias.array().as(af::dtype::f32);
                gradOutputF32 = gradOutput.array().as(af::dtype::f32);
              },
              /* incrementCount = */ false);
          auto oDescF32 = TensorDescriptor(gradOutputF32);
          // Perform bias gradient computation
          benchmarks->bwdBiasBenchmark->audit([&convolutionBackwardBias,
                                               &biasF32,
                                               &gradOutputF32,
                                               &oDescF32]() {
            convolutionBackwardBias(biasF32, gradOutputF32, oDescF32);
          });
        } else {
          // Grad output and bias types are already the same, so perform the
          // computation using whatever input type is given
          benchmarks->bwdBiasBenchmark->audit(
              [&convolutionBackwardBias, &bias, &gradOutput, &oDesc]() {
                convolutionBackwardBias(
                    bias.array(), gradOutput.array(), oDesc);
              });
        }
      } else {
        // No benchmark; proceed normally
        convolutionBackwardBias(bias.array(), gradOutput.array(), oDesc);
      }
    }

    // Gradients with respect to the input
    auto convolutionBackwardData =
        [&hndl, &in, &benchmarks, oneg, zerog, dx, dy](
            const af::array& inArray,
            const af::array& wtArray,
            const af::array& gradOutputArray,
            TensorDescriptor& iDesc,
            FilterDescriptor& wDesc,
            ConvDescriptor& cDesc,
            TensorDescriptor& oDesc) {
          if (benchmarks && DynamicBenchmark::getBenchmarkMode()) {
            setCudnnMathType(
                cDesc,
                benchmarks->bwdDataBenchmark
                    ->getOptions<DynamicBenchmarkOptions<KernelMode>>());
          }

          DevicePtr wPtr(wtArray);
          if (in.isCalcGrad()) {
            bool isStrided = (dx * dy) > 1;
            auto bwdDataAlgoBestPerf = getBwdDataAlgo(
                iDesc.descriptor,
                wDesc.descriptor,
                cDesc.descriptor,
                oDesc.descriptor,
                isStrided,
                inArray.type());

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
            auto gradInput =
                Variable(af::array(inArray.dims(), inArray.type()), false);
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
        };

    if (benchmarks && DynamicBenchmark::getBenchmarkMode()) {
      KernelMode dataBwdOption =
          benchmarks->bwdDataBenchmark
              ->getOptions<DynamicBenchmarkOptions<KernelMode>>()
              ->currentOption();

      if (in.type() == af::dtype::f16 && dataBwdOption == KernelMode::F32 &&
          dataBwdOption == KernelMode::F32_ALLOW_CONVERSION) {
        // The input type of fp16, but the result of the dynamic benchmark is
        // that using fp32 kernels is faster for computing bwd with fp16
        // kernels, including the cast
        af::array inArrayF32;
        af::array wtArrayF32;
        af::array gradOutputArrayF32;
        benchmarks->bwdDataBenchmark->audit(
            [&in,
             &inArrayF32,
             &wt,
             &wtArrayF32,
             &gradOutput,
             &gradOutputArrayF32]() {
              inArrayF32 = in.array().as(af::dtype::f32);
              wtArrayF32 = wt.array().as(af::dtype::f32);
              gradOutputArrayF32 = gradOutput.array().as(af::dtype::f32);
            },
            /* incrementCount = */ false);

        auto iDescF32 = TensorDescriptor(inArrayF32);
        auto wDescF32 = FilterDescriptor(wtArrayF32);
        auto cDescF32 =
            ConvDescriptor(af::dtype::f32, px, py, sx, sy, dx, dy, groups);
        auto oDescF32 = TensorDescriptor(gradOutputArrayF32);
        // core bwd data computation
        benchmarks->bwdDataBenchmark->audit([&convolutionBackwardData,
                                             &inArrayF32,
                                             &wtArrayF32,
                                             &gradOutputArrayF32,
                                             &iDescF32,
                                             &wDescF32,
                                             &cDescF32,
                                             &oDescF32]() {
          convolutionBackwardData(
              inArrayF32,
              wtArrayF32,
              gradOutputArrayF32,
              iDescF32,
              wDescF32,
              cDescF32,
              oDescF32);
        });
      } else {
        benchmarks->bwdDataBenchmark->audit([&convolutionBackwardData,
                                             &in,
                                             &wt,
                                             &gradOutput,
                                             &iDesc,
                                             &wDesc,
                                             &cDesc,
                                             &oDesc]() {
          convolutionBackwardData(
              in.array(),
              wt.array(),
              gradOutput.array(),
              iDesc,
              wDesc,
              cDesc,
              oDesc);
        });
      }
    } else {
      // No benchmarking - proceed normally
      convolutionBackwardData(
          in.array(),
          wt.array(),
          gradOutput.array(),
          iDesc,
          wDesc,
          cDesc,
          oDesc);
    }

    // Gradients with respect to the filter
    auto convolutionBackwardFilter = [&hndl, &wt, &benchmarks, oneg, zerog](
                                         const af::array& inArray,
                                         const af::array& wtArray,
                                         const af::array& gradOutputArray,
                                         TensorDescriptor& iDesc,
                                         FilterDescriptor& wDesc,
                                         ConvDescriptor& cDesc,
                                         TensorDescriptor& oDesc) {
      if (benchmarks && DynamicBenchmark::getBenchmarkMode()) {
        setCudnnMathType(
            cDesc,
            benchmarks->bwdFilterBenchmark
                ->getOptions<DynamicBenchmarkOptions<KernelMode>>());
      }

      DevicePtr iPtr(inArray);
      if (wt.isCalcGrad()) {
        auto bwdFilterAlgoBestPerf = getBwdFilterAlgo(
            iDesc.descriptor,
            wDesc.descriptor,
            cDesc.descriptor,
            oDesc.descriptor,
            inArray.type());

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

    if (benchmarks && DynamicBenchmark::getBenchmarkMode()) {
      KernelMode dataBwdOption =
          benchmarks->bwdFilterBenchmark
              ->getOptions<DynamicBenchmarkOptions<KernelMode>>()
              ->currentOption();

      if (in.type() == af::dtype::f16 && dataBwdOption == KernelMode::F32 &&
          dataBwdOption == KernelMode::F32_ALLOW_CONVERSION) {
        // The input type of fp16, but the result of the dynamic benchmark is
        // that using fp32 kernels is faster for computing bwd with fp16
        // kernels, including the cast
        af::array inArrayF32;
        af::array wtArrayF32;
        af::array gradOutputArrayF32;
        benchmarks->bwdFilterBenchmark->audit(
            [&in,
             &inArrayF32,
             &wt,
             &wtArrayF32,
             &gradOutput,
             &gradOutputArrayF32]() {
              inArrayF32 = in.array().as(af::dtype::f32);
              wtArrayF32 = wt.array().as(af::dtype::f32);
              gradOutputArrayF32 = gradOutput.array().as(af::dtype::f32);
            },
            /* incrementCount = */ false);

        auto iDescF32 = TensorDescriptor(inArrayF32);
        auto wDescF32 = FilterDescriptor(wtArrayF32);
        auto cDescF32 =
            ConvDescriptor(af::dtype::f32, px, py, sx, sy, dx, dy, groups);
        auto oDescF32 = TensorDescriptor(gradOutputArrayF32);
        // core bwd data computation
        benchmarks->bwdFilterBenchmark->audit([&convolutionBackwardFilter,
                                               &inArrayF32,
                                               &wtArrayF32,
                                               &gradOutputArrayF32,
                                               &iDescF32,
                                               &wDescF32,
                                               &cDescF32,
                                               &oDescF32]() {
          convolutionBackwardFilter(
              inArrayF32,
              wtArrayF32,
              gradOutputArrayF32,
              iDescF32,
              wDescF32,
              cDescF32,
              oDescF32);
        });
      } else {
        benchmarks->bwdFilterBenchmark->audit([&convolutionBackwardFilter,
                                               &in,
                                               &wt,
                                               &gradOutput,
                                               &iDesc,
                                               &wDesc,
                                               &cDesc,
                                               &oDesc]() {
          convolutionBackwardFilter(
              in.array(),
              wt.array(),
              gradOutput.array(),
              iDesc,
              wDesc,
              cDesc,
              oDesc);
        });
      }
    } else {
      convolutionBackwardFilter(
          in.array(),
          wt.array(),
          gradOutput.array(),
          iDesc,
          wDesc,
          cDesc,
          oDesc);
    }
  };

  if (hasBias) {
    return Variable(output, {input, weights, bias}, gradFunc);
  }
  return Variable(output, {input, weights}, gradFunc);
}

} // namespace fl
