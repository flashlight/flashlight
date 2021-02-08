/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// https://www.internalfb.com/intern/diffusion/FBS/browsefile/master/fbcode/caffe2/aten/src/ATen/native/miopen/Conv_miopen.cpp?lines=295%2C325%2C370%2C372%2C573%2C653

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

namespace fl {
namespace {

constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

} // namespace

namespace miopen {
namespace immediate {

// Get the algorithm which gives best performance.
miopenConvSolution_t getBestAlgorithm(
    const std::vector<miopenConvSolution_t>& algoPerfs) {
  if (algoPerfs.empty()) {
    throw std::invalid_argument(
        "Empty input error at getBestAlgorithm(vector<miopenConvSolution_t>&)");
  }

  float fastestTime = std::numeric_limits<float>::max();
  int fastestIdx = -1;
  for (int i = 0; i < algoPerfs.size(); ++i) {
    if (algoPerfs[i].workspace_size < kWorkspaceSizeLimitBytes &&
        algoPerfs[i].time < fastestTime) {
      fastestTime = algoPerfs[i].time;
      fastestIdx = i;
    } else {
      FL_LOG(fl::INFO) << "algoPerfs[i]=" << PrettyString(algoPerfs[i])
                       << "fastestTime=" << fastestTime
                       << " kWorkspaceSizeLimitBytes="
                       << kWorkspaceSizeLimitBytes;
    }
  }
  if (fastestTime < std::numeric_limits<float>::max()) {
    return algoPerfs[fastestIdx];
  } else {
    throw std::runtime_error(
        "No matching algorithm at getBestAlgorithm(vector<miopenConvSolution_t>&)");
  }
}

miopenConvSolution_t getFwdAlgo(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t& xDesc,
    const void* x,
    const miopenTensorDescriptor_t& wDesc,
    const void* w,
    const miopenConvolutionDescriptor_t& convDesc,
    const miopenTensorDescriptor_t& yDesc,
    void* y) {
  size_t maxSolutionCount = 0;
  MIOPEN_CHECK_ERR(miopenConvolutionForwardGetSolutionCount(
      handle, wDesc, xDesc, convDesc, yDesc, &maxSolutionCount));

  size_t solutionCount = 0;
  std::vector<miopenConvSolution_t> solutions(maxSolutionCount);

  MIOPEN_CHECK_ERR(miopenConvolutionForwardGetSolution(
      handle,
      wDesc,
      xDesc,
      convDesc,
      yDesc,
      maxSolutionCount,
      &solutionCount,
      solutions.data()));

  solutions.resize(solutionCount);

  for (size_t i = 0; i < solutionCount; i++) {
    miopenConvSolution_t solution = solutions[i];
    MIOPEN_CHECK_ERR(miopenConvolutionForwardCompileSolution(
        handle, wDesc, xDesc, convDesc, yDesc, solution.solution_id));
  }

  return fl::miopen::immediate::getBestAlgorithm(solutions);
}

miopenConvSolution_t getBwdDataAlgo(
    const miopenTensorDescriptor_t& xDesc,
    const miopenTensorDescriptor_t& wDesc,
    const miopenConvolutionDescriptor_t& convDesc,
    const miopenTensorDescriptor_t& yDesc,
    bool isStrided,
    const af::dtype arithmeticPrecision) {
  size_t numBwdDataAlgoRequested = 0;
  size_t numBwdDataAlgoReturned = 0;

  MIOPEN_CHECK_ERR(miopenConvolutionBackwardDataGetSolutionCount(
      fl::getMiOpenHandle(),
      yDesc,
      wDesc,
      convDesc,
      xDesc,
      &numBwdDataAlgoRequested));

  std::vector<miopenConvSolution_t> bwdDataAlgoPerfs(numBwdDataAlgoRequested);

  // https://github.com/ROCmSoftwarePlatform/MIOpen/blob/1673b9f0ff6148f1972080240a70c73b5915ff0b/include/miopen/miopen.h#L1199
  MIOPEN_CHECK_ERR(miopenConvolutionBackwardDataGetSolution(
      fl::getMiOpenHandle(),
      yDesc,
      wDesc,
      convDesc,
      xDesc,
      numBwdDataAlgoRequested,
      &numBwdDataAlgoReturned,
      bwdDataAlgoPerfs.data()));

  for (size_t i = 0; i < numBwdDataAlgoReturned; i++) {
    miopenConvSolution_t solution = bwdDataAlgoPerfs[i];
    MIOPEN_CHECK_ERR(miopenConvolutionBackwardDataCompileSolution(
        fl::getMiOpenHandle(),
        yDesc,
        wDesc,
        convDesc,
        xDesc,
        solution.solution_id));
  }

  return fl::miopen::immediate::getBestAlgorithm(bwdDataAlgoPerfs);
}

miopenConvSolution_t getBwdFilterAlgo(
    const miopenTensorDescriptor_t& xDesc,
    const miopenTensorDescriptor_t& wDesc,
    const miopenConvolutionDescriptor_t& convDesc,
    const miopenTensorDescriptor_t& yDesc,
    const af::dtype arithmeticPrecision) {
  size_t numBwdFilterAlgoRequested = 0;
  size_t numBwdFilterAlgoReturned = 0;

  MIOPEN_CHECK_ERR(miopenConvolutionBackwardWeightsGetSolutionCount(
      fl::getMiOpenHandle(),
      yDesc,
      xDesc,
      convDesc,
      wDesc,
      &numBwdFilterAlgoRequested));

  std::vector<miopenConvSolution_t> bwdFilterAlgoPerfs(
      numBwdFilterAlgoRequested);

  miopenConvolutionBackwardWeightsGetSolution(
      fl::getMiOpenHandle(),
      yDesc,
      xDesc,
      convDesc,
      wDesc,
      numBwdFilterAlgoRequested,
      &numBwdFilterAlgoReturned,
      bwdFilterAlgoPerfs.data());

  for (size_t i = 0; i < numBwdFilterAlgoReturned; i++) {
    miopenConvSolution_t solution = bwdFilterAlgoPerfs[i];
    MIOPEN_CHECK_ERR(miopenConvolutionBackwardDataCompileSolution(
        fl::getMiOpenHandle(),
        yDesc,
        wDesc,
        convDesc,
        xDesc,
        solution.solution_id));
  }

  return fl::miopen::immediate::getBestAlgorithm(bwdFilterAlgoPerfs);
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
  FL_VARIABLE_DTYPES_MATCH_CHECK(input, weights, bias);

  auto hasBias = bias.elements() > 0;

  auto inDesc = TensorDescriptor(input);
  auto wtDesc = FilterDescriptor(weights);
  auto convDesc = ConvDescriptor(input.type(), px, py, sx, sy, dx, dy, groups);

  int dn = input.dims(3) + 2;
  std::vector<int> odims(dn); // in BDYX
  MIOPEN_CHECK_ERR(miopenGetConvolutionNdForwardOutputDim(
      convDesc.descriptor,
      inDesc.descriptor,
      wtDesc.descriptor,
      &dn,
      odims.data()));
  auto output = af::array(odims[3], odims[2], odims[1], odims[0], input.type());
  auto outDesc = TensorDescriptor(output);

  miopenHandle_t handle = getMiOpenHandle();
  {
    DevicePtr inPtr(input.array());
    DevicePtr wtPtr(weights.array());
    DevicePtr outPtr(output);

    miopenConvSolution_t fwdAlgo = fl::miopen::immediate::getFwdAlgo(
        handle,
        inDesc.descriptor,
        inPtr.get(),
        wtDesc.descriptor,
        wtPtr.get(),
        convDesc.descriptor,
        outDesc.descriptor,
        outPtr.get());

    size_t workSpaceSize = 0;
    MIOPEN_CHECK_ERR(miopenConvolutionForwardGetSolutionWorkspaceSize(
        handle,
        wtDesc.descriptor,
        inDesc.descriptor,
        convDesc.descriptor,
        outDesc.descriptor,
        fwdAlgo.solution_id,
        &workSpaceSize));

    // try {
    //   wspace = af::array(solutions[0].workspace_size, af::dtype::b8);
    // } catch (const std::exception& e) {
    //   if (solutions.size() > 1) {
    //     wspace = af::array(solutions[1].workspace_size, af::dtype::b8);
    //   } else {
    //     throw;
    //   }
    // }

    auto wspace = af::array(workSpaceSize, af::dtype::b8);
    DevicePtr wspacePtr(wspace);

    MIOPEN_CHECK_ERR(miopenConvolutionForwardImmediate(
        handle,
        wtDesc.descriptor,
        wtPtr.get(),
        inDesc.descriptor,
        inPtr.get(),
        convDesc.descriptor,
        outDesc.descriptor,
        outPtr.get(),
        wspacePtr.get(),
        workSpaceSize,
        fwdAlgo.solution_id));

    if (hasBias) {
      output = output + fl::tileAs(bias, output.dims()).array();
    }
  }

  auto gradFunc = [sx, sy, px, py, dx, dy, hasBias, groups](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
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

    auto hndl = getMiOpenHandle();

    auto scalarsType = in.type() == f16 ? f32 : in.type();
    const void* oneg = fl::kOne(scalarsType);
    const void* zerog = fl::kZero(scalarsType);

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
          MIOPEN_CHECK_ERR(miopenConvolutionBackwardBias(
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
    auto convolutionBackwardData = [&hndl, &in, oneg, zerog, dx, dy, &wt](
                                       const af::array& inArray,
                                       const af::array& wtArray,
                                       const af::array& gradOutputArray,
                                       TensorDescriptor& iDesc,
                                       FilterDescriptor& wDesc,
                                       ConvDescriptor& cDesc,
                                       TensorDescriptor& oDesc) {
      DevicePtr wPtr(wtArray);
      if (in.isCalcGrad()) {
        bool isStrided = (dx * dy) > 1;
        auto bwdDataAlgoBestPerf = fl::miopen::immediate::getBwdDataAlgo(
            iDesc.descriptor,
            wDesc.descriptor,
            cDesc.descriptor,
            oDesc.descriptor,
            isStrided,
            inArray.type());

        size_t workSpaceSize = 0;
        MIOPEN_CHECK_ERR(miopenConvolutionBackwardDataGetWorkSpaceSize(
            hndl,
            oDesc.descriptor,
            wDesc.descriptor,
            cDesc.descriptor,
            iDesc.descriptor,
            &workSpaceSize));

        auto ws = af::array(workSpaceSize, af::dtype::b8);

        auto gradInput =
            Variable(af::array(inArray.dims(), inArray.type()), false);
        {
          DevicePtr gradInputPtr(gradInput.array());
          DevicePtr gradResultPtr(gradOutputArray);
          DevicePtr wsPtr(ws);

          MIOPEN_CHECK_ERR(miopenConvolutionBackwardDataImmediate(
              hndl,
              oDesc.descriptor,
              gradResultPtr.get(),
              wDesc.descriptor,
              wPtr.get(),
              cDesc.descriptor,
              iDesc.descriptor,
              gradInputPtr.get(),
              wsPtr.get(),
              workSpaceSize,
              bwdDataAlgoBestPerf.solution_id));
        }
        in.addGrad(gradInput);
      }
    };

    // No benchmarking - proceed normally
    convolutionBackwardData(
        in.array(), wt.array(), gradOutput.array(), iDesc, wDesc, cDesc, oDesc);

    // Gradients with respect to the filter
    auto convolutionBackwardFilter = [&hndl, &wt, oneg, zerog](
                                         const af::array& inArray,
                                         const af::array& wtArray,
                                         const af::array& gradOutputArray,
                                         TensorDescriptor& iDesc,
                                         FilterDescriptor& wDesc,
                                         ConvDescriptor& cDesc,
                                         TensorDescriptor& oDesc) {
      DevicePtr iPtr(inArray);
      if (wt.isCalcGrad()) {
        auto bwdFilterAlgoBestPerf = getBwdFilterAlgo(
            iDesc.descriptor,
            wDesc.descriptor,
            cDesc.descriptor,
            oDesc.descriptor,
            inArray.type());

        af::array ws =
            af::array(bwdFilterAlgoBestPerf.workspace_size, af::dtype::b8);

        auto gradWeight =
            Variable(af::array(wtArray.dims(), wtArray.type()), false);
        {
          DevicePtr gradWeightPtr(gradWeight.array());
          DevicePtr gradResultPtr(gradOutputArray);
          DevicePtr wsPtr(ws);

          MIOPEN_CHECK_ERR(miopenConvolutionBackwardWeightsImmediate(
              hndl,
              oDesc.descriptor,
              gradResultPtr.get(),
              iDesc.descriptor,
              iPtr.get(),
              cDesc.descriptor,
              wDesc.descriptor,
              gradWeightPtr.get(),
              wsPtr.get(),
              bwdFilterAlgoBestPerf.workspace_size,
              bwdFilterAlgoBestPerf.solution_id));
        }
        wt.addGrad(gradWeight);
      }
    };

    convolutionBackwardFilter(
        in.array(), wt.array(), gradOutput.array(), iDesc, wDesc, cDesc, oDesc);
  };

  if (hasBias) {
    return Variable(output, {input, weights, bias}, gradFunc);
  }
  return Variable(output, {input, weights}, gradFunc);
}

} // namespace immediate
} // namespace miopen

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
  if (input.type() == f16) {
    throw std::runtime_error("Half precision is not supported in opencl.");
  }
  Variable dummy_bias = Variable(af::array(), false);
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
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks) {
  const int chan = input.dims(kChannelSizeIdx);
  if ((chan % groups) != 0) {
    throw std::runtime_error(
        "Number of channels must be devisible by number of groups");
  }
  if (input.type() == f32) {
    return fl::miopen::immediate::conv2d(
        input, weights, bias, sx, sy, px, py, dx, dy, groups);
  }
  std::stringstream ss;
  ss << "Type " << input.type() << " is not supported by MiOpen conv2d";
  throw std::invalid_argument(ss.str());
}

} // namespace fl
