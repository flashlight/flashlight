/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>

#include <arrayfire.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

namespace fl {

Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon) {
  // Check if axes is valid
  auto maxAxis = *std::max_element(axes.begin(), axes.end());
  auto minAxis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (maxAxis - minAxis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("batchnorm() axes array should be continuous");
  }

  std::vector<int> axisComplement;
  for (int d = 0; d < AF_MAX_DIMS; ++d) {
    if (std::find(axes.begin(), axes.end(), d) == axes.end()) {
      axisComplement.push_back(d);
    }
  }
  af::dim4 featDims(1, 1, 1, 1);
  auto normDims = input.dims();
  for (auto ax : axes) {
    featDims[ax] = input.dims(ax);
    normDims[ax] = 1;
  }

  if (runningMean.isempty()) {
    runningMean =
        Variable(af::constant(0.0, featDims.elements(), input.type()), false);
  }
  if (runningVar.isempty()) {
    runningVar =
        Variable(af::constant(1.0, featDims.elements(), input.type()), false);
  }
  auto runningMeanDims = fl::moddims(runningMean, featDims);
  auto runningVarDims = fl::moddims(runningVar, featDims);

  fl::Variable result;
  if (train) {
    auto inputCopyNoGrad = Variable(input.array(), false);
    auto sampleMean = fl::mean(input, axisComplement);
    auto sampleVar = fl::var(
        input,
        axisComplement,
        /*isbiased=*/true);

    result = (input - fl::tileAs(sampleMean, input)) /
        fl::tileAs(fl::sqrt(sampleVar + epsilon), input);

    runningMeanDims = (1 - momentum) * runningMeanDims + momentum * sampleMean;
    runningVarDims = (1 - momentum) * runningVarDims + momentum * sampleVar;
    runningMean = fl::moddims(runningMeanDims, runningMean.dims());
    runningVar = fl::moddims(runningVarDims, runningVar.dims());
  } else {
    result = (input - fl::tileAs(runningMeanDims, input)) /
        fl::tileAs(fl::sqrt(runningVarDims + epsilon), input);
  }

  if (!weight.isempty()) {
    result = result * fl::tileAs(fl::moddims(weight, featDims), input);
  }

  if (!bias.isempty()) {
    result = result + fl::tileAs(fl::moddims(bias, featDims), input);
  }

  return result;
}

} // namespace fl
