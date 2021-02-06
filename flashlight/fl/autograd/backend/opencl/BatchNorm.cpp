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
#include "flashlight/fl/common/DevicePtr.h"

namespace fl {

namespace {

// Flashlight accept HWCN order according to docs
constexpr size_t kHIdx = 0;
constexpr size_t kWIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

} // namespace

Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double epsilon) {
  auto output = af::array(input.dims(), input.type());

  int nfeatures = 1;
  for (auto ax : axes) {
    nfeatures *= input.dims(ax);
  }

  if (runningVar.isempty()) {
    runningVar = Variable(af::constant(1.0, nfeatures, input.type()), false);
  }

  if (runningMean.isempty()) {
    runningMean = Variable(af::constant(0.0, nfeatures, input.type()), false);
  }

  // Check if axes are valid
  auto max_axis = *std::max_element(axes.begin(), axes.end());
  auto min_axis = *std::min_element(axes.begin(), axes.end());
  bool axesContinuous = (axes.size() == (max_axis - min_axis + 1));
  if (!axesContinuous) {
    throw std::invalid_argument("axis array should be continuous");
  }

  /* ... */

  /****************************************************************************/
  // Setup backward func

  auto gradFunc =
      [train, epsilon, nfeatures
       /* ... */](std::vector<Variable>& inputs, const Variable& grad_output) {
        if (!train) {
          throw std::logic_error(
              "can't compute batchnorm grad when train was not specified");
        }

        auto& inputRef = inputs[0];
        auto weightRef = inputs[1].isempty()
            ? Variable(af::constant(1.0, nfeatures, inputRef.type()), false)
            : inputs[1];
        auto biasRef = inputs[2].isempty()
            ? Variable(af::constant(0.0, nfeatures, inputRef.type()), false)
            : inputs[2];
        auto grad_input =
            Variable(af::array(inputRef.dims(), inputRef.type()), false);

        /* ... */

        // Update grad
        inputRef.addGrad(grad_input);
        // extracting grads from grad_weightsDNNL for weight and bias
        if (weightRef.isCalcGrad()) {
          auto gradWeight = Variable(/* ... */);
          weightRef.addGrad(gradWeight);

          auto gradBias = Variable(/* ... */);
          if (!biasRef.isempty()) {
            biasRef.addGrad(gradBias);
          }
        }
      };

  throw std::runtime_error("batchnorm not implemented for opencl");

  return Variable(output, {input, weight, bias}, gradFunc);
}

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
  if (input.type() == f16) {
    throw std::runtime_error("Half precision is not supported in opencl.");
  }

  return batchnorm(
      input, weight, bias, runningMean, runningVar, axes, train, epsilon);
}

} // namespace fl
