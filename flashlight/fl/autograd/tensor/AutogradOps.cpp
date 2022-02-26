/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/AutogradOps.h"
#include "flashlight/fl/autograd/tensor/AutogradExtension.h"
#include "flashlight/fl/autograd/tensor/AutogradExtensionBackends.h"
#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups) {
  auto dummyBias = Tensor(input.type());
  return conv2d(input, weights, dummyBias, sx, sy, px, py, dx, dy, groups);
}

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups) {
  return input.backend().getExtension<AutogradExtension>().conv2d(
      input, weights, bias, sx, sy, px, py, dx, dy, groups);
}

Tensor pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode) {
  return input.backend().getExtension<AutogradExtension>().pool2d(
      input, wx, wy, sx, sy, px, py, mode);
}

Tensor batchnorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& runningMean,
    Tensor& runningVar,
    const std::vector<int>& axes,
    const bool train,
    const double momentum,
    const double epsilon) {
  Tensor saveMean; // empty
  Tensor saveVar; // empty
  return batchnorm(
      saveMean,
      saveVar,
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      axes,
      train,
      momentum,
      epsilon);
}

Tensor batchnorm(
    Tensor& saveMean,
    Tensor& saveVar,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& runningMean,
    Tensor& runningVar,
    const std::vector<int>& axes,
    const bool train,
    const double momentum,
    const double epsilon) {
  return input.backend().getExtension<AutogradExtension>().batchnorm(
      saveMean,
      saveVar,
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      axes,
      train,
      momentum,
      epsilon);
}

std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropout) {
  Tensor reserveSpace;
  return rnn(
      reserveSpace,
      input,
      hiddenState,
      cellState,
      weights,
      hiddenSize,
      numLayers,
      mode,
      bidirectional,
      dropout);
}

std::tuple<Tensor, Tensor, Tensor> rnn(
    Tensor& reserveSpace,
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropout) {
  return input.backend().getExtension<AutogradExtension>().rnn(
      reserveSpace,
      input,
      hiddenState,
      cellState,
      weights,
      hiddenSize,
      numLayers,
      mode,
      bidirectional,
      dropout);
}

namespace detail {

Tensor conv2dBackwardData(
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
    std::shared_ptr<DynamicBenchmark> dataGradBenchmark) {
  return input.backend().getExtension<AutogradExtension>().conv2dBackwardData(
      gradOutput,
      input,
      weight,
      sx,
      sy,
      px,
      py,
      dx,
      dy,
      groups,
      dataGradBenchmark);
}

Tensor conv2dBackwardFilter(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& filter,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const int dx,
    const int dy,
    const int groups,
    std::shared_ptr<DynamicBenchmark> filterGradBenchmark) {
  return input.backend().getExtension<AutogradExtension>().conv2dBackwardFilter(
      gradOutput,
      input,
      filter,
      sx,
      sy,
      px,
      py,
      dx,
      dy,
      groups,
      filterGradBenchmark);
}

// Returns the gradient with respect to the bias
Tensor conv2dBackwardBias(
    const Tensor& gradOutput,
    const Tensor& bias,
    std::shared_ptr<DynamicBenchmark> biasGradBenchmark) {
  return bias.backend().getExtension<AutogradExtension>().conv2dBackwardBias(
      gradOutput, bias, biasGradBenchmark);
}

Tensor pool2dBackward(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& poolOutput,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode) {
  return input.backend().getExtension<AutogradExtension>().pool2dBackward(
      gradOutput, input, poolOutput, wx, wy, sx, sy, px, py, mode);
}

// Returns the gradinets with respect tot he input, hidden state cell state, and
// weights respectively
// Why one function for gradient of all of them? Most
// implementations don't support computing separate gradients. If support for
// this is added in most places, split out this function.
std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput,
    const Tensor& saveMean,
    const Tensor& saveVar,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<int>& axes,
    const bool train,
    const float epsilon) {
  return gradOutput.backend()
      .getExtension<AutogradExtension>()
      .batchnormBackward(
          gradOutput, saveMean, saveVar, input, weight, axes, train, epsilon);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const std::shared_ptr<detail::RNNGradData> gradData,
    const Tensor& reserveSpace,
    const Tensor& output,
    const int numLayers,
    const int hiddenSize,
    const RnnMode mode,
    const bool bidirectional,
    const float dropProb) {
  return input.backend().getExtension<AutogradExtension>().rnnBackward(
      input,
      hiddenState,
      cellState,
      weights,
      gradData,
      reserveSpace,
      output,
      numLayers,
      hiddenSize,
      mode,
      bidirectional,
      dropProb);
}

} // namespace detail

} // namespace fl
