/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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
  return detail::conv2d(
      input,
      weights,
      bias,
      sx,
      sy,
      px,
      py,
      dx,
      dy,
      groups,
      /* payload = */ nullptr);
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
  return detail::pool2d(
      input, wx, wy, sx, sy, px, py, mode, /* payload = */ nullptr);
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
  return detail::batchnorm(
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
      epsilon,
      /* payload = */ nullptr);
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
  return detail::batchnorm(
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
      epsilon,
      /*payload = */ nullptr);
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
  return detail::rnn(
      input,
      hiddenState,
      cellState,
      weights,
      hiddenSize,
      numLayers,
      mode,
      bidirectional,
      dropout,
      /* payload = */ nullptr);
}

namespace detail {

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
    const int groups,
    std::shared_ptr<detail::AutogradPayload> payload) {
  return input.backend().getExtension<AutogradExtension>().conv2d(
      input, weights, bias, sx, sy, px, py, dx, dy, groups, payload);
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
    const double epsilon,
    std::shared_ptr<detail::AutogradPayload> payload) {
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
      epsilon,
      payload);
}

Tensor pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode,
    std::shared_ptr<detail::AutogradPayload> payload) {
  return input.backend().getExtension<AutogradExtension>().pool2d(
      input, wx, wy, sx, sy, px, py, mode, payload);
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
    const float dropout,
    std::shared_ptr<detail::AutogradPayload> payload) {
  return input.backend().getExtension<AutogradExtension>().rnn(
      input,
      hiddenState,
      cellState,
      weights,
      hiddenSize,
      numLayers,
      mode,
      bidirectional,
      dropout,
      payload);
}

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
    std::shared_ptr<DynamicBenchmark> dataGradBenchmark,
    std::shared_ptr<detail::AutogradPayload> payload) {
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
      dataGradBenchmark,
      payload);
}

std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& filter,
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
    std::shared_ptr<detail::AutogradPayload> payload) {
  return input.backend()
      .getExtension<AutogradExtension>()
      .conv2dBackwardFilterBias(
          gradOutput,
          input,
          filter,
          bias,
          sx,
          sy,
          px,
          py,
          dx,
          dy,
          groups,
          filterGradBenchmark,
          biasGradBenchmark,
          payload);
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
    const PoolingMode mode,
    std::shared_ptr<detail::AutogradPayload> payload) {
  return input.backend().getExtension<AutogradExtension>().pool2dBackward(
      gradOutput, input, poolOutput, wx, wy, sx, sy, px, py, mode, payload);
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
    const float epsilon,
    std::shared_ptr<detail::AutogradPayload> payload) {
  return gradOutput.backend()
      .getExtension<AutogradExtension>()
      .batchnormBackward(
          gradOutput,
          saveMean,
          saveVar,
          input,
          weight,
          axes,
          train,
          epsilon,
          payload);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const std::shared_ptr<detail::RNNGradData> gradData,
    const Tensor& output,
    const int numLayers,
    const int hiddenSize,
    const RnnMode mode,
    const bool bidirectional,
    const float dropProb,
    std::shared_ptr<detail::AutogradPayload> payload) {
  return input.backend().getExtension<AutogradExtension>().rnnBackward(
      input,
      hiddenState,
      cellState,
      weights,
      gradData,
      output,
      numLayers,
      hiddenSize,
      mode,
      bidirectional,
      dropProb,
      payload);
}

} // namespace detail

} // namespace fl
