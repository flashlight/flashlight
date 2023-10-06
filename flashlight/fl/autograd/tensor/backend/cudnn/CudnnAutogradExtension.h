/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "flashlight/fl/autograd/tensor/AutogradExtension.h"

namespace fl {

class DynamicBenchmark;

class CudnnAutogradExtension : public AutogradExtension {
  // TODO(jacobkahn): implement getCudnnHandle

 public:
  bool isDataTypeSupported(const fl::dtype& dtype) const override;

  enum class KernelMode { F32 = 0, F32_ALLOW_CONVERSION = 1, F16 = 2 };

  std::shared_ptr<fl::DynamicBenchmark> createBenchmarkOptions() override;

  /**************************** Forward ****************************/
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
      std::shared_ptr<detail::AutogradPayload> payload) override;

  Tensor pool2d(
      const Tensor& input,
      const int wx,
      const int wy,
      const int sx,
      const int sy,
      const int px,
      const int py,
      const PoolingMode mode,
      std::shared_ptr<detail::AutogradPayload> payload) override;

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
      std::shared_ptr<detail::AutogradPayload> payload) override;

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
      std::shared_ptr<detail::AutogradPayload> payload) override;

  /**************************** Backward ****************************/
  // ]----- Convolution
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
      std::shared_ptr<detail::AutogradPayload> payload) override;

  std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
      const Tensor& gradOutput,
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
      std::shared_ptr<DynamicBenchmark> filterBench,
      std::shared_ptr<DynamicBenchmark> biasBench,
      std::shared_ptr<detail::AutogradPayload> autogradPayload) override;

  // ]----- pool2D
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
      std::shared_ptr<detail::AutogradPayload> payload) override;

  // ]----- batchnorm
  std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
      const Tensor& gradOutput,
      const Tensor& saveMean,
      const Tensor& saveVar,
      const Tensor& input,
      const Tensor& weight,
      const std::vector<int>& axes,
      const bool train,
      const float epsilon,
      std::shared_ptr<detail::AutogradPayload> payload) override;

  // ]----- rnn
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
      std::shared_ptr<detail::AutogradPayload> payload) override;
};

} // namespace fl
