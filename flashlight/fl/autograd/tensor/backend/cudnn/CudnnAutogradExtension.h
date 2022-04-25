/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
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
  static bool registered;

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
      const int groups) override;

  Tensor pool2d(
      const Tensor& input,
      const int wx,
      const int wy,
      const int sx,
      const int sy,
      const int px,
      const int py,
      const PoolingMode mode) override;

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
      const double epsilon) override;

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
      const float dropout) override;

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
      std::shared_ptr<DynamicBenchmark> dataGradBenchmark) override;

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
      std::shared_ptr<DynamicBenchmark> filterGradBenchmark) override;

  Tensor conv2dBackwardBias(
      const Tensor& gradOutput,
      const Tensor& bias,
      std::shared_ptr<DynamicBenchmark> biasGradBenchmark) override;

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
      const PoolingMode mode) override;

  // ]----- batchnorm
  std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
      const Tensor& gradOutput,
      const Tensor& saveMean,
      const Tensor& saveVar,
      const Tensor& input,
      const Tensor& weight,
      const std::vector<int>& axes,
      const bool train,
      const float epsilon) override;

  // ]----- rnn
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
      const float dropProb) override;
};

} // namespace fl
