/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/autograd/tensor/AutogradOps.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorExtension.h"

namespace fl {

class DynamicBenchmark;

namespace detail {
struct RNNGradData;
}

class AutogradExtension : public TensorExtension<AutogradExtension> {
 public:
  virtual ~AutogradExtension() = default;

  static constexpr TensorExtensionType extensionType =
      TensorExtensionType::Autograd;

  /**
   * Create benchmark options
   */
  virtual std::shared_ptr<fl::DynamicBenchmark> createBenchmarkOptions() {
    return nullptr;
  }

  /**************************** Forward ****************************/
  virtual Tensor conv2d(
      const Tensor& input,
      const Tensor& weights,
      const Tensor& bias,
      const int sx,
      const int sy,
      const int px,
      const int py,
      const int dx,
      const int dy,
      const int groups) = 0;

  virtual Tensor pool2d(
      const Tensor& input,
      const int wx,
      const int wy,
      const int sx,
      const int sy,
      const int px,
      const int py,
      const PoolingMode mode) = 0;

  virtual Tensor batchnorm(
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
      const double epsilon) = 0;

  virtual std::tuple<Tensor, Tensor, Tensor> rnn(
      Tensor& reserveSpace,
      const Tensor& input,
      const Tensor& hiddenState,
      const Tensor& cellState,
      const Tensor& weights,
      const int hiddenSize,
      const int numLayers,
      const RnnMode mode,
      const bool bidirectional,
      const float dropout) = 0;

  /**************************** Backward ****************************/
  // ]----- conv2d
  virtual Tensor conv2dBackwardData(
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
      std::shared_ptr<DynamicBenchmark> dataGradBenchmark) = 0;

  virtual Tensor conv2dBackwardFilter(
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
      std::shared_ptr<DynamicBenchmark> filterGradBenchmark) = 0;

  virtual Tensor conv2dBackwardBias(
      const Tensor& gradOutput,
      const Tensor& bias,
      std::shared_ptr<DynamicBenchmark> biasGradBenchmark) = 0;

  // ]----- pool2D
  virtual Tensor pool2dBackward(
      const Tensor& gradOutput,
      const Tensor& input,
      const Tensor& poolOutput,
      const int wx,
      const int wy,
      const int sx,
      const int sy,
      const int px,
      const int py,
      const PoolingMode mode) = 0;

  // ]----- batchnorm
  virtual std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
      const Tensor& gradOutput,
      const Tensor& saveMean,
      const Tensor& saveVar,
      const Tensor& input,
      const Tensor& weight,
      const std::vector<int>& axes,
      const bool train,
      const float epsilon) = 0;

  // ]----- rnn
  virtual std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
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
      const float dropProb) = 0;
};

} // namespace fl
