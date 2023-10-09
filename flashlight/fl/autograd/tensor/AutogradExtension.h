/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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

/*
 * A base type that can be used to construct autograd payloads - this is
 * arbitrary data that can persist between forward and backward operations.
 */
struct AutogradPayloadData {};

/**
 * A simple type with semantics for assigning autograd payloads. It has the
 * following features:
 * - always shallow copies the underlying payload data
 * - is nullable (when inspected by value)
 * - using a shared pointer to a payload allows setting the underlying payload
 * transparently
 *
 * API functions defined to take an AutogradPayload come with the following
 * guarantees:
 * - TODO: write me -- same type of payload so can be safely downcast
 */
struct AutogradPayload {
  std::shared_ptr<detail::AutogradPayloadData> data;
};

} // namespace detail

class AutogradExtension : public TensorExtension<AutogradExtension> {
 public:
  virtual ~AutogradExtension() = default;

  static constexpr TensorExtensionType extensionType =
      TensorExtensionType::Autograd;

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
      const int groups,
      std::shared_ptr<detail::AutogradPayload> payload) = 0;

  virtual Tensor pool2d(
      const Tensor& input,
      const int wx,
      const int wy,
      const int sx,
      const int sy,
      const int px,
      const int py,
      const PoolingMode mode,
      std::shared_ptr<detail::AutogradPayload> payload) = 0;

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
      const double epsilon,
      std::shared_ptr<detail::AutogradPayload> payload) = 0;

  virtual std::tuple<Tensor, Tensor, Tensor> rnn(
      const Tensor& input,
      const Tensor& hiddenState,
      const Tensor& cellState,
      const Tensor& weights,
      const int hiddenSize,
      const int numLayers,
      const RnnMode mode,
      const bool bidirectional,
      const float dropout,
      std::shared_ptr<detail::AutogradPayload> payload) = 0;

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
      std::shared_ptr<DynamicBenchmark> dataGradBenchmark,
      std::shared_ptr<detail::AutogradPayload> payload) = 0;

  virtual std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
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
      std::shared_ptr<detail::AutogradPayload> autogradPayload) = 0;

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
      const PoolingMode mode,
      std::shared_ptr<detail::AutogradPayload> payload) = 0;

  // ]----- batchnorm
  virtual std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
      const Tensor& gradOutput,
      const Tensor& saveMean,
      const Tensor& saveVar,
      const Tensor& input,
      const Tensor& weight,
      const std::vector<int>& axes,
      const bool train,
      const float epsilon,
      std::shared_ptr<detail::AutogradPayload> payload) = 0;

  // ]----- rnn
  virtual std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
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
      std::shared_ptr<detail::AutogradPayload> payload) = 0;
};

} // namespace fl
