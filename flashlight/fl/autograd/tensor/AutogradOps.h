/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <tuple>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

class DynamicBenchmark;

namespace detail {
struct AutogradPayload;
}

/**
 * Applies a 2D convolution over an input signal given filter weights. In
 the
 * simplest case, the output with shape [\f$X_{out}\f$, \f$Y_{out}\f$,
 * \f$C_{out}\f$, \f$N\f$] of the convolution with input [\f$X_{in}\f$,
 * \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight [\f$K_x\f$, \f$K_y\f$,
 * \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]
 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$,
 \f$C_{in}\f$,
 * \f$N\f$]
 * @param weights a Tensor with shape [\f$K_x\f$, \f$K_y\f$, \f$C_{in}\f$,
 * \f$C_{out}\f$]
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param dx dilation along the first kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param dy dilation along the second kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param groups number of filter groups
 * @param benchmarks [optional] a `ConvBenchmarks` instance to use to
 * dynamically benchmark configuration attributes for computations.
 * @return a Tensor with shape [\f$X_{out}\f$, \f$Y_{out}\f$, \f$C_{out}\f$,
 * \f$N\f$]]
 */
FL_API Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const int sx = 1,
    const int sy = 1,
    const int px = 0,
    const int py = 0,
    const int dx = 1,
    const int dy = 1,
    const int groups = 1);

/**
 * Applies a 2D convolution over an input signal given filter weights and
 * biases. In the simplest case, the output with shape [\f$X_{out}\f$,
 * \f$Y_{out}\f$, \f$C_{out}\f$, \f$N\f$] of the convolution with input
 * [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight
 [\f$K_x\f$,
 * \f$K_y\f$, \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \text{bias}(C_{out_j}) +
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]

 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$,
 \f$C_{in}\f$,
 * \f$N\f$]
 * @param weights a Tensor with shape [\f$K_x\f$, \f$K_y\f$, \f$C_{in}\f$,
 * \f$C_{out}\f$]
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param dx dilation along the first kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param dy dilation along the second kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param groups number of filter groups
 * @param benchmarks [optional] a `ConvBenchmarks` instance to use to
 * dynamically benchmark configuration attributes for computations.
 * @param bias a Tensor with shape [\f$C_{out}\f$]
 * @return a Tensor with shape [\f$X_{out}\f$, \f$Y_{out}\f$, \f$C_{out}\f$,
 * \f$N\f$]]
 */
FL_API Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    const int sx = 1,
    const int sy = 1,
    const int px = 0,
    const int py = 0,
    const int dx = 1,
    const int dy = 1,
    const int groups = 1);

/**
 * Applies a 2D pooling over an input signal composed of several input planes.
 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C\f$,
 *   \f$N\f$]
 * @param wx pooling window size in the first dimension
 * @param wy pooling window size in the second dimension
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param mode pooling mode, which supports:
 * - MAX
 * - AVG_INCLUDE_PADDING
 * - AVG_EXCLUDE_PADDING
 */
FL_API Tensor pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx = 1,
    const int sy = 1,
    const int px = 0,
    const int py = 0,
    const PoolingMode mode = PoolingMode::MAX);

/**
* Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with
* additional channel dimension) as described in the paper
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
* Covariate Shift] (https://arxiv.org/abs/1502.03167) .
* \f[
*   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma +
* \beta
* \f]
* The mean and standard-deviation are calculated per-dimension over the
* mini-batches and \f$\gamma\f$ and \f$\beta\f$ are learnable parameter vectors
* of size \f$C\f$, the input size. By default, during training this layer keeps
* running estimates of its computed mean and variance, which are then used for
* normalization during evaluation.

* @param input a Tensor with size [\f$H\f$, \f$W\f$, \f$C\f$, \f$N\f$]
* @param weight a Tensor with size [\f$C\f$] for \f$\gamma\f$
* @param bias a Tensor with size [\f$C\f$] for \f$\beta\f$
* @param runningMean a buffer storing intermediate means during training
* @param runningVar a buffer storing intermediate variances during training
* @param axes dimensions to perform normalization on. If having size greater
* than one, reduce over all of them.
* @param train a flag indicating if running in training mode
* @param momentum value of momentum
* @param epsilon value of \f$\epsilon\f$
* @return a Tensor with same shape as `input`
*/
FL_API Tensor batchnorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& runningMean,
    Tensor& runningVar,
    const std::vector<int>& axes,
    const bool train,
    const double momentum,
    const double epsilon);

FL_API Tensor batchnorm(
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
    const double epsilon);

/**
* Applies an RNN unit to an input sequence.
* A general RNN operator can be expressed as following:
* \f[
  (h_t, c_t) = f_W(x_t, h_{t-1}, c_{t-1})
* \f]
* where \f$h_t\f$, \f$c_t\f$ are the hidden/cell state at time \f$t\f$,
* \f$x_t\f$ is the input at time \f$t\f$
*
* \note{cuDNN and oneDNN RNN weights are incompatible since the structure of
* the computation is different for each. There is no mapping between weights
* from each of those backends.}
*
* @param input Tensor of input with shape [input size, batch size, sequence
* length]
* @param hiddenState Tensor of hidden state with shape [hidden size, batch
* size, total layers]
* @param cellState [LSTM only] Tensor of cell state with same shape as
* hidden state
* @param weights Learnable parameters in the RNN unit
* @param hiddenSize number of features in the hidden state
* @param numLayers number of recurrent layers
* @param mode defines the type of RNN unit
*  - RELU
*  - TANH
*  - LSTM
*  - GRU
* @param bidirectional if `True`, becomes a bidirectional RNN, unidirectional
otherwise
* @param dropout if non-zero, introduces a `Dropout` layer on the outputs of
* each RNN layer except the last one,q with dropout probability equal to dropout

* @return a tuple of three Tensors:
* - `y`: input with shape [input size, batch size, sequence length *
* directions]
* - `hiddenState`: hidden state for the current time step
* - `cellState`: cell state for the current time step
*/
FL_API std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropout);

namespace detail {

FL_API Tensor conv2d(
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
    std::shared_ptr<detail::AutogradPayload> payload);

FL_API Tensor batchnorm(
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
    std::shared_ptr<detail::AutogradPayload> payload);

FL_API Tensor pool2d(
    const Tensor& input,
    const int wx,
    const int wy,
    const int sx,
    const int sy,
    const int px,
    const int py,
    const PoolingMode mode,
    std::shared_ptr<detail::AutogradPayload> payload);

FL_API std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropout,
    std::shared_ptr<detail::AutogradPayload> payload);

// Returns the gradient with respect to the input
FL_API Tensor conv2dBackwardData(
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
    std::shared_ptr<detail::AutogradPayload> payload);

// Returns the gradient with respect to the filter and bias (if given)
FL_API std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
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
    std::shared_ptr<detail::AutogradPayload> payload);

FL_API Tensor pool2dBackward(
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
    std::shared_ptr<detail::AutogradPayload> payload);

// Returns the gradinets with respect tot he input, weight, and bias,
// respectively
// Why one function for gradient of all of them? Most implementations don't
// support computing separate gradients. If support for this is added in most
// places, split out this function.
FL_API std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
    const Tensor& gradOutput,
    const Tensor& saveMean,
    const Tensor& saveVar,
    const Tensor& input,
    const Tensor& weight,
    const std::vector<int>& axes,
    const bool train,
    const float epsilon,
    std::shared_ptr<detail::AutogradPayload> payload);

struct RNNGradData {
  fl::Tensor dy;
  fl::Tensor dhy;
  fl::Tensor dcy;
};

// input gradient, hidden state gradient, cell state gradient, weights
// gradient
// @param[in] gradData grad output for each comp
FL_API std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
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
    std::shared_ptr<detail::AutogradPayload> payload);

} // namespace detail

} // namespace fl
