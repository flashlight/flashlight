

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Applies a 2D convolution over an input signal given filter weights. In the
 * simplest case, the output with shape [\f$X_{out}\f$, \f$Y_{out}\f$,
 * \f$C_{out}\f$, \f$N\f$] of the convolution with input [\f$X_{in}\f$,
 * \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight [\f$K_x\f$, \f$K_y\f$,
 * \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]

 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C_{in}\f$,
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
Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    int dx = 1,
    int dy = 1,
    int groups = 1,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks = nullptr);

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
Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    int dx = 1,
    int dy = 1,
    int groups = 1,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks = nullptr);

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
Tensor pool2d(
    const Tensor& input,
    int wx,
    int wy,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    PoolingMode mode = PoolingMode::MAX);

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
Tensor batchnorm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& runningMean,
    Tensor& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon);

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
* each RNN layer except the last one, with dropout probability equal to dropout

* @return a tuple of three Tensors:
* - `y`: input with shape [input size, batch size, sequence length *
* directions]
* - `hiddenState`: hidden state for the current time step
* - `cellState`: cell state for the current time step
*/
std::tuple<Tensor, Tensor, Tensor> rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropout);

namespace detail {

namespace {

struct RNNGradData {
  fl::Tensor dy;
  fl::Tensor dhy;
  fl::Tensor dcy;
};

} // namespace

/**
 * Computes the gradients for a 2D convolution. Computes gradients with respect
 to the input, weights, and bias, as specified. Gradients Applies a 2D
 convolution over an input signal given filter weights. In the
 * simplest case, the output with shape [\f$X_{out}\f$, \f$Y_{out}\f$,
 * \f$C_{out}\f$, \f$N\f$] of the convolution with input [\f$X_{in}\f$,
 * \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight [\f$K_x\f$, \f$K_y\f$,
 * \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]

 * @param input a Tensor with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C_{in}\f$,
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
Tensor conv2DBackward(
    Tensor& inputGradOut,
    Tensor& weightGradOut,
    Tensor& biasGradOut,
    const bool computeInputGrad,
    const bool computeWeightGrad,
    const bool computeBiasGrad,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& gradOutput,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks);

Tensor conv2DBackwardData(
    Tensor& dataGradOut,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weight,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> dataGradBenchmark);

Tensor conv2DBackwardWeight(
    Tensor& weightGradOut,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& weight,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> weightGradBenchmark);

Tensor conv2DBackwardBias(
    Tensor& biasGradOut,
    const Tensor& gradOutput,
    std::shared_ptr<detail::ConvBenchmarks> biasGradBenchmark);

void pool2dBackward(
    Tensor& inputGradOut,
    const bool computeInputGrad,
    const Tensor& input,
    const Tensor& gradOutput,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode,
    const Tensor& poolOutput);

void batchnormBackward(
    Tensor& inputGradOut,
    Tensor& weightGradOut,
    Tensor& biasGradOut,
    const bool computeInputGrad,
    const bool computeWeightGrad,
    const bool computeBiasGrad,
    const Tensor& gradOutput,
    const Tensor& saveMean,
    const Tensor& saveVar,
    const Shape& inputDescDims,
    const Shape& weightDescDims,
    const float epsilon);

void rnnBackward(
    Tensor& yGradOut,
    Tensor& hiddenStateGradOut,
    Tensor& cellStateGradOut,
    const bool computeYGrad,
    const bool computeHiddenStateGrad,
    const bool computeCellStateGrad,
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const Tensor& gradOutput,
    const std::shared_ptr<RNNGradData> gradData,
    const Tensor& y,
    const size_t workspaceSize,
    const size_t reserveSize,
    Tensor reserveSpace,
    const int numLayers,
    const int hiddenSize,
    const RnnMode mode,
    const bool bidirectional,
    const float dropProb);

} // namespace detail

} // namespace fl
