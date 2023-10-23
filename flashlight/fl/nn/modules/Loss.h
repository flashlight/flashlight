/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/AdaptiveSoftMax.h"
#include "flashlight/fl/nn/modules/Container.h"

namespace fl {

class Shape;
class Tensor;

/**
 * Computes the [mean squared
 error](https://en.wikipedia.org/wiki/Mean_squared_error) between elements
 * across two tensors:
 * \f[
   \mathcal{L}(x, y) = \frac{1}{n} \sum_{i = 0}^n \left( x_i - y_i \right)^2
   \f]
 * for input tensor \f$x\f$ and target tensor \f$y\f$ each of which contain
 \f$n\f$ elements.
 */
class FL_API MeanSquaredError : public BinaryModule {
 public:
  MeanSquaredError() = default;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(BinaryModule)
};

/**
 * Computes the [mean absolute
 error](https://en.wikipedia.org/wiki/Mean_absolute_error) (equivalent to the
 \f$L_1\f$ loss):
 * \f[
   \mathcal{L}(x, y) = \frac{1}{n} \sum_{i = 0}^n \left| x_i - y_i \right|
   \f]
 * for input tensor \f$x\f$ and target tensor \f$y\f$ each of which contain
 \f$n\f$ elements.
 */
class FL_API MeanAbsoluteError : public BinaryModule {
 public:
  MeanAbsoluteError() = default;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(BinaryModule)
};

/**
 * Computes the binary cross entropy loss between an input tensor \f$x\f$ and a
 * target tensor \f$y\f$. The binary cross entropy loss is:
 * \f[
   B(x, y) = \frac{1}{n} \sum_{i = 0}^n -\left( w_i \times (y_i \times \log(x_i)
   + (1 - y_i) \times \log(1 - x_i)) \right) \f]
 * where \f$w\f$ is an optional weight parameter for rescaling.
 *
 * Both the inputs and the targets are expected to be between 0 and 1.
 */
class FL_API BinaryCrossEntropy : public BinaryModule {
 public:
  BinaryCrossEntropy() = default;

  using BinaryModule::forward;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  /**
   * Perform forward loss computation with an additional weight tensor.
   *
   * @param inputs a tensor with the predicted values
   * @param targets a tensor with the target values
   * @param weights a rescaling weight given to the loss of each element.
   */
  Variable forward(
      const Variable& inputs,
      const Variable& targets,
      const Variable& weights);

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(BinaryModule)
};

/**
 * Computes the categorical cross entropy loss between an input and a target
 * tensor. The input is expected to contain log probabilities (which can be
 * accomplished via `LogSoftmax`). The targets should contain the index of the
 * ground truth class for each input example.
 *
 * In the batch case, the output loss tensor \f$\{l_1,...,l_N\}^\top\f$, put
 \f$l_n = -x_{n, y_n}\f$
 * (only consider the probability of the correct class). Then reduce via:
 * \f[
   \mathcal{L}(x, y) = \sum_{i = 1}^N l_i
   \f]
 * if using a sum reduction,
 * \f[
   \mathcal{L}(x, y) = \frac{1}{n} \sum_{i = 1}^N l_i
   \f]
 * if using a mean reduction. If using no reduction ('none'), the result will be
 * reshaped to the target dimensions, giving a loss for each example. See
 * `ReduceMode`.
 */
class FL_API CategoricalCrossEntropy : public BinaryModule {
 private:
  ReduceMode reduction_;
  int ignoreIndex_{-1};

  FL_SAVE_LOAD_WITH_BASE(
      BinaryModule,
      reduction_,
      fl::versioned(ignoreIndex_, 1))

 public:
  /**
   * Creates a `CategoricalCrossEntropy`.
   *
   * @param reduction a reduction with which to compute the loss. See
   * documentation on `ReduceMode` for available options.
   * @param ignoreIndex a target value that is ignored and does not contribute
   * to the loss or the input gradient. If `reduce` is MEAN, the loss is
   * averaged over non-ignored targets.
   */
  explicit CategoricalCrossEntropy(
      ReduceMode reduction = ReduceMode::MEAN,
      int ignoreIndex = -1)
      : reduction_(reduction), ignoreIndex_(ignoreIndex) {}

  /**
   * Computes the categorical cross entropy loss for some input and target
   * tensors.
   *
   * @param inputs a `Variable` with shape [\f$C\f$, \f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$] where \f$C\f$ is the number of classes.
   * @param targets an integer `Variable` with shape [\f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$]. The values must be in [\f$0\f$, \f$C - 1\f$]
   */
  Variable forward(const Variable& inputs, const Variable& targets) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

/**
 * An efficient approximation of the softmax function and negative
 * log-likelihood loss. Computes the Adaptive Softmax, as given by [Grave et al
 * (2017)](https://arxiv.org/pdf/1609.04309.pdf): _Efficient softmax
 * approximation for GPUs_. Efficient when the number of classes over which the
 * softmax is being computed is very high and the label distribution is highly
 * imbalanced.
 *
 * Adaptive softmax buckets the inputs based on their frequency, where clusters
 * may be different number of targets each. For each minibatch, only clusters
 * for which at least one target is present are evaluated. Forward pass for
 * low-frequency inputs are approximated with lower rank matrices so as to speed
 * up computation.
 */
class FL_API AdaptiveSoftMaxLoss : public BinaryModule {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      BinaryModule,
      activation_,
      reduction_,
      fl::versioned(ignoreIndex_, 1))
  std::shared_ptr<AdaptiveSoftMax> activation_;
  ReduceMode reduction_;
  int ignoreIndex_{-1};

  Variable
  cast(const Variable& input, const Shape& outDims, const Tensor& indices);

 public:
  AdaptiveSoftMaxLoss() = default;

  /**
   * Create an `AdaptiveSoftMaxLoss` with given parameters
   *
   * @param reduction the reduction mode - see `ReductionMode` See
   * documentation on `ReduceMode` for available options.
   * @param ignoreIndex a target value that is ignored and does not contribute
   * to the loss or the input gradient. If `reduce` is MEAN, the loss is
   * averaged over non-ignored targets.
   */
  explicit AdaptiveSoftMaxLoss(
      std::shared_ptr<AdaptiveSoftMax> activation,
      ReduceMode reduction = ReduceMode::MEAN,
      int ignoreIndex = -1);
  std::shared_ptr<AdaptiveSoftMax> getActivation() const;

  /**
   * Computes the categorical cross entropy loss for some input and target
   * tensors (uses adaptive softmax function to do this efficiently)
   *
   * @param inputs a `Variable` with shape [\f$C\f$, \f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$] where \f$C\f$ is the number of classes.
   * @param targets an integer `Variable` with shape [\f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$]. The values must be in [\f$0\f$, \f$C - 1\f$]
   */
  Variable forward(const Variable& inputs, const Variable& targets) override;

  void setParams(const Variable& var, int position) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

typedef MeanSquaredError MSE;
typedef MeanAbsoluteError MAE;
typedef MeanAbsoluteError L1Loss;

} // namespace fl

CEREAL_REGISTER_TYPE(fl::MeanSquaredError)
CEREAL_REGISTER_TYPE(fl::MeanAbsoluteError)
CEREAL_REGISTER_TYPE(fl::BinaryCrossEntropy)
CEREAL_REGISTER_TYPE(fl::CategoricalCrossEntropy)
CEREAL_REGISTER_TYPE(fl::AdaptiveSoftMaxLoss)
CEREAL_CLASS_VERSION(fl::CategoricalCrossEntropy, 1)
CEREAL_CLASS_VERSION(fl::AdaptiveSoftMaxLoss, 1)
