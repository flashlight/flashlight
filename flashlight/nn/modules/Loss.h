/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <flashlight/common/Defines.h>
#include "Container.h"

namespace fl {

/**
 * A module denoting some loss computation.
 */
class Loss : public Container {
 public:
  Loss() = default;

  /**
   * Forward computation for the loss. With an input and a target.
   *
   * @param inputs a tensor with the the predicted values
   * @param targets a tensor with the target values
   */
  virtual Variable forward(const Variable& inputs, const Variable& targets) = 0;

  /// @private
  Variable forward(const Variable& inputs) override;

  Variable operator()(const Variable& inputs, const Variable& targets);

  virtual ~Loss() = default;

 private:
  FL_SAVE_LOAD_WITH_BASE(Container)
};

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
class MeanSquaredError : public Loss {
 public:
  MeanSquaredError() = default;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Loss)
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
class MeanAbsoluteError : public Loss {
 public:
  MeanAbsoluteError() = default;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Loss)
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
class BinaryCrossEntropy : public Loss {
 public:
  BinaryCrossEntropy() = default;

  using Loss::forward;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  /**
   * Perform forward loss computation with an additional weight tensor.
   *
   * @param inputs a tensor with the predicted values
   * @param targets a tensor with the taret values
   * @param weights a rescaling weight given to the loss of each element.
   */
  Variable forward(
      const Variable& inputs,
      const Variable& targets,
      const Variable& weights);

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Loss)
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
class CategoricalCrossEntropy : public Loss {
 private:
  ReduceMode reduction_;

  FL_SAVE_LOAD_WITH_BASE(Loss, reduction_)

 public:
  /**
   * Creates a `CategoricalCrossEntropy`.
   *
   * @param reduction a reduction with which to compute the loss. See
   * documentation on `ReduceMode` for available options.
   */
  explicit CategoricalCrossEntropy(ReduceMode reduction = ReduceMode::MEAN)
      : reduction_(reduction) {}

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

  std::string prettyString() const override;
};

typedef MeanSquaredError MSE;
typedef MeanAbsoluteError MAE;
typedef MeanAbsoluteError L1Loss;

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Loss)
CEREAL_REGISTER_TYPE(fl::MeanSquaredError)
CEREAL_REGISTER_TYPE(fl::MeanAbsoluteError)
CEREAL_REGISTER_TYPE(fl::BinaryCrossEntropy)
CEREAL_REGISTER_TYPE(fl::CategoricalCrossEntropy)
