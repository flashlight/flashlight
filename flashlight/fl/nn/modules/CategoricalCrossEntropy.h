/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

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
class CategoricalCrossEntropy : public BinaryModule {
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

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::CategoricalCrossEntropy)
CEREAL_CLASS_VERSION(fl::CategoricalCrossEntropy, 1)
