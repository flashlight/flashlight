/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/optim/Optimizers.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/** An implementation of the Adagrad optimizer.
 * For more details see the paper
 * [Adaptive Subgradient Methods for Online Learning and Stochastic
 * Optimization](
 *    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
 */
class FL_API AdagradOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(FirstOrderOptimizer, eps_, wd_, variance_)

  AdagradOptimizer() = default; // Intentionally private

  float eps_;
  float wd_;
  std::vector<Tensor> variance_; // store sum_{tau=0}^{tau=t} grad_tau*grad_tau

 public:
  /** Construct an Adagrad optimizer
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learningRate The learning rate.
   * @param epsilon A small value used for numerical stability.
   */
  explicit AdagradOptimizer(
      const std::vector<Variable>& parameters,
      float learningRate = 1.0,
      float epsilon = 1e-8,
      float weightDecay = 0);

  void step() override;

  std::string prettyString() const override;
};
} // namespace fl

CEREAL_REGISTER_TYPE(fl::AdagradOptimizer)
