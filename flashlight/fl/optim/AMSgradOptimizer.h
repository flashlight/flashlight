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

/** An implementation of the AMSgrad optimizer.
 * For more details see the paper
 * [On the Convergence of Adam and Beyond]
 *    https://openreview.net/pdf?id=ryQu7f-RZ).
 */
class FL_API AMSgradOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      FirstOrderOptimizer,
      beta1_,
      beta2_,
      eps_,
      wd_,
      biasedFirst_,
      biasedSecond_,
      maxExpAvgSq_)

  AMSgradOptimizer() = default; // Intentionally private

  float beta1_;
  float beta2_;
  float eps_;
  float wd_;
  std::vector<Tensor> biasedFirst_;
  std::vector<Tensor> biasedSecond_;
  std::vector<Tensor> maxExpAvgSq_;

 public:
  /** Construct an AMSgrad optimizer
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learningRate The learning rate.
   * @param beta1 AMSgrad hyperparameter \f$ \beta_1 \f$.
   * @param beta2 AMSgrad hyperparameter \f$ \beta_2 \f$.
   * @param epsilon A small value used for numerical stability.
   * @param weightDecay The amount of L2 weight decay to use for all the
   * parameters.
   */
  AMSgradOptimizer(
      const std::vector<Variable>& parameters,
      float learningRate,
      float beta1 = 0.9,
      float beta2 = 0.999,
      float epsilon = 1e-8,
      float weightDecay = 0);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AMSgradOptimizer)
