/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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

#include <vector>

#include <arrayfire.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/optim/Optimizers.h"

namespace fl {

/** An implementation of the AMSgrad optimizer.
 * For more details see the paper
 * [On the Convergence of Adam and Beyond]
 *    https://openreview.net/pdf?id=ryQu7f-RZ).
 */
class AMSgradOptimizer : public FirstOrderOptimizer {
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
  std::vector<af::array> biasedFirst_;
  std::vector<af::array> biasedSecond_;
  std::vector<af::array> maxExpAvgSq_;

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
