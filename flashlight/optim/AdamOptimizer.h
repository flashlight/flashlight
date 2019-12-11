/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
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

#include "flashlight/autograd/Variable.h"
#include "flashlight/optim/Optimizers.h"

namespace fl {

/** An implementation of the Adam optimizer.
 * For more details see the paper
 * [Adam: A Method for Stochastic Optimization](
 *    https://arxiv.org/abs/1412.6980).
 */
class AdamOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      FirstOrderOptimizer,
      beta1_,
      beta2_,
      eps_,
      wd_,
      count_,
      biasedFirst_,
      biasedSecond_)

  AdamOptimizer() = default; // Intentionally private

  double beta1_;
  double beta2_;
  double eps_;
  double wd_;
  int count_;
  std::vector<af::array> biasedFirst_;
  std::vector<af::array> biasedSecond_;

 public:
  /** Construct an Adam optimizer.
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learningRate The learning rate.
   * @param beta1 Adam hyperparameter \f$ \beta_1 \f$.
   * @param beta2 Adam hyperparameter \f$ \beta_2 \f$.
   * @param epsilon A small value used for numerical stability.
   * @param weightDecay The amount of L2 weight decay to use for all the
   * parameters.
   */
  AdamOptimizer(
      const std::vector<Variable>& parameters,
      double learningRate,
      double beta1 = 0.9,
      double beta2 = 0.999,
      double epsilon = 1e-8,
      double weightDecay = 0);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AdamOptimizer)
