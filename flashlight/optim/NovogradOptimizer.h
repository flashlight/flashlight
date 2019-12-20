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

/** An implementation of the Novograd optimizer.
 * For more details see the paper
 * [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of
 * Deep Networks](https://arxiv.org/abs/1905.11286).
 */
class NovogradOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      FirstOrderOptimizer,
      beta1_,
      beta2_,
      eps_,
      wd_,
      accGradNorm_,
      accGrad_)

  NovogradOptimizer() = default; // Intentionally private

  double beta1_;
  double beta2_;
  double eps_;
  double wd_;
  int count_;
  std::vector<af::array> accGradNorm_;
  std::vector<af::array> accGrad_;

 public:
  /** Construct a Novograd optimizer
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learningRate The learning rate.
   * @param beta1 Novograd hyperparameter \f$ \beta_1 \f$.
   * @param beta2 Novograd hyperparameter \f$ \beta_2 \f$.
   * @param epsilon A small value used for numerical stability.
   * @param weightDecay The amount of L2 weight decay to use for all the
   * parameters.
   */
  explicit NovogradOptimizer(
      const std::vector<Variable>& parameters,
      double learningRate,
      double beta1 = 0.95,
      double beta2 = 0.98,
      double epsilon = 1e-8,
      double weightDecay = 0);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::NovogradOptimizer)
