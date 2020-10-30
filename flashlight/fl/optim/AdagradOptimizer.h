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

/** An implementation of the Adagrad optimizer.
 * For more details see the paper
 * [Adaptive Subgradient Methods for Online Learning and Stochastic
 * Optimization](
 *    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
 */
class AdagradOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(FirstOrderOptimizer, eps_, wd_, variance_)

  AdagradOptimizer() = default; // Intentionally private

  float eps_;
  float wd_;
  std::vector<af::array>
      variance_; // store sum_{tau=0}^{tau=t} grad_tau*grad_tau

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
