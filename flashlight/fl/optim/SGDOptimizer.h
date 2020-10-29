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

#include "flashlight/fl/optim/Optimizers.h"

namespace fl {

/** A Stochastic Gradient Descent (SGD) optimizer. At its most basic this
 * implements the update
 * \f[
 *   w = w - lr * g
 * \f]
 *
 * When momentum is used the update becomes
 * \f[
 *   v &= \rho * v + g \\
 *   w &= w - lr * v
 * \f]
 *
 * Reference for SGD and Momentum:
 * http://cs231n.github.io/neural-networks-3/#sgd
 */
class SGDOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      FirstOrderOptimizer,
      useNesterov_,
      fl::serializeAs<double>(mu_),
      fl::serializeAs<double>(wd_),
      velocities_)

  SGDOptimizer() = default; // Intentionally private

  bool useNesterov_;
  float mu_;
  float wd_;
  std::vector<af::array> velocities_;

 public:
  /** SGDOptimizer constructor.
   * @param parameters The parameters from e.g. `model.parameters()`
   * @param learningRate The learning rate.
   * @param momentum The momentum.
   * @param weightDecay The amount of L2 weight decay to use for all the
   * parameters.
   * @param useNesterov Whether or not to use nesterov style momentum.
   */
  SGDOptimizer(
      const std::vector<Variable>& parameters,
      float learningRate,
      float momentum = 0,
      float weightDecay = 0,
      bool useNesterov = false);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::SGDOptimizer)
