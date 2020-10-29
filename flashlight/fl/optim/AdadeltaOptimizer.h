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

/** An implementation of the Adadelta optimizer.
 * For more details see the paper
 * [Adadelta: An Adaptive Learning Rate Method](
 *    https://arxiv.org/pdf/1212.5701.pdf).
 */
class AdadeltaOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      FirstOrderOptimizer,
      fl::serializeAs<double>(rho_),
      fl::serializeAs<double>(eps_),
      fl::serializeAs<double>(wd_),
      accGrad_,
      accDelta_)

  AdadeltaOptimizer() = default; // Intentionally private

  float rho_;
  float eps_;
  float wd_;
  std::vector<af::array> accGrad_;
  std::vector<af::array> accDelta_;

 public:
  /** Construct an Adadelta optimizer.
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learningRate The learning rate for scaling delta. The original
   * paper does not include this term (i.e. learningRate = 1.0).
   * @param rho The decay rate for accumulating squared gradients and deltas.
   * @param epsilon A small value used for numerical stability.
   * @param weightDecay The amount of L2 weight decay to use for all the
   * parameters.
   */
  explicit AdadeltaOptimizer(
      const std::vector<Variable>& parameters,
      float learningRate = 1.0,
      float rho = 0.9,
      float epsilon = 1e-8,
      float weightDecay = 0);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AdadeltaOptimizer)
