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

/** An implementation of the RMSProp optimizer. For more details see Geoff
 * Hinton's [lecture slides](
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
 * and https://arxiv.org/pdf/1308.0850v5.pdf.
 */
class RMSPropOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      FirstOrderOptimizer,
      useFirst_,
      fl::serializeAs<double>(rho_),
      fl::serializeAs<double>(eps_),
      fl::serializeAs<double>(wd_),
      first_,
      second_)

  RMSPropOptimizer() = default; // Intentionally private

  bool useFirst_;
  float rho_;
  float eps_;
  float wd_;
  std::vector<af::array> first_;
  std::vector<af::array> second_;

 public:
  /** Construct an RMSProp optimizer.
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learningRate The learning rate.
   * @param rho The weight in the term \f$ rho * m + (1-rho) * g^2 \f$.
   * @param epsilon A small value used for numerical stability.
   * @param weightDecay The amount of L2 weight decay to use for all the
   * parameters.
   * @param use_first Use the first moment in the update. When `true` keep
   * a running mean of the gradient and subtract it from the running mean of
   * the squared gradients.
   */
  RMSPropOptimizer(
      const std::vector<Variable>& parameters,
      float learningRate,
      float rho = 0.99,
      float epsilon = 1e-8,
      float weightDecay = 0,
      bool use_first = false);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::RMSPropOptimizer)
