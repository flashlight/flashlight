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

/** An implementation of the Adam optimizer.
 * For more details see the paper
 * [Adam: A Method for Stochastic Optimization](
 *    https://arxiv.org/abs/1412.6980).
 */
class FL_API AdamOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      FirstOrderOptimizer,
      fl::serializeAs<double>(beta1_),
      fl::serializeAs<double>(beta2_),
      fl::serializeAs<double>(eps_),
      fl::serializeAs<double>(wd_),
      count_,
      biasedFirst_,
      biasedSecond_)

  AdamOptimizer() = default; // Intentionally private

  float beta1_;
  float beta2_;
  float eps_;
  float wd_;
  int count_;
  std::vector<Tensor> biasedFirst_;
  std::vector<Tensor> biasedSecond_;

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
      float learningRate,
      float beta1 = 0.9,
      float beta2 = 0.999,
      float epsilon = 1e-8,
      float weightDecay = 0);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AdamOptimizer)
