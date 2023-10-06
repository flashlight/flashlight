/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/optim/Optimizers.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/** Nesterov Accelerated Gradient with modification for the changeable lr
 * through time. Implements the version from
 * https://github.com/pytorch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/fairseq/optim/nag.py#L43
 */
class FL_API NAGOptimizer : public FirstOrderOptimizer {
 private:
  FL_SAVE_LOAD_WITH_BASE(FirstOrderOptimizer, mu_, wd_, velocities_, oldLr_)

  NAGOptimizer() = default; // Intentionally private

  float mu_;
  float wd_;
  std::vector<Tensor> velocities_;
  float oldLr_;

 public:
  /** NAGOptimizer constructor.
   * @param parameters The parameters from e.g. `model.parameters()`
   * @param learningRate The learning rate.
   * @param momentum The momentum.
   * @param weightDecay The amount of L2 weight decay to use for all the
   * parameters.
   */
  NAGOptimizer(
      const std::vector<Variable>& parameters,
      float learningRate,
      float momentum = 0.99,
      float weightDecay = 0);

  void step() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::NAGOptimizer)
