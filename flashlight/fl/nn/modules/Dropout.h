/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Implements Dropout normalization, as given by [Hinton et al
 * (2012)](https://arxiv.org/abs/1207.0580): _Improving neural networks by
 * preventing co-adaptation of feature detectors_. Effectively regularizes by
 * randomly zeroing out values in the input based on a given ratio.
 *
 * All values that are not zeroed out are scaled by a factor of
 * \f$\frac{1}{1 - p}\f$. Thus, with the same network, at test time,
 * evaluating the module gives the identity.
 */
class FL_API Dropout : public UnaryModule {
 private:
  double ratio_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, ratio_)

 public:
  /**
   * Creates a `Dropout` layer.
   *
   * @param drop_ratio the probability that a weight will be set to zero
   */
  Dropout(double drop_ratio = 0.5);

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Dropout)
