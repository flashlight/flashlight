/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

class FL_API Normalize : public UnaryModule {
 public:
  /**
   * Constructs a Normalize module.
   *
   * @param value the target normalization value.
   * @param axes reduce over specified axes
   * @param p as p in  Lp norm
   * @param eps min clamping value to avoid overflows
   * @param normalization mode, as supported by normalize()
   */
  explicit Normalize(
      const std::vector<int>& axes,
      double p = 2,
      double eps = 1e-12,
      double value = 1);

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;

 private:
  Normalize() = default;

  std::vector<int> axes_;
  double p_;
  double eps_;
  double value_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, axes_, p_, eps_, value_)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Normalize)
