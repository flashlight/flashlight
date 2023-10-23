/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/types/utility.hpp>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Adds a padding of value `val` before and after each dimension
 * \f$i\f$ of size specified by the tuple `padi` to the input.
 */
class FL_API Padding : public UnaryModule {
 private:
  Padding() = default; // intentionally private

  std::vector<std::pair<int, int>> m_pad;
  double m_val;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, m_pad, m_val)

 public:
  /**
   * Constructs a Padding module that pads the first dimension of the input. If
   * the input is of shape
   * [\f$dim_0\f$, \f$dim_1\f$, \f$dim_2\f$, \f$dim_3\f$],
   * the output will be of shape [\f$paddingBefore+dim_0+paddingAfter\f$,
   * \f$dim_1\f$, \f$dim_2\f$, \f$dim_3\f$]
   * @param[in] padding a vector of tuples representing padding (before,
   * after) tuples for each axis
   * @param val the value to be padded
   */
  Padding(std::vector<std::pair<int, int>> padding, double val);

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Padding)
