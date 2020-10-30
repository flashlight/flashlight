/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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
class Padding : public UnaryModule {
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
   * @param pad0 the size of the padding
   *  (\f$paddingBefore\f$, \f$paddingAfter\f$)
   * @param val the value to be padded
   */
  Padding(std::pair<int, int> pad0, double val);

  /**
   * Constructs a Padding module that pads the first and second dimensions
   * of the input. If the input is of shape
   * [\f$dim_0\f$, \f$dim_1\f$, \f$dim_2\f$,
   * \f$dim_3\f$], the output will be of shape
   * [\f$paddingBefore_0+dim_0+paddingAfter_0\f$,
   * \f$paddingBefore_1+dim_1+paddingAfter_1\f$, \f$dim_2\f$, \f$dim_3\f$]
   * @param pad0 the size of the padding for the first dimension
   *  (\f$paddingBefore_0\f$, \f$paddingAfter_0\f$)
   * @param pad1 the size of the padding for the second dimension
   *  (\f$paddingBefore_1\f$, \f$paddingAfter_1\f$)
   * @param val the value to be padded
   */
  Padding(std::pair<int, int> pad0, std::pair<int, int> pad1, double val);

  /**
   * Constructs a Padding module that pads the first three dimensions
   * of the input. If the input is of shape
   * [\f$dim_0\f$, \f$dim_1\f$, \f$dim_2\f$,
   * \f$dim_3\f$], the output will be of shape
   * [\f$paddingBefore_0+dim_0+paddingAfter_0\f$,
   * \f$paddingBefore_1+dim_1+paddingAfter_1\f$,
   * \f$paddingBefore_2+dim_2+paddingAfter_2\f$, \f$dim_3\f$]
   * @param pad0 the size of the padding for the first dimension
   *  (\f$paddingBefore_0\f$, \f$paddingAfter_0\f$)
   * @param pad1 the size of the padding for the second dimension
   *  (\f$paddingBefore_1\f$, \f$paddingAfter_1\f$)
   * @param pad2 the size of the padding for the third dimension
   *  (\f$paddingBefore_2\f$, \f$paddingAfter_2\f$)
   * @param val the value to be padded
   */
  Padding(
      std::pair<int, int> pad0,
      std::pair<int, int> pad1,
      std::pair<int, int> pad2,
      double val);

  /**
   * Constructs a Padding module that pads all four dimensions
   * of the input. If the input is of shape
   * [\f$dim_0\f$, \f$dim_1\f$, \f$dim_2\f$,
   * \f$dim_3\f$], the output will be of shape
   * [\f$paddingBefore_0+dim_0+paddingAfter_0\f$,
   * \f$paddingBefore_1+dim_1+paddingAfter_1\f$,
   * \f$paddingBefore_2+dim_2+paddingAfter_2\f$,
   * \f$paddingBefore_3+dim_3+paddingAfter_3\f$]
   * @param pad0 the size of the padding for the first dimension
   *  (\f$paddingBefore_0\f$, \f$paddingAfter_0\f$)
   * @param pad1 the size of the padding for the second dimension
   *  (\f$paddingBefore_1\f$, \f$paddingAfter_1\f$)
   * @param pad2 the size of the padding for the third dimension
   *  (\f$paddingBefore_2\f$, \f$paddingAfter_2\f$)
   * @param pad3 the size of the padding for the third dimension
   *  (\f$paddingBefore_3\f$, \f$paddingAfter_3\f$)
   * @param val the value to be padded
   */
  Padding(
      std::pair<int, int> pad0,
      std::pair<int, int> pad1,
      std::pair<int, int> pad2,
      std::pair<int, int> pad3,
      double val);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Padding)
