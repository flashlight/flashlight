/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Modifies the dimensions of a `Variable` and rearranges its elements without
 * modifying the order of elements in the underlying `af::array`. When
 * specifying the number of elements in the array:
 * - If `-1` is specified on a particular axis, that axis will be assigned a
 * dimension based on the number of total elements in the tensor. Only one axis
 * value can be `-1`.
 * - If `0` is specified on a particular axis, that axis will have the same
 * dimension as does the input tensor. For example: given an input tensor with
 * shape `(10, 20, 30, 40)` and a `View` with shape `(-1, 0, 100)`, the output
 * tensor will have shape `(120, 20, 100)`.
 */
class View : public UnaryModule {
 private:
  View() = default; // Intentionally private

  af::dim4 dims_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, dims_)

 public:
  /**
   * Creates a `View` with the given dimensions.
   *
   * @param dims an `af::dim4` representing the dimensions of the `View`.
   */
  explicit View(af::dim4 dims);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

  ~View() = default;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::View)
