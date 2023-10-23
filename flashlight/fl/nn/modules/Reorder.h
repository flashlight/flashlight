/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/** Reorders the data according to the specified dimensions. The order of the
 * data may change and is guaranteed to be contiguous in memory.
 * \code
   // A layer which transposes a matrix
   auto transposeLayer = Reorder(1, 0);

   auto var = Variable(Tensor({1, 2, 3, 4}), false);

   // Make the last dimension the first dimension
   var = Reorder(3, 0, 1, 2)(var);
   // Dims will be {4, 1, 2, 3}
   std::cout << var.shape() << std::endl;
 * \endcode
 */
class FL_API Reorder : public UnaryModule {
 private:
  Reorder() = default;

  Shape shape_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, shape_)

 public:
  /**
   * Construct a Reorder layer. The dimension values must not repeat and must
   * be between 0 and 3 inclusive.
   *
   * @param shape The shape to which the input will be reshaped.
   */
  explicit Reorder(Shape shape);

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Reorder)
