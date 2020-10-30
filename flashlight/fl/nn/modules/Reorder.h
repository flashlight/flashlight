/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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

   auto var = Variable(af::array(1, 2, 3, 4), false);

   // Make the last dimension the first dimension
   var = Reorder(3, 0, 1, 2)(var);
   // Dims will be {4, 1, 2, 3}
   std::cout << var.dims() << std::endl;
 * \endcode
 */
class Reorder : public UnaryModule {
 private:
  Reorder() = default;

  int dim0_, dim1_, dim2_, dim3_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, dim0_, dim1_, dim2_, dim3_)

 public:
  /** Construct a Reorder layer. The dimension values must not repeat and must
   * be between 0 and 3 inclusive.
   * @param dim0 The dimension of the input which to becomes the new first
   * dimension
   * @param dim1 The dimension of the input which to becomes the new second
   * dimension
   * @param dim2 The dimension of the input which to becomes the new third
   * dimension
   * @param dim3 The dimension of the input which to becomes the new fourth
   * dimension
   */
  Reorder(int dim0, int dim1, int dim2 = 2, int dim3 = 3);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Reorder)
