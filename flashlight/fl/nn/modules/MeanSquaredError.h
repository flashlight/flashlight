/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Computes the [mean squared
 error](https://en.wikipedia.org/wiki/Mean_squared_error) between elements
 * across two tensors:
 * \f[
   \mathcal{L}(x, y) = \frac{1}{n} \sum_{i = 0}^n \left( x_i - y_i \right)^2
   \f]
 * for input tensor \f$x\f$ and target tensor \f$y\f$ each of which contain
 \f$n\f$ elements.
 */
class MeanSquaredError : public BinaryModule {
 public:
  MeanSquaredError() = default;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(BinaryModule)
};

typedef MeanSquaredError MSE;

} // namespace fl

CEREAL_REGISTER_TYPE(fl::MeanSquaredError)
