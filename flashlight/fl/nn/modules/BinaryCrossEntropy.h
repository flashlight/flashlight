
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
 * Computes the binary cross entropy loss between an input tensor \f$x\f$ and a
 * target tensor \f$y\f$. The binary cross entropy loss is:
 * \f[
   B(x, y) = \frac{1}{n} \sum_{i = 0}^n -\left( w_i \times (y_i \times \log(x_i)
   + (1 - y_i) \times \log(1 - x_i)) \right) \f]
 * where \f$w\f$ is an optional weight parameter for rescaling.
 *
 * Both the inputs and the targets are expected to be between 0 and 1.
 */
class BinaryCrossEntropy : public BinaryModule {
 public:
  BinaryCrossEntropy() = default;

  using BinaryModule::forward;

  Variable forward(const Variable& inputs, const Variable& targets) override;

  /**
   * Perform forward loss computation with an additional weight tensor.
   *
   * @param inputs a tensor with the predicted values
   * @param targets a tensor with the target values
   * @param weights a rescaling weight given to the loss of each element.
   */
  Variable forward(
      const Variable& inputs,
      const Variable& targets,
      const Variable& weights);

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(BinaryModule)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::BinaryCrossEntropy)
