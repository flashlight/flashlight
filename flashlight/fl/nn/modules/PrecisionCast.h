/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Serialization.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Precision Cast Module. Casts the input from its original precision to the
 * target precision. Precision cast only alters the underlying array and leaves
 * other attributes of the input variable unchanged.
 */
class FL_API PrecisionCast : public Module {
 private:
  fl::dtype targetType_;
  PrecisionCast() = default;
  FL_SAVE_LOAD_WITH_BASE(Module, targetType_)

 public:
  /**
   * Constructor of the Cast Module (PrecisionCast class).
   *
   * @param targetType An ArrayFire type that specifies the target type of the
   * cast. Inputs to the the `forward` function will be casted to `targetType`.
   */
  explicit PrecisionCast(fl::dtype targetType);

  /**
   * Casts every input variable according to the `targetType_`. The value of
   * `targetType_` is set during the initialization of the module.
   *
   * @param inputs A reference to the vector containing the input variables.
   *
   * @return A vector that contains the casted variables.
   */
  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;

  Variable forward(const Variable& input);
  Variable operator()(const Variable& input);
  std::unique_ptr<Module> clone() const override;
  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::PrecisionCast)
