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
 * Identity returns the inputs at forward.
 */
class FL_API Identity : public Module {
 public:
  Identity() = default;
  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;
  std::unique_ptr<Module> clone() const override;
  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Identity)
