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
 * Identity returns the inputs at forward.
 */
class Identity : public Module {
 public:
  Identity() = default;
  virtual std::vector<Variable> forward(
      const std::vector<Variable>& inputs) override;
  virtual std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};
} // namespace fl

CEREAL_REGISTER_TYPE(fl::Identity)
