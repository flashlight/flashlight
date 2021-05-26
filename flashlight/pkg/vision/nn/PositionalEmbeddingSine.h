/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"
#include "flashlight/fl/nn/nn.h"

namespace fl {
namespace pkg {
namespace vision {

class PositionalEmbeddingSine : public Container {
 public:
  PositionalEmbeddingSine(
      const int numPosFeats,
      const int temperature,
      const bool normalize,
      const float scale);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::vector<Variable> operator()(const std::vector<Variable>& input);

  std::string prettyString() const override;

 private:
  PositionalEmbeddingSine() = default;
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      numPosFeats_,
      temperature_,
      normalize_,
      scale_)
  int numPosFeats_;
  int temperature_;
  bool normalize_;
  float scale_;
};

} // namespace vision
} // namespace pkg
} // namespace fl
CEREAL_REGISTER_TYPE(fl::pkg::vision::PositionalEmbeddingSine)
