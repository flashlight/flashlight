/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Container.h"
#include "flashlight/pkg/vision/models/ResnetFrozenBatchNorm.h"
#include "flashlight/pkg/vision/nn/FrozenBatchNorm.h"

namespace fl {
namespace pkg {
namespace vision {

using namespace fl::pkg::vision;

class Resnet50Backbone : public Container {
 public:
  Resnet50Backbone();

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::string prettyString() const override;

  FL_BASIC_CONTAINER_CLONING(Resnet50Backbone)

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

} // namespace vision
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::vision::Resnet50Backbone)
