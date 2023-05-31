/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the MIT-style license found in the
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
  Resnet50Backbone(const Resnet50Backbone& other);
  Resnet50Backbone& operator=(const Resnet50Backbone& other);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::unique_ptr<Module> clone() const override;
  std::string prettyString() const override;

 private:
  std::shared_ptr<Sequential> backbone_;
  std::shared_ptr<Sequential> tail_;
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

} // namespace vision
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::vision::Resnet50Backbone)
