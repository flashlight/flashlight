/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/vision/nn/PositionalEmbeddingSine.h"
#include "flashlight/pkg/vision/nn/Transformer.h"

namespace fl {
namespace app {
namespace objdet {

// TODO (padentomasello) this can just be a function
class MLP : public Sequential {
 public:
  MLP(const int32_t inputDim,
      const int32_t hiddenDim,
      const int32_t outputDim,
      const int32_t numLayers);

 private:
  MLP() = default;
  FL_SAVE_LOAD_WITH_BASE(fl::Sequential)
};

class Detr : public Container {
 public:
  Detr(
      std::shared_ptr<Transformer> transformer,
      std::shared_ptr<Module> backbone,
      const int32_t hiddenDim,
      const int32_t numClasses,
      const int32_t numQueries,
      const bool auxLoss);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;
  Variable forwardBackbone(const Variable& input);
  std::vector<Variable> forwardTransformer(const std::vector<Variable>& input);

  std::string prettyString() const override;

  std::vector<fl::Variable> paramsWithoutBackbone();

  std::vector<fl::Variable> backboneParams();

 private:
  Detr() = default;
  std::shared_ptr<Module> backbone_;
  std::shared_ptr<Transformer> transformer_;
  std::shared_ptr<Linear> classEmbed_;
  std::shared_ptr<MLP> bboxEmbed_;
  std::shared_ptr<Embedding> queryEmbed_;
  std::shared_ptr<PositionalEmbeddingSine> posEmbed_;
  std::shared_ptr<Conv2D> inputProj_;
  int32_t hiddenDim_;
  int32_t numClasses_;
  int32_t numQueries_;
  bool auxLoss_;
  FL_SAVE_LOAD_WITH_BASE(
      fl::Container,
      backbone_,
      transformer_,
      classEmbed_,
      bboxEmbed_,
      queryEmbed_,
      posEmbed_,
      inputProj_)
};

} // namespace objdet
} // namespace app
} // namespace fl
CEREAL_REGISTER_TYPE(fl::app::objdet::Detr)
CEREAL_REGISTER_TYPE(fl::app::objdet::MLP)
