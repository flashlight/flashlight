/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/models/Detr.h"

namespace {

double calculateGain(double negativeSlope) {
  return std::sqrt(2.0 / (1 + std::pow(negativeSlope, 2)));
}

std::shared_ptr<fl::Linear> makeLinear(int inDim, int outDim) {
  int fanIn = inDim;
  float gain = calculateGain(std::sqrt(5.0));
  float std = gain / std::sqrt(fanIn);
  float bound = std::sqrt(3.0) * std;
  auto w = fl::uniform(outDim, inDim, -bound, bound, f32, true);
  bound = 1.0 / std::sqrt(fanIn);
  auto b = fl::uniform(af::dim4(outDim), -bound, bound, af::dtype::f32, true);
  return std::make_shared<fl::Linear>(w, b);
}

std::shared_ptr<fl::Conv2D> makeConv2D(int inDim, int outDim, int wx, int wy) {
  int fanIn = wx * wy * inDim;
  float gain = calculateGain(std::sqrt(5.0f));
  float std = gain / std::sqrt(fanIn);
  float bound = std::sqrt(3.0f) * std;
  auto w = fl::uniform({wx, wy, inDim, outDim}, -bound, bound, f32, true);
  bound = 1.0f / std::sqrt(fanIn);
  auto b = fl::uniform(
      af::dim4(1, 1, outDim, 1), -bound, bound, af::dtype::f32, true);
  return std::make_shared<fl::Conv2D>(w, b, 1, 1);
}

} // namespace

namespace fl {
namespace pkg {
namespace vision {

MLP::MLP(
    const int32_t inputDim,
    const int32_t hiddenDim,
    const int32_t outputDim,
    const int32_t numLayers) {
  add(makeLinear(inputDim, hiddenDim));
  for (int i = 1; i < numLayers - 1; i++) {
    add(ReLU());
    add(makeLinear(hiddenDim, hiddenDim));
  }
  add(ReLU());
  add(makeLinear(hiddenDim, outputDim));
}

Detr::Detr(
    std::shared_ptr<Transformer> transformer,
    std::shared_ptr<Module> backbone,
    const int32_t hiddenDim,
    const int32_t numClasses,
    const int32_t numQueries,
    const bool auxLoss)
    : backbone_(backbone),
      transformer_(transformer),
      classEmbed_(makeLinear(hiddenDim, numClasses + 1)),
      bboxEmbed_(std::make_shared<MLP>(hiddenDim, hiddenDim, 4, 3)),
      queryEmbed_(
          std::make_shared<Embedding>(fl::normal({hiddenDim, numQueries}))),
      posEmbed_(std::make_shared<PositionalEmbeddingSine>(
          hiddenDim / 2,
          10000,
          true,
          6.283185307179586f)),
      inputProj_(makeConv2D(2048, hiddenDim, 1, 1)),
      numClasses_(numClasses),
      numQueries_(numQueries),
      auxLoss_(auxLoss) {
  add(transformer_);
  add(classEmbed_);
  add(bboxEmbed_);
  add(queryEmbed_);
  add(inputProj_);
  add(backbone_);
  add(posEmbed_);
}

std::vector<Variable> Detr::forward(const std::vector<Variable>& input) {
  // input: {input, mask}
  if (input.size() != 2) {
    throw std::invalid_argument(
        "Detr takes 2 Variables as input but gets " +
        std::to_string(input.size()));
  }
  auto feature = forwardBackbone(input.front());
  return forwardTransformer({feature, input[1]});
}

Variable Detr::forwardBackbone(const Variable& input) {
  return backbone_->forward({input})[1];
}

std::vector<Variable> Detr::forwardTransformer(const std::vector<Variable>& input) {
  // input: {feature, mask}
  fl::Variable mask = fl::Variable(
      af::resize(
          input[1].array(),
          input[0].dims(0),
          input[0].dims(1),
          AF_INTERP_NEAREST),
      true);
  auto inputProjection = inputProj_->forward(input[0]);
  auto posEmbed = posEmbed_->forward({mask})[0];
  auto hs = transformer_->forward(
      inputProjection,
      mask.as(inputProjection.type()),
      queryEmbed_->param(0).as(inputProjection.type()),
      posEmbed.as(inputProjection.type()));

  auto outputClasses = classEmbed_->forward(hs[0]);
  auto outputCoord = sigmoid(bboxEmbed_->forward(hs)[0]);

  return {outputClasses, outputCoord};
}

std::string Detr::prettyString() const {
  std::ostringstream ss;
  ss << "Detection Transformer";
  ss << Container::prettyString();
  return ss.str();
}

std::vector<fl::Variable> Detr::paramsWithoutBackbone() {
  std::vector<fl::Variable> results;
  std::vector<std::vector<fl::Variable>> childParams;
  childParams.push_back(transformer_->params());
  childParams.push_back(classEmbed_->params());
  childParams.push_back(bboxEmbed_->params());
  childParams.push_back(queryEmbed_->params());
  childParams.push_back(inputProj_->params());
  for (auto params : childParams) {
    results.insert(results.end(), params.begin(), params.end());
  }
  return results;
}

std::vector<fl::Variable> Detr::backboneParams() {
  return backbone_->params();
}

} // namespace vision
} // namespace pkg
} // namespace fl
