/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/nn/PositionalEmbeddingSine.h"

#include <cassert>

#include "flashlight/fl/tensor/Index.h"

namespace fl::pkg::vision {

std::string PositionalEmbeddingSine::prettyString() const {
  return "PositionalEmbeddingSine";
}

PositionalEmbeddingSine::PositionalEmbeddingSine(
    const int numPosFeats,
    const int temperature,
    const bool normalize,
    const float scale)
    : numPosFeats_(numPosFeats),
      temperature_(temperature),
      normalize_(normalize),
      scale_(scale){};

PositionalEmbeddingSine::PositionalEmbeddingSine(
    const PositionalEmbeddingSine& other)
    : numPosFeats_(other.numPosFeats_),
      temperature_(other.temperature_),
      normalize_(other.normalize_),
      scale_(other.scale_) {
  train_ = other.train_;
  for (auto& mod : other.modules_) {
    add(mod->clone());
  }
}

PositionalEmbeddingSine& PositionalEmbeddingSine::operator=(
    const PositionalEmbeddingSine& other) {
  train_ = other.train_;
  numPosFeats_ = other.numPosFeats_;
  temperature_ = other.temperature_;
  normalize_ = other.normalize_;
  scale_ = other.scale_;
  clear();
  for (auto& mod : other.modules_) {
    add(mod->clone());
  }
  return *this;
}

std::unique_ptr<Module> PositionalEmbeddingSine::clone() const {
  return std::make_unique<PositionalEmbeddingSine>(*this);
}

std::vector<Variable> PositionalEmbeddingSine::forward(
    const std::vector<Variable>& inputs) {
  assert(inputs.size() == 1);
  auto input = inputs[0];

  auto inputDims = input.shape();
  // Input mask will be [ w x h x 1 x b ]
  // but implementation expects [ w x h x b ] in order to do interleaves easier
  auto nonMask = fl::reshape(
      input.tensor(), {inputDims[0], inputDims[1], inputDims[3], 1});

  auto expandDims = [](const Tensor& in) {
    auto dims = in.shape();
    assert(dims[3] == 1);
    return fl::reshape(in, {1, dims[0], dims[1], dims[2]});
  };

  auto interleave = [](Tensor x, Tensor y) {
    auto dims = x.shape();
    x = x.flatten();
    y = y.flatten();
    x = fl::reshape(x, {1, x.dim(0)});
    y = fl::reshape(y, {1, y.dim(0)});
    auto joined = fl::concatenate(0, x, y);
    dims[0] = dims[0] * 2;
    return fl::reshape(joined, dims);
  };

  Tensor xEmbed = fl::cumsum(nonMask, 0);
  Tensor yEmbed = fl::cumsum(nonMask, 1);
  if (normalize_) {
    const float eps = 1e-6;
    yEmbed =
        (yEmbed / yEmbed(fl::span, yEmbed.dim(1) - 1, fl::span) + eps) * scale_;
    xEmbed =
        (xEmbed / xEmbed(xEmbed.dim(0) - 1, fl::span, fl::span) + eps) * scale_;
  }

  auto dim = fl::arange({numPosFeats_}, 0, fl::dtype::f32);
  dim = fl::power(temperature_, ((2 * fl::floor(dim / 2)) / numPosFeats_));

  auto posX = expandDims(xEmbed) / dim;
  auto posY = expandDims(yEmbed) / dim;

  auto posXSin = fl::sin(posX(fl::range(0, fl::end, 2), fl::span));
  auto posXCos = fl::cos(posX(fl::range(1, fl::end, 2), fl::span));
  auto posYSin = fl::sin(posY(fl::range(0, fl::end, 2), fl::span));
  auto posYCos = fl::cos(posY(fl::range(1, fl::end, 2), fl::span));

  posX = interleave(posXSin, posXCos);
  posY = interleave(posYSin, posYCos);
  auto result = fl::concatenate(0, posY, posX);
  result = fl::transpose(result, {1, 2, 0, 3});
  return {fl::Variable(result, false)};
}

std::vector<Variable> PositionalEmbeddingSine::operator()(
    const std::vector<Variable>& input) {
  return forward(input);
}

} // namespace fl
