/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/SinusoidalPositionEmbedding2D.h"

#include <math.h>
#include <stdexcept>
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

SinusoidalPositionEmbedding2D::SinusoidalPositionEmbedding2D(
    int32_t layerDim,
    double inputScale /* = 1. */)
    : layerDim_(layerDim), inputScale_(inputScale) {
  int32_t halfDim = layerDim_ / 2;
  if (halfDim % 2 == 1) {
    halfDim++;
  }
  // size is layerDim_ / 2 + 1 x 1 x 1 x 1
  auto componentRange = af::iota(af::dim4(halfDim)) + 1;
  scaleX_ = af::cos(componentRange) *
      af::exp(2 * componentRange * std::log(10) / layerDim_);
  scaleY_ = af::sin(componentRange) *
      af::exp(2 * componentRange * std::log(10) / layerDim_);
  params_.emplace_back(fl::truncNormal(af::dim4(layerDim), 0.02));
}

std::vector<Variable> SinusoidalPositionEmbedding2D::forward(
    const std::vector<Variable>& input,
    bool aug,
    float region,
    bool doShrink) {
  // input  C x HW x B
  if (input[0].dims(0) != layerDim_) {
    throw std::invalid_argument(
        "Input dimenstion " + std::to_string(input[0].dims(0)) +
        " and Embedding dimension " + std::to_string(layerDim_) +
        " are different");
  }
  int B = input[0].dims(2);
  int axisDim = std::sqrt(input[0].dims(1) - 1);
  int axisDimTotal = axisDim * axisDim;
  // 1 2 ... axisDim 1 2 ... axisDim
  auto xPositions = af::moddims(
      af::iota(af::dim4(axisDim, 1), af::dim4(1, axisDim, B)) + 1,
      af::dim4(1, axisDim * axisDim, B));
  // 1 1 ... 1 2 2 ... 2 ... axisDim axisDim ... axisDim
  auto yPositions = af::moddims(
      af::iota(af::dim4(1, axisDim), af::dim4(axisDim, 1, B)) + 1,
      af::dim4(1, axisDim * axisDim, B));
  // norm positions to be [-1, 1]
  float halfAxisDim = (axisDim - 1) / 2.;
  float center = (1 + axisDim) / 2.;
  xPositions = (xPositions - center) / halfAxisDim * region;
  yPositions = (yPositions - center) / halfAxisDim * region;
  // std::cerr << "here before aug" << std::endl;
  if (train_ && aug) {
    // do global shift in [-1, 1]
    xPositions = xPositions +
        af::tile(af::randu(af::dim4(1, 1, B)), 1, axisDimTotal) - 0.5;
    yPositions = yPositions +
        af::tile(af::randu(af::dim4(1, 1, B)), 1, axisDimTotal) - 0.5;
    // std::cerr << "here after global shift" << std::endl;
    // do local shift
    float deltaMax = 2. / axisDim;
    xPositions =
        xPositions + af::randu(xPositions.dims()) * deltaMax - deltaMax / 2.;
    yPositions =
        yPositions + af::randu(yPositions.dims()) * deltaMax - deltaMax / 2.;
    // std::cerr << "here before shrink" << std::endl;
    // shrink
    if (doShrink) {
      auto shrinkScale =
          af::tile(af::randu(af::dim4(1, 1, B)), 1, axisDimTotal) * 0.7 + 0.7;
      xPositions = xPositions * shrinkScale;
      yPositions = yPositions * shrinkScale;
    }
  }
  // std::cerr << "here after aug" << std::endl;
  int32_t halfDim = layerDim_ / 2;
  xPositions = af::tile(xPositions, scaleX_.dims(0));
  yPositions = af::tile(yPositions, scaleY_.dims(0));
  // std::cerr << "here before pos * scale" << std::endl;
  af::array positionsScaledSin =
      (xPositions * af::tile(scaleX_, 1, axisDimTotal, B) +
       yPositions * af::tile(scaleY_, 1, axisDimTotal, B)) *
      M_PI;
  // std::cerr << "here before y pos * scale" << std::endl;
  af::array positionsScaledCos =
      (xPositions * af::tile(scaleX_(af::seq(halfDim)), 1, axisDimTotal, B) +
       yPositions * af::tile(scaleY_(af::seq(halfDim)), 1, axisDimTotal, B)) *
      M_PI;
  // std::cerr << "here before sin/cos" << std::endl;
  Variable sinPos = Variable(af::sin(positionsScaledSin), false);
  Variable cosPos = Variable(af::cos(positionsScaledCos), false);

  // std::cerr << "here concat " << sinPos.dims() << " " << cosPos.dims() <<
  // std::endl;
  Variable embeddingsPos = fl::concatenate({sinPos, cosPos}, 0);
  // std::cerr << "here concat class token" << params_[0].dims() << " " <<
  // embeddingsPos.dims() << std::endl; contcat start token
  embeddingsPos = fl::concatenate(
      {fl::tile(params_[0], af::dim4(1, 1, B)), embeddingsPos}, 1);
  // std::cerr << "return" <<  embeddingsPos.dims() << " "  << input[0].dims()
  // << std::endl;
  return {
      input[0] * inputScale_ +
      tileAs(embeddingsPos, input[0]).as(input[0].type())};
}

std::vector<Variable> SinusoidalPositionEmbedding2D::operator()(
    const std::vector<Variable>& input) {
  return forward(input);
}

std::string SinusoidalPositionEmbedding2D::prettyString() const {
  std::ostringstream ss;
  ss << "Sinusoidal Position Embedding 2D Layer (embDim: " << layerDim_
     << "), (input scale " << inputScale_ << ")";
  return ss.str();
}

SinusoidalPositionEmbedding2D::SinusoidalPositionEmbedding2D() {}

} // namespace fl
