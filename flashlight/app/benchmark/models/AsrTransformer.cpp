/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/benchmark/models/AsrTransformer.h"

namespace fl::app::benchmark {

AsrTransformer::AsrTransformer(int64_t nFeature, int64_t nLabel) {
  double dropout = 0.4;
  convFrontend_->add(std::make_shared<fl::View>(Shape({-1, 1, nFeature, 0})));
  // Time x 1 x nFeature x Batch
  std::vector<int> lnDims = {0, 1, 2};
  convFrontend_->add(std::make_shared<fl::LayerNorm>(lnDims));
  convFrontend_->add(
      std::make_shared<fl::Conv2D>(nFeature, 1536, 7, 1, 3, 1, -1, 0, 1, 1));
  convFrontend_->add(std::make_shared<fl::GatedLinearUnit>(2));
  convFrontend_->add(std::make_shared<fl::Dropout>(0.3));
  convFrontend_->add(std::make_shared<fl::Reorder>(Shape({2, 0, 3, 1})));
  // nFeature x Time x Batch x 1
  add(convFrontend_);
  sinpos_ = std::make_shared<fl::SinusoidalPositionEmbedding>(768, 1.0);
  add(sinpos_);
  for (int trIdx = 0; trIdx < 36; trIdx++) {
    auto layer = std::make_shared<fl::Transformer>(
        768, 192, 3072, 4, 0, dropout, dropout, false, false);
    transformers_.push_back(layer);
    add(layer);
  }
  linear_ = std::make_shared<fl::Linear>(768, nLabel);
  add(linear_);
}

AsrTransformer::AsrTransformer(const AsrTransformer& other) {
  convFrontend_ = std::make_shared<fl::Sequential>(*other.convFrontend_);
  // nFeature x Time x Batch x 1
  add(convFrontend_);
  sinpos_ = std::make_shared<fl::SinusoidalPositionEmbedding>(*other.sinpos_);
  add(sinpos_);
  for (const auto& transformer : transformers_) {
    auto layer = std::make_shared<fl::Transformer>(*transformer);
    transformers_.push_back(layer);
    add(std::move(layer));
  }
  linear_ = std::make_shared<fl::Linear>(*other.linear_);
  add(linear_);
}

AsrTransformer& AsrTransformer::operator=(const AsrTransformer& other) {
  convFrontend_ = std::make_shared<fl::Sequential>(*other.convFrontend_);
  // nFeature x Time x Batch x 1
  add(convFrontend_);
  sinpos_ = std::make_shared<fl::SinusoidalPositionEmbedding>(*other.sinpos_);
  add(sinpos_);
  for (const auto& transformer : transformers_) {
    auto layer = std::make_shared<fl::Transformer>(*transformer);
    transformers_.push_back(layer);
    add(std::move(layer));
  }
  linear_ = std::make_shared<fl::Linear>(*other.linear_);
  add(linear_);
  return *this;
}

std::unique_ptr<Module> AsrTransformer::clone() const {
  return std::make_unique<AsrTransformer>(*this);
}

std::vector<fl::Variable> AsrTransformer::forward(
    const std::vector<fl::Variable>& input) {
  auto out = input[0];
  auto xSizes = input[1].tensor();
  // expected input dims T x C x 1 x B
  int T = out.dim(0), B = out.dim(3);
  // TODO{fl::Tensor} - check first non-singleton dimension
  auto inputMaxSize = fl::tile(fl::amax(xSizes, {0}), {1, B});
  Tensor inputNotPaddedSize = fl::ceil(xSizes * T / inputMaxSize);
  auto padMask =
      fl::iota({T, 1}, {1, B}) < fl::tile(inputNotPaddedSize, {T, 1});
  out = convFrontend_->forward(out);
  out = sinpos_->forward({out}).front();
  for (int trIdx = 0; trIdx < 36; trIdx++) {
    out = transformers_[trIdx]->forward({out, fl::noGrad(padMask)}).front();
  }
  out = linear_->forward(out);
  return {out.astype(input[0].type())};
}

std::string AsrTransformer::prettyString() const {
  std::ostringstream ss;
  ss << "Model myModel: ";
  for (int trIdx = 0; trIdx < 36; trIdx++) {
    ss << transformers_[trIdx]->prettyString() << "\n";
  }
  ss << linear_->prettyString() << "\n";
  return ss.str();
}

} // namespace fl
