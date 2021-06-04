/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/nn/modules/modules.h"
#include <iostream>

namespace slimIPL {
class myModel : public fl::Container {
public:
  myModel(int64_t nFeature, int64_t nLabel) {
    double dropout = 0.4;
    convFrontend_->add(
        std::make_shared<fl::View>(af::dim4(-1, 1, nFeature, 0)));
    // Time x 1 x nFeature x Batch
    std::vector<int> lnDims = {0, 1, 2};
    convFrontend_->add(std::make_shared<fl::LayerNorm>(lnDims));
    convFrontend_->add(
        std::make_shared<fl::Conv2D>(nFeature, 1536, 7, 1, 3, 1, -1, 0, 1, 1));
    convFrontend_->add(std::make_shared<fl::GatedLinearUnit>(2));
    convFrontend_->add(std::make_shared<fl::Dropout>(0.3));
    convFrontend_->add(std::make_shared<fl::Reorder>(2, 0, 3, 1));
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

  std::vector<fl::Variable>
  forward(const std::vector<fl::Variable> &input) override {
    auto out = input[0];
    auto xSizes = input[1].array();
    // expected input dims T x C x 1 x B
    int T = out.dims(0), B = out.dims(3);
    auto inputMaxSize = af::tile(af::max(xSizes), 1, B);
    af::array inputNotPaddedSize = af::ceil(xSizes * T / inputMaxSize);
    auto padMask = af::iota(af::dim4(T, 1), af::dim4(1, B)) <
                   af::tile(inputNotPaddedSize, T, 1);
    out = convFrontend_->forward(out);
    out = sinpos_->forward({out}, true, 2000, 1.1).front();
    for (int trIdx = 0; trIdx < 36; trIdx++) {
      out = transformers_[trIdx]->forward({out, fl::noGrad(padMask)}).front();
    }
    out = linear_->forward(out);
    return {out.as(input[0].type())};
  }

  std::string prettyString() const override {
    std::ostringstream ss;
    ss << "Model myModel: ";
    for (int trIdx = 0; trIdx < 36; trIdx++) {
      ss << transformers_[trIdx]->prettyString() << "\n";
    }
    ss << linear_->prettyString() << "\n";
    return ss.str();
  }

private:
  myModel() = default;

  std::shared_ptr<fl::Sequential> convFrontend_{
      std::make_shared<fl::Sequential>()};
  std::shared_ptr<fl::SinusoidalPositionEmbedding> sinpos_;
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;
  std::shared_ptr<fl::Linear> linear_;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, convFrontend_, sinpos_, transformers_,
                         linear_)
};
} // namespace slimIPL

extern "C" fl::Module *createModule(int64_t nFeature, int64_t nLabel) {
  auto m = std::make_unique<slimIPL::myModel>(nFeature, nLabel);
  return m.release();
}

CEREAL_REGISTER_TYPE(slimIPL::myModel)
