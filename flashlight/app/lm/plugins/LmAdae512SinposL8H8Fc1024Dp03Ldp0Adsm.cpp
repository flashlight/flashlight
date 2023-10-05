/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/nn/modules/modules.h"

/**
 * This is example of plugin for language model architecture which is expected
 * the input with size Time x Batch x 1 x 1 and used with the adaptive softmax
 * as criterion (so the last linear layer is absent here)
 * This architecture is using also adaptive embedding and sinusoidal positional
 * embedding.
 */
class LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm : public fl::Container {
 public:
  LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm(int64_t nLabel) {
    // Time x B x 1 x 1
    std::vector<int> cutoffs = {10000, 50000, (int)nLabel};
    frontend_ = std::make_shared<fl::Sequential>();
    frontend_->add(std::make_shared<fl::AdaptiveEmbedding>(512, cutoffs));
    // nFeature x Time x B x 1
    frontend_->add(
        std::make_shared<fl::SinusoidalPositionEmbedding>(512, 22.63));
    frontend_->add(std::make_shared<fl::Dropout>(0.3));
    // nFeature x Time x Batch x 1
    add(frontend_);
    for (int trIdx = 0; trIdx < 8; trIdx++) {
      auto layer = std::make_shared<fl::Transformer>(
          512, 64, 1024, 8, 0, 0.3, 0., true, false);
      transformers_.push_back(layer);
      add(layer);
    }
  }

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override {
    auto out = input[0];
    auto xSizes = input[1].tensor();
    // expected input dims T x B x 1 x 1
    int T = out.dim(0), B = out.dim(1);
    // TODO{fl::Tensor} - first non-signleton dimension check
    auto inputMaxSize = fl::tile(fl::amax(xSizes, {0}), {1, B});
    Tensor inputNotPaddedSize = fl::ceil(xSizes * T / inputMaxSize);
    auto padMask =
        fl::iota({T, 1}, {1, B}) < fl::tile(inputNotPaddedSize, {T, 1});
    out = frontend_->forward(out);
    for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
      out = transformers_[trIdx]->forward({out, fl::noGrad(padMask)}).front();
    }
    return {out};
  }

  std::string prettyString() const override {
    std::ostringstream ss;
    ss << "Model LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm: ";
    ss << frontend_->prettyString() << "\n";
    for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
      ss << transformers_[trIdx]->prettyString() << "\n";
    }
    return ss.str();
  }

 private:
  LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm() = default;

  std::shared_ptr<fl::Sequential> frontend_;
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, frontend_, transformers_)
};
} // namespace rasrLM

extern "C" fl::Module* createModule(int64_t, int64_t nLabel) {
  auto m = std::make_unique<LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm>(nLabel);
  return m.release();
}

CEREAL_REGISTER_TYPE(LmAdae512SinposL8H8Fc1024Dp03Ldp0Adsm)
