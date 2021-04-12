/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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
class LmModel : public fl::Container {
 public:
  LmModel(int64_t nLabel) {
    // Time x B x 1 x 1
    std::vector<int> cutoffs = {10000, 50000, (int)nLabel};
    frontend_ = std::make_shared<fl::Sequential>();
    frontend_->add(std::make_shared<fl::AdaptiveEmbedding>(1024, cutoffs));
    // nFeature x Time x B x 1
    frontend_->add(std::make_shared<fl::SinusoidalPositionEmbedding>(1024, 32));
    frontend_->add(std::make_shared<fl::Dropout>(0.1));
    // nFeature x Time x Batch x 1
    add(frontend_);
    for (int trIdx = 0; trIdx < 16; trIdx++) {
      auto layer = std::make_shared<fl::Transformer>(
          1024, 128, 4096, 8, 0, 0.1, 0., true, false);
      transformers_.push_back(layer);
      add(layer);
    }
  }

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override {
    auto out = input[0];
    // Avoid fp16 usage in any embedding-ralated calls.
    out = frontend_->forward(out);

    // Run all transformer forward passes in fp16
    out = out.as(f16);
    for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
      out = transformers_[trIdx]->forward({out, fl::Variable()}).front();
    }

    // Make sure passing fp32 tensor to criterion.
    // Avoid fp16 usage in any embedding-ralated calls.
    return {out.as(f32)};
  }

  std::string prettyString() const override {
    std::ostringstream ss;
    ss << "LmModel: ";
    ss << frontend_->prettyString() << "\n";
    for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
      ss << transformers_[trIdx]->prettyString() << "\n";
    }
    return ss.str();
  }

 private:
  LmModel() = default;

  std::shared_ptr<fl::Sequential> frontend_;
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, frontend_, transformers_)
};

extern "C" fl::Module* createModule(int64_t, int64_t nLabel) {
  auto m = std::make_unique<LmModel>(nLabel);
  return m.release();
}

CEREAL_REGISTER_TYPE(LmModel)
