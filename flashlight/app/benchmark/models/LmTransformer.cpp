/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/benchmark/models/LmTransformer.h"

namespace fl {
namespace app {
namespace benchmark {

LmTransformer::LmTransformer(int64_t nLabel, bool fp16) : fp16_(fp16) {
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

std::vector<fl::Variable> LmTransformer::forward(
    const std::vector<fl::Variable>& input) {
  auto out = input[0];
  // Avoid fp16 usage in any embedding-ralated calls.
  out = frontend_->forward(out);

  if (fp16_) {
    // Run all transformer forward passes in fp16
    out = out.as(f16);
  }
  for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
    out = transformers_[trIdx]->forward({out, fl::Variable()}).front();
  }

  // Make sure passing fp32 tensor to criterion.
  // Avoid fp16 usage in any embedding-ralated calls.
  return {out.as(f32)};
}

std::string LmTransformer::prettyString() const {
  std::ostringstream ss;
  ss << "LmModel: ";
  ss << frontend_->prettyString() << "\n";
  for (int trIdx = 0; trIdx < transformers_.size(); trIdx++) {
    ss << transformers_[trIdx]->prettyString() << "\n";
  }
  return ss.str();
}

} // namespace benchmark
} // namespace app
} // namespace fl
