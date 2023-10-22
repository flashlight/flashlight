/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/nn/modules/modules.h"

namespace fl {
namespace app {
namespace benchmark {

/**
 * This is a typical [RASR Transformer](https://arxiv.org/abs/2010.11745) model
 * designed for speech recognition. We use CTC criterion on top of it in this
 * benchmark.
 */
class AsrTransformer : public fl::Container {
 public:
  AsrTransformer(int64_t nFeature, int64_t nLabel);
  AsrTransformer(const AsrTransformer& other);
  AsrTransformer& operator=(const AsrTransformer& other);
  std::unique_ptr<Module> clone() const override;

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override;

  std::string prettyString() const override;

 private:
  AsrTransformer() = default;

  std::shared_ptr<fl::Sequential> convFrontend_{
      std::make_shared<fl::Sequential>()};
  std::shared_ptr<fl::SinusoidalPositionEmbedding> sinpos_;
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;
  std::shared_ptr<fl::Linear> linear_;
};

} // namespace benchmark
} // namespace app
} // namespace fl
