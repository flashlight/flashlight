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
 * This is a typical [Transformer Language
 * Model](https://arxiv.org/abs/1809.10853) with adaptive embedding. We use
 * [adaptive softmax](https://arxiv.org/abs/1609.04309) criterion on top of it
 * in this benchmark.
 */
class LmTransformer : public fl::Container {
 public:
  explicit LmTransformer(int64_t nLabel, bool fp16 = false);
  LmTransformer(const LmTransformer& other);
  LmTransformer& operator=(const LmTransformer& other);
  std::unique_ptr<Module> clone() const override;

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override;

  std::string prettyString() const override;

 private:
  LmTransformer() = default;

  bool fp16_;

  std::shared_ptr<fl::Sequential> frontend_;
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;
};

} // namespace benchmark
} // namespace app
} // namespace fl
