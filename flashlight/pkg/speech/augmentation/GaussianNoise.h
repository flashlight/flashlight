/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/augmentation/SoundEffect.h"

#include <random>
#include <string>
#include <vector>

#include "flashlight/pkg/speech/augmentation/SoundEffectUtil.h"

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

/**
 * Add gaussian noise to the samples with given Signal to Noise Ratio (SNR)
 */
class GaussianNoise : public SoundEffect {
 public:
  struct Config {
    float proba_ = 1.0;
    double minSnr_ = 0;
    double maxSnr_ = 30;
    std::string prettyString() const;
  };

  explicit GaussianNoise(
      const GaussianNoise::Config& config,
      unsigned int seed = 0);
  ~GaussianNoise() override = default;
  void apply(std::vector<float>& signal) override;
  std::string prettyString() const override;

 private:
  const GaussianNoise::Config conf_;
  RandomNumberGenerator rng_;
};

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
