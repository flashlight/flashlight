/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/augmentation/SoundEffect.h"

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {
/**
 * The additive noise sound effect loads noise files and augments them to the
 * signal with hyper parameters that are chosen randomly within a configured
 * range, including:
 * - number of noise clips that are augmented on the input.
 * - noise shift. The noise clip is shifted with a random value. This means that
 * we choose a random index of the noise and start there when adding it to
 * input. We tile the noise to cover the augmentation interval if it is too
 * short to do so.
 * - augmentation interval location. When ratio < 1, an interval of that ratio
 * of the input is augmented.
 * - SNR: the noise is added with random SNR. In order to minimize change to the
 * input we use the following formula. output = input + noise *
 * rms(signal)/rms(noise) / snrDB. rms(signal) is calculated only on the
 * augmented interval. rms(noise) is calculated on the sum of all noise clipse
 * over the augmented interval.
 */
class AdditiveNoise : public SoundEffect {
 public:
  struct Config {
    double ratio = 1.0;
    double minSnr_ = 5;
    double maxSnr_ = 30;
    int nClipsMin_ = 0;
    int nClipsMax_ = 3;
    std::string listFilePath_;
    unsigned int randomSeed_ = std::mt19937::default_seed;
    RandomPolicy dsetRndPolicy_;
    std::string prettyString() const;
  };

  explicit AdditiveNoise(const AdditiveNoise::Config& config);
  ~AdditiveNoise() override = default;
  void apply(std::vector<float>& signal) override;
  std::string prettyString() const override;

 private:
  const AdditiveNoise::Config conf_;
  std::mt19937 randomEngine_;
  std::uniform_int_distribution<> uniformDistribution_;
  std::uniform_int_distribution<> randomNumClipsPerUtterance_;
  std::uniform_real_distribution<> randomSnr_;
  std::unique_ptr<DatasetRandomiser<std::string>> datasetRandomiser_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
