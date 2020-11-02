/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <flashlight/fl/flashlight.h>

#include "flashlight/app/asr/experimental/soundeffect/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundEffect.h"
#include "flashlight/app/asr/experimental/soundeffect/SoundLoader.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

class AdditiveNoise : public SoundEffect {
 public:
  struct Config {
    double maxTimeRatio_ = 1.0;
    double minSnr_ = 0.5;
    double maxSnr_ = 2.0;
    int nClipsPerUtteranceMin_ = 0;
    int nClipsPerUtteranceMax_ = 3.0;
    std::string listFilePath_;
    // Using random noise without replacement when false.
    bool randomNoiseWithReplacement_ = true;
    unsigned int randomSeed_ = std::mt19937::default_seed;

    std::string prettyString() const;
  };

  AdditiveNoise(const AdditiveNoise::Config config);
  ~AdditiveNoise() override = default;
  Sound applyImpl(Sound signal) override;

  std::string prettyString() const override;
  std::string name() const override {
    return "AdditiveNoise";
  };

 private:
  std::vector<Sound> loadNoises();
  Interval augmentationInterval(size_t size);
  Sound getNoise(size_t size);

  const AdditiveNoise::Config conf_;
  std::mt19937 randomEngine_;
  std::uniform_int_distribution<> uniformDistribution_;
  std::uniform_int_distribution<> randomNumClipsPerUtterance_;
  std::uniform_real_distribution<> randomSnr_;
  std::shared_ptr<SoundLoader> soundLoader_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
