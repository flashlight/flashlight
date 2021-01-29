/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/augmentation/SoundEffect.h"

#include <random>
#include <string>
#include <vector>

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

/**
 * https://kaldi-asr.org/doc/wav-reverberate_8cc_source.html
 */
class ReverbAndNoise : public SoundEffect {
 public:
  struct Config {
    /**
     * probability of applying reverb.
     */
    float proba_ = 1.0;

    std::string rirListFilePath_;
    /**
     * Force the scaling factor of RIR samples to the specified ratio. When
     * value is zero, automatic scaling is applied. This functionality is the
     * same as in
     * https://kaldi-asr.org/doc/wav-reverberate_8cc_source.html#l00210. Scaling
     * is required due to "...over-excitation caused by amplifying the audio
     * using a RIR ..." (https://arxiv.org/pdf/1811.06795.pdf section VI.E).
     */
    float volume_ = 0;
    size_t sampleRate_ = 16000;

    double ratio_ = 1.0;
    double minSnr_ = 0;
    double maxSnr_ = 30;
    int nClipsMin_ = 1;
    int nClipsMax_ = 3;
    std::string noiseListFilePath_;

    unsigned int randomSeed_ = std::mt19937::default_seed;
    std::string prettyString() const;
  };

  explicit ReverbAndNoise(
      const ReverbAndNoise::Config& config,
      unsigned int seed = 0);
  ~ReverbAndNoise() override = default;
  void apply(std::vector<float>& sound) override;
  std::string prettyString() const override;

 private:
  // augments source with reverberation noise
  void reverb(
      std::vector<float>& source,
      float initial,
      float firstDelay,
      float rt60);

  const ReverbAndNoise::Config conf_;
  RandomNumberGenerator rng_;
  std::vector<std::string> rirFiles_;
  std::vector<std::string> noiseFiles_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
