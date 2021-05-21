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
namespace app {
namespace asr {
namespace sfx {

/**
 * Applies reverberation of generated RIR, crudely calculated based on random:
 * absorption coefficient, room size, and jitter.
 * This a c++ port of:
 * https://github.com/facebookresearch/denoiser/blob/master/denoiser/augment.py
 */
class ReverbEcho : public SoundEffect {
 public:
  struct Config {
    /**
     * probability of aapplying reverb.
     */
    float proba_ = 1.0;
    /**
     * amplitude of the first echo as a fraction of the input signal. For each
     * sample, actually sampled from`[0, initial]`. Larger values means louder
     * reverb. Physically, this would depend on the absorption of the room
    walls.
     */
    float initialMin_ = 0;
    float initialMax_ = 0.3;
    /**
     * range of values to sample the RT60 in seconds, i.e. after RT60
     * seconds, the echo amplitude is 1e-3 of the first echo. The
     * default values follow the recommendations of
     * https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf,
     * Section 2.4. Physically this would also be related to the
     * absorption of the room walls and there is likely a relation
     * between `RT60` and`initial`, which we ignore here.
     */
    float rt60Min_ = 0.3;
    float rt60Max_ = 1.3;
    /**
     * range of values to sample the first echo delay in seconds. The default
     * values are equivalent to sampling a room of 3 to 10 meters.
     */
    float firstDelayMin_ = 0.01;
    float firstDelayMax_ = 0.03;
    /**
     * how many train of echos with differents jitters to add.Higher values
     * means a denser reverb.
     */
    size_t repeat_ = 3;
    /**
     * jitter used to make each repetition of the reverb echo train slightly
     * different.For instance a jitter of 0.1 means the delay  between two echos
     * will be in the range `firstDelay + -10 %`, with the jittering noise
     * being resampled after each single echo.-
     */
    float jitter_ = 0.1;
    /**
     * fraction of the reverb of the clean speech to add back to the ground
     * truth .0 = dereverberation, 1 = no dereverberation.
     */
    size_t sampleRate_ = 16000;
    std::string prettyString() const;
  };

  explicit ReverbEcho(const ReverbEcho::Config& config, unsigned int seed = 0);
  ~ReverbEcho() override = default;
  void apply(std::vector<float>& sound) override;
  std::string prettyString() const override;

 private:
  // augments source with reverberation noise
  void applyReverb(
      std::vector<float>& source,
      float initial,
      float firstDelay,
      float rt60);

  const ReverbEcho::Config conf_;
  RandomNumberGenerator rng_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
