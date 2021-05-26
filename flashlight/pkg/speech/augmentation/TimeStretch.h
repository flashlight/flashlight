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
#include "flashlight/pkg/speech/augmentation/SoxWrapper.h"

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

#ifdef FL_BUILD_APP_ASR_SFX_SOX

/**
 * Stretches signal within specified ratio range using libSOX as backend.
 */
class TimeStretch : public SoundEffect {
 public:
  struct Config {
    /**
     * probability of applying reverb.
     */
    float proba_ = 1.0;
    double minFactor_ = 0.8; /* stretch factor. 1.0 means copy. */
    double maxFactor_ = 1.25; /* stretch factor. 1.0 means copy. */
    size_t sampleRate_ = 16000;
    std::string prettyString() const;
  };

  explicit TimeStretch(
      const TimeStretch::Config& config,
      unsigned int seed = 0);
  ~TimeStretch() override = default;
  void apply(std::vector<float>& data) override;
  std::string prettyString() const override;

 private:
  const TimeStretch::Config conf_;
  RandomNumberGenerator rng_;
  // The next 2 pointers are kept for optimization and readability.
  // They point to existing objects that are not constructed or
  // destroyed by this class
  SoxWrapper const* sox_;
  const sox_effect_handler_t* stretchEffect_;
};

#else /* ifdef FL_BUILD_APP_ASR_SFX_SOX */
// Definition with null implementation to stub out TimeStretch
// when building sound effects without libsox.
class TimeStretch : public SoundEffect {
 public:
  struct Config {
    float proba_;
    double minFactor_;
    double maxFactor_;
    size_t sampleRate_;
    std::string prettyString() const {
      return "";
    }
  };
  explicit TimeStretch(
      const TimeStretch::Config& config,
      unsigned int seed = 0) {}
  ~TimeStretch() override = default;
  void apply(std::vector<float>& data) override {}
  std::string prettyString() const override {
    return "";
  }
};
#endif

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
