/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <flashlight/fl/flashlight.h>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/Sound.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {
constexpr float kSpeedOfSoundMeterPerSec = 343.0;

class SoundEffect {
 public:
  SoundEffect() {}
  virtual ~SoundEffect() = default;
  Sound apply(Sound signal);
  // applyImpl() will no be be called until after after enableCountDown calls to
  // apply().
  virtual void setEnableCountdown(int enableCountDown);

  // Retuns a functions that augments @data using this object.
  std::function<void(std::vector<float>* data)> asStdFunction();

  virtual std::string prettyString() const;
  virtual std::string name() const = 0;
  virtual Sound applyImpl(Sound signal) = 0;

 protected:
  std::mutex mutex_;
  // Block augmentation while enableCountDown_ > 0
  int enableCountDown_ = 0;
  long augSuccess_ = 0;
  long augFailure_ = 0;
};

class SoundEffectChain : public SoundEffect {
 public:
  SoundEffectChain() {}
  ~SoundEffectChain() override = default;
  Sound applyImpl(Sound signal) override;
  void add(std::shared_ptr<SoundEffect> SoundEffect) {
    soundEffects_.push_back(SoundEffect);
  }
  void setEnableCountdown(int enableCountDown) override;

  std::string prettyString() const override;
  std::string name() const override {
    return "SoundEffectChain";
  }

 private:
  std::vector<std::shared_ptr<SoundEffect>> soundEffects_;
};

// Normalize amplitude to range -1..1 using dynamic range compression/expension
class Normalize : public SoundEffect {
 public:
  explicit Normalize() {}
  ~Normalize() override = default;
  Sound applyImpl(Sound signal) override;

  std::string name() const override {
    return "Normalize";
  }
  std::string prettyString() const override;
};

class ClampAmplitude : public SoundEffect {
 public:
  explicit ClampAmplitude() {}
  ~ClampAmplitude() override = default;
  Sound applyImpl(Sound signal) override;

  std::string name() const override {
    return "ClampAmplitude";
  }
  std::string prettyString() const override;
};

class Amplify : public SoundEffect {
 public:
  struct Config {
    float ratioMin_;
    float ratioMax_;
    unsigned int randomSeed_;
  };

  Amplify(const Amplify::Config& config);
  ~Amplify() override = default;
  Sound applyImpl(Sound signal) override;

  std::string name() const override {
    return "Amplify";
  }
  std::string prettyString() const override;

 private:
  std::mt19937 randomEngine_;
  std::uniform_real_distribution<> randomRatio_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
