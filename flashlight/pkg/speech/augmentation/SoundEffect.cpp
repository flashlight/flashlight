/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/augmentation/SoundEffect.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

std::string SoundEffectChain::prettyString() const {
  std::stringstream ss;
  ss << '{' << std::endl;
  for (const std::shared_ptr<SoundEffect>& sfx : soundEffects_) {
    ss << "{" << sfx->prettyString() << '}' << std::endl;
  }
  ss << '}';
  return ss.str();
}

void SoundEffectChain::add(std::shared_ptr<SoundEffect> SoundEffect) {
  soundEffects_.push_back(SoundEffect);
}

void SoundEffectChain::apply(std::vector<float>& sound) {
  for (std::shared_ptr<SoundEffect>& effect : soundEffects_) {
    effect->apply(sound);
  }
}

bool SoundEffectChain::empty() {
  return soundEffects_.empty();
}

Normalize::Normalize(bool onlyIfTooHigh) : onlyIfTooHigh_(onlyIfTooHigh) {}

void Normalize::apply(std::vector<float>& sound) {
  float maxAbs = 0.0f;
  for (float i : sound) {
    maxAbs = std::fmax(maxAbs, std::fabs(i));
  }
  if (!onlyIfTooHigh_ || maxAbs > 1.0f) {
    std::transform(
        sound.begin(),
        sound.end(),
        sound.begin(),
        [maxAbs](float amp) -> float { return amp / maxAbs; });
  }
}

std::string Normalize::prettyString() const {
  std::stringstream ss;
  ss << "Normalize={onlyIfTooHigh=" << onlyIfTooHigh_ << "}";
  return ss.str();
}

std::string ClampAmplitude::prettyString() const {
  return "ClampAmplitude";
}

void ClampAmplitude::apply(std::vector<float>& sound) {
  std::transform(
      sound.begin(), sound.end(), sound.begin(), [](float amp) -> float {
        return std::fmax(std::fmin(amp, 1.0), -1.0);
      });
}

Amplify::Amplify(const Amplify::Config& config)
    : randomEngine_(config.randomSeed_),
      randomRatio_(config.ratioMin_, config.ratioMax_) {}

std::string Amplify::prettyString() const {
  return "Amplify";
}

void Amplify::apply(std::vector<float>& sound) {
  float ratio = 0;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    ratio = randomRatio_(randomEngine_);
  }
  std::transform(
      sound.begin(), sound.end(), sound.begin(), [ratio](float amp) -> float {
        return amp * ratio;
      });
}

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
