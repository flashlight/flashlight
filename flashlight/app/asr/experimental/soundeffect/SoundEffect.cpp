/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/SoundEffect.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "flashlight/lib/common/System.h"

using ::fl::lib::pathsConcat;

namespace fl {
namespace app {
namespace asr {
namespace sfx {

std::string SoundEffect::prettyString() const {
  std::stringstream ss;
  ss << "name=" << name() << " enableCountDown_=" << enableCountDown_
     << " augSuccess_ = " << augSuccess_ << " augFailure_ = " << augFailure_
     << " successRate="
     << (((double)augSuccess_ / (double)(augSuccess_ + augFailure_)) * 100.0)
     << "%";
  return ss.str();
}

std::function<void(std::vector<float>*)> SoundEffect::asStdFunction() {
  return [this](std::vector<float>* data) {
    // 1. Create a temporary empty signal Sound object.
    // 2. Swap the input data into a temporary signal Sound object.
    // 3. Augment the data in the Sound object
    // 4. swap the data back to the input vector.
    Sound signal(std::make_shared<std::vector<float>>());
    signal.getCpuData()->swap(*data);
    Sound augmented = this->apply(signal);
    augmented.getCpuData()->swap(*data);
  };
}

void SoundEffect::setEnableCountdown(int enableCountDown) {
  enableCountDown_ = enableCountDown;
}

Sound SoundEffect::apply(Sound signal) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (enableCountDown_ > 0) {
      --enableCountDown_;
      return signal;
    }
  }

  if (signal.empty()) {
    return signal;
  }

  Sound augmented;
  try {
    augmented = applyImpl(signal.getCopy());
    ++augSuccess_;
  } catch (std::exception& ex) {
    ++augFailure_;
    FL_LOG(fl::WARNING) << "SoundEffect::apply(Sound augmented={"
                        << augmented.prettyString() << "}) failed with error={"
                        << ex.what() << "} this={" << prettyString() << '}';
    return signal;
  }
  return augmented;
}

void SoundEffectChain::setEnableCountdown(int enableCountDown) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (std::shared_ptr<SoundEffect>& effect : soundEffects_) {
    effect->setEnableCountdown(0);
  }
  SoundEffect::setEnableCountdown(enableCountDown);
}

std::string SoundEffectChain::prettyString() const {
  std::stringstream ss;
  ss << '{' << std::endl;
  for (const std::shared_ptr<SoundEffect>& sfx : soundEffects_) {
    ss << sfx->name() << "={" << sfx->prettyString() << '}' << std::endl;
  }
  ss << '}';
  return ss.str();
}

Sound SoundEffectChain::applyImpl(Sound signal) {
  Sound augmented = signal;
  for (std::shared_ptr<SoundEffect>& effect : soundEffects_) {
    signal = augmented;
    augmented = effect->apply(signal);
  }
  return augmented;
}

Sound Normalize::applyImpl(Sound signal) {
  return signal.normalizeIfHigh();
}

std::string Normalize::prettyString() const {
  std::stringstream ss;
  ss << "SoundEffect={" << SoundEffect::prettyString() << '}';
  return ss.str();
}

std::string ClampAmplitude::prettyString() const {
  std::stringstream ss;
  ss << "SoundEffect={" << SoundEffect::prettyString() << '}';
  return ss.str();
}

Sound ClampAmplitude::applyImpl(Sound signal) {
  signal.setGpuData(fl::clamp(signal.getGpuData(), -1.0, +1.0));
  return signal;
}

Amplify::Amplify(const Amplify::Config& config)
    : randomEngine_(config.randomSeed_),
      randomRatio_(config.ratioMin_, config.ratioMax_) {}

std::string Amplify::prettyString() const {
  std::stringstream ss;
  ss << "Amplify{SoundEffect={" << SoundEffect::prettyString() << "}}";
  return ss.str();
}

Sound Amplify::applyImpl(Sound signal) {
  float ratio = 0;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    ratio = randomRatio_(randomEngine_);
  }
  signal *= ratio;
  return signal;
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
