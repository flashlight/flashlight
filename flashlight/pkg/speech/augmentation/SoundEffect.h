/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

/**
 * Base class for sound effects.
 */
class SoundEffect {
 public:
  SoundEffect() = default;
  virtual ~SoundEffect() = default;
  virtual void apply(std::vector<float>& sound) = 0;
  virtual std::string prettyString() const = 0;
};

/**
 * A container for chaining sound effect. It serially applies calls to all added
 * sound effects.
 */
class SoundEffectChain : public SoundEffect {
 public:
  SoundEffectChain() {}
  ~SoundEffectChain() override = default;
  void apply(std::vector<float>& sound) override;
  std::string prettyString() const override;
  void add(std::shared_ptr<SoundEffect> SoundEffect);
  bool empty();

 protected:
  std::vector<std::shared_ptr<SoundEffect>> soundEffects_;
};

/**
 * Normalize amplitudes to range -1..1 using dynamic range linear compression.
 * No-op if the signal's amplitudes are already within that range.
 */
class Normalize : public SoundEffect {
 public:
  explicit Normalize(bool onlyIfTooHigh = true);
  ~Normalize() override = default;
  void apply(std::vector<float>& sound) override;
  std::string prettyString() const override;

 private:
  bool onlyIfTooHigh_;
};

/**
 * Clamps amplitudes to range -1..1.
 * No-op if the signal's amplitudes are already within that range.
 */
class ClampAmplitude : public SoundEffect {
 public:
  explicit ClampAmplitude() {}
  ~ClampAmplitude() override = default;
  void apply(std::vector<float>& sound) override;

  std::string prettyString() const override;
};

/**
 * Amplifies (or decreases amplitude of) the signal with a random ratio in the
 * specified range.
 */
class Amplify : public SoundEffect {
 public:
  struct Config {
    float ratioMin_;
    float ratioMax_;
    unsigned int randomSeed_;
  };

  explicit Amplify(const Amplify::Config& config);
  ~Amplify() override = default;
  void apply(std::vector<float>& sound) override;
  std::string prettyString() const override;

 private:
  std::mt19937 randomEngine_;
  std::uniform_real_distribution<> randomRatio_;
  std::mutex mutex_;
};

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
