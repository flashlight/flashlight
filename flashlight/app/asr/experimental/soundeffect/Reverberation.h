// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>
#include <mutex>
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

// Crudely estimates RIR based on absorption coefficient and room size.
class ReverbEcho : public SoundEffect {
 public:
  struct Config {
    size_t sampleRate_ = 16000;
    unsigned int randomSeed_ = std::mt19937::default_seed;
    int lengthMilliseconds_ = 1600; // Convolution kernel length
    // https://www.acoustic-supplies.com/absorption-coefficient-chart/
    float absorptionCoefficientMin_ = 0.01; // painted brick
    float absorptionCoefficientMax_ = 0.99; // best absorptive wall materials
    float distanceToWallInMetersMin_ = 1.0;
    float distanceToWallInMetersMax_ = 50.0;
    size_t numEchosMin_ = 1; // number of pairs of reflective objects
    size_t numEchosMax_ = 10; // number of pairs of reflective objects
    float jitter_ = 0.1;

    std::string prettyString() const;
  };

  ReverbEcho(const ReverbEcho::Config& config);
  ~ReverbEcho() override = default;
  Sound getRoomImpulseResponse();

  Sound applyImpl(Sound signal) override;

  std::string prettyString() const override {
    return "ReverbEcho{conf_=" + conf_.prettyString() + "}}";
  };

  std::string name() const override {
    return "ReverbEcho";
  };

 private:
  const ReverbEcho::Config conf_;
  std::mt19937 randomEngine_;
  std::uniform_real_distribution<float> randomUnit_;
  std::uniform_real_distribution<float> randomDecay_;
  std::uniform_real_distribution<float> randomDelay_;
  std::uniform_int_distribution<int> randomNumEchos_;
  const int kernelSize_;
  // kernelCenter_ = kernelSize_/2 but it makes the code more readable
  const int kernelCenter_;
};

// Apply RIR from a dataset.
class ReverbDataset : public SoundEffect {
 public:
  struct Config {
    unsigned int randomSeed_ = std::mt19937::default_seed;
    std::string listFilePath_;
    bool randomRirWithReplacement_ = true;

    std::string prettyString() const;
  };

  ReverbDataset(const ReverbDataset::Config& config);
  ~ReverbDataset() override = default;
  Sound loadRirFile();

  Sound applyImpl(Sound signal) override;

  std::string prettyString() const override {
    return "ReverbDataset{conf_=" + conf_.prettyString() + "}}";
  };

  std::string name() const override {
    return "ReverbDataset";
  };

 private:
  const ReverbDataset::Config conf_;
  std::shared_ptr<SoundLoader> soundLoader_;
};

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
