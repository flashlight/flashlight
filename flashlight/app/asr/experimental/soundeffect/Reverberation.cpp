/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/Reverberation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <utility>

#include <arrayfire.h>

namespace fl {
namespace app {
namespace asr {
namespace sfx {

Sound ReverbEcho::applyImpl(Sound signal) {
  return signal.reverb(getRoomImpulseResponse());
}

namespace {
// This constant seems to create nice reverberation. We need many more
// parameters to calculate surface area to volume ratio. Insted we just choose
// this value.
constexpr float kDistToVolumeRatio = 0.01;
constexpr float kSabineConstant = 0.1611;
// Rt60: Reverberation Time
// https://en.wikipedia.org/wiki/Reverberation#Sabine_equation
// t60 = 0.1611*(V/(S*a))
// V: is the volume of the room in m3
// S: total surface area of room in m2
// a: is the average absorption coefficient of room surfaces
// Assuming distance to the 4 walls is the same and room hight is 2 meters.
// Volume proportional to distanceMeters^3 if we are
//  V ~= distanceMeters^3
// Surface area is proportional to distanceMeters^2
//  S ~= distanceMeters^2
// So:
//  V/S ~= distanceMeters
// a is the abosorption coefficient
float calcRt60(float distanceMeters, float absorptionCoefficient) {
  return kSabineConstant * distanceMeters * kDistToVolumeRatio /
      absorptionCoefficient;
}

float absorptionCoefficient(float distanceMeters, float rt60) {
  return kSabineConstant * distanceMeters * kDistToVolumeRatio / rt60;
}

} // namespace

ReverbEcho::ReverbEcho(const ReverbEcho::Config& conf)
    : conf_(conf),
      randomEngine_(conf.randomSeed_),
      randomUnit_(-1.0, 1.0),
      randomDecay_(
          calcRt60(
              conf_.distanceToWallInMetersMin_,
              conf_.absorptionCoefficientMax_),
          calcRt60(
              conf_.distanceToWallInMetersMax_,
              conf_.absorptionCoefficientMin_)),
      randomDelay_(
          conf_.distanceToWallInMetersMin_ / kSpeedOfSoundMeterPerSec,
          conf_.distanceToWallInMetersMax_ / kSpeedOfSoundMeterPerSec),
      randomNumEchos_(
          static_cast<int>(conf_.numEchosMin_),
          static_cast<int>(conf_.numEchosMax_)),
      // The kernel looks lengthMilliseconds_ into the past and into the future.
      kernelSize_((conf_.lengthMilliseconds_ / 1000.0) * conf_.sampleRate_ * 2),
      kernelCenter_(kernelSize_ / 2) {
  assert(conf_.numEchosMin_ <= conf_.numEchosMax_);
}

/**
 *  The impulse reponse's first half is the past, the second half is the
 * future, and current time is at the center. The current time has value 1. The
 * future has all zeros. The past starts from values near 1 and attenuate as a
 * function of time 't' and the reverberation time 'rt60' as: 10^(-3 * t/rt60)
 */
Sound ReverbEcho::getRoomImpulseResponse() {
  // The next block is all about thread safety
  float firstDelay = 0.0f;
  float rt60 = 0.0f;
  int numEchos = 0;
  unsigned int localRandomSeed;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    firstDelay = randomDelay_(randomEngine_);
    rt60 = randomDecay_(randomEngine_);
    numEchos = randomNumEchos_(randomEngine_);
    localRandomSeed = (randomUnit_)(randomEngine_) * (1 << 30);
  }
  std::mt19937 localRandomEngine(localRandomSeed);
  std::uniform_real_distribution<float> localRandomJitter(
      (1.0f - conf_.jitter_) * firstDelay, (1.0f + conf_.jitter_) * firstDelay);

  auto kernelVector = std::make_shared<std::vector<float>>(kernelSize_, 0);
  for (int i = 0; i < numEchos; ++i) {
    float delay = 0.0;
    float frac = 1.0;

    while (true) {
      delay += (firstDelay + localRandomJitter(localRandomEngine));
      const int indexShift = delay * conf_.sampleRate_;

      if (indexShift > kernelCenter_) {
        break;
      }

      const float attenuation =
          std::pow(10, -3 * localRandomJitter(localRandomEngine) / rt60);
      frac *= attenuation;

      if ((frac < 1e-3)) {
        break;
      }

      // dividing frac by numEchos to avoid the echo summing up too high
      kernelVector->at(kernelCenter_ - indexShift) += frac / numEchos;
    }
  }

  kernelVector->at(kernelCenter_) = 1.0;
  return Sound(kernelVector);
}

std::string ReverbEcho::Config::prettyString() const {
  std::stringstream ss;
  ss << " absorptionCoefficientMin_=" << absorptionCoefficientMin_
     << " absorptionCoefficientMax_=" << absorptionCoefficientMax_
     << " distanceToWallInMetersMin_=" << distanceToWallInMetersMin_
     << " distanceToWallInMetersMax_=" << distanceToWallInMetersMax_
     << " numEchosMin_=" << numEchosMin_ << " numEchosMax_=" << numEchosMax_
     << " jitter_=" << jitter_ << " sampleRate_=" << sampleRate_
     << " randomSeed_=" << randomSeed_
     << " lengthMilliseconds_=" << lengthMilliseconds_;
  return ss.str();
}

ReverbDataset::ReverbDataset(const ReverbDataset::Config& conf) : conf_(conf) {}

Sound ReverbDataset::loadRirFile() {
  if (!soundLoader_) {
    if (conf_.randomRirWithReplacement_) {
      soundLoader_ = std::make_shared<SoundLoaderRandomWithReplacement>(
          conf_.listFilePath_, conf_.randomSeed_);
    } else {
      soundLoader_ = std::make_shared<SoundLoaderRandomWithoutReplacement>(
          conf_.listFilePath_, conf_.randomSeed_);
    }
  }

  try {
    return soundLoader_->loadRandom();
  } catch (const std::exception& ex) {
    std::cerr
        << "ReverbDataset::RirDatasetLoader::loadRirFile() failed with error="
        << ex.what();
  }
}

Sound ReverbDataset::applyImpl(Sound signal) {
  Sound rir = loadRirFile();
  auto kernel = af::join(
      /*dim=*/0,
      af::flip(rir.getGpuData().array(), /*dim=*/0),
      af::constant(0.0, rir.size()));
  rir.setGpuData(fl::Variable(kernel, false));

  return signal.reverb(rir);
}

std::string ReverbDataset::Config::prettyString() const {
  std::stringstream ss;
  ss << " listFilePath_=" << listFilePath_
     << " randomRirWithReplacement_=" << randomRirWithReplacement_
     << " randomSeed_=" << randomSeed_;
  return ss.str();
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
