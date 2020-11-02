/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <flashlight/fl/flashlight.h>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/app/asr/experimental/soundeffect/ListFile.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

// Interval range is like that of a C loop. Inclusive of first and exclusive of
// second. {0,0}, {10,10} are empty intervals.
using Interval = std::pair<int, int>;

enum class Backend { GPU, CPU, None };
// Accepts: "gpu", "GPU", "cpu", "CPU"
// Otherwise returns None
Backend backendFromString(const std::string& backend);
std::string backendPrettyString(Backend backend);

/**
 * Sound class is an abstraction of all the stuff we to move through the sound
 * effect pipeline. This includes Sound, format, filename, and debugging
 * stuff.
 */
class Sound {
 public:
  Sound() = default;
  Sound(const Sound& other) = default;
  explicit Sound(std::shared_ptr<std::vector<float>> cpuData);
  explicit Sound(fl::Variable gpuData);
  Sound getCopy();

  static Sound readFromFile(const std::string& filename);
  static Sound readFromFile(const ListFileEntry& listFileEntry);
  void writeToFile();

  Backend getBackend() const;
  fl::Variable getGpuData();
  std::shared_ptr<std::vector<float>> getCpuData();
  void setGpuData(fl::Variable gpuData);
  void setCpuData(std::shared_ptr<std::vector<float>> cpuData);

  size_t size() const;
  bool empty() const;
  float power();
  float rootMeanSquare();
  float maxAbs();

  Sound getInterval(const Interval& interval);
  Sound& operator+=(Sound other);
  Sound& operator-=(Sound other);
  Sound& operator*=(float a);
  Sound& operator/=(float a);
  Sound& addNoise(Sound noise, float snr);
  Sound& reverb(Sound kernel);
  // sound /= max(abs(sound)) if abs(sound) > 1.0
  Sound normalize();
  Sound normalizeIfHigh();
  std::string prettyString() const;

  ListFileEntry listFileEntry_;
  fl::app::asr::SoundInfo info_ = {/*frames=*/0,
                                   /*samplerate=*/16000,
                                   /*channels=*/1};
  fl::app::asr::SoundFormat outputFormat_ = fl::app::asr::SoundFormat::FLAC;
  fl::app::asr::SoundSubFormat outputSubformat_ =
      fl::app::asr::SoundSubFormat::PCM_16;

 private:
  // backend is mutable because it is ok to change backend on a const object.
  mutable Backend backend_ = Backend::None;
  // Only one can be valid at a time.
  mutable fl::Variable gpuData_;
  mutable std::shared_ptr<std::vector<float>> cpuData_;
};

// Interval
// Semi-internal functionality. Used for testing but possibly generally useable.
std::string intervalPrettyString(const Interval& interval);
int intervalSize(const Interval& interval);
Interval intervalIntersection(const std::vector<Interval>& intervals);
Interval intervalShift(const Interval& interval, int shift);
fl::Variable getInterval(fl::Variable all, const Interval& interval);
std::shared_ptr<std::vector<float>> getInterval(
    std::shared_ptr<std::vector<float>> all,
    const Interval& interval);

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
