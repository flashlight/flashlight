/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/experimental/soundeffect/Sound.h"

#include <cassert>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/modules/Conv2D.h"
#include "flashlight/lib/common/System.h"

using ::fl::app::asr::loadSound;
using ::fl::app::asr::loadSoundInfo;
using ::fl::lib::dirCreateRecursive;
using ::fl::lib::dirname;

namespace fl {
namespace app {
namespace asr {
namespace sfx {

Sound::Sound(std::shared_ptr<std::vector<float>> cpuData)
    : backend_(Backend::CPU), cpuData_(cpuData) {}

Sound::Sound(fl::Variable gpuData)
    : backend_(Backend::GPU), gpuData_(gpuData) {}

Sound Sound::readFromFile(const std::string& filename) {
  Sound sound;
  sound.listFileEntry_.audioFilePath_ = filename;

  try {
    sound.cpuData_ = std::make_shared<std::vector<float>>(
        loadSound<float>(sound.listFileEntry_.audioFilePath_));
    sound.info_ = loadSoundInfo(sound.listFileEntry_.audioFilePath_);
  } catch (std::exception& ex) {
    std::cerr << "Sound::readFromFile(filename=" << filename
              << ") fails to read file with error=" << ex.what() << std::endl;
  }

  return sound;
}

Sound Sound::readFromFile(const ListFileEntry& listFileEntry) {
  Sound sound;
  sound.listFileEntry_ = listFileEntry;

  try {
    sound.setCpuData(std::make_shared<std::vector<float>>(
        loadSound<float>(sound.listFileEntry_.audioFilePath_)));
    sound.info_ = loadSoundInfo(sound.listFileEntry_.audioFilePath_);
  } catch (std::exception& ex) {
    std::stringstream ss;
    ss << "Sound::readFromFile(listFileEntry=" << listFileEntry.prettyString()
       << ") fails to read file with error=" << ex.what();
    throw std::runtime_error(ss.str());
  }
  if (sound.empty()) {
    std::stringstream ss;
    ss << "Sound::readFromFile(listFileEntry=" << listFileEntry.prettyString()
       << ") empty data={" << sound.prettyString() << "}";
    throw std::runtime_error(ss.str());
  }

  return sound;
}

std::string Sound::prettyString() const {
  std::stringstream ss;
  ss << "backend_=" << backendPrettyString(backend_) << " size=" << size()
     << " info_={" << info_.prettyString() << '}' << " listFileEntry_={"
     << listFileEntry_.prettyString() << '}';
  return ss.str();
}

void Sound::writeToFile() {
  if (listFileEntry_.audioFilePath_.empty()) {
    std::stringstream ss;
    ss << "Sound:writeToFile() sound={" << prettyString()
       << "} listFileEntry_.audioFilePath_ is empty.";
    throw std::invalid_argument(ss.str());
  }
  if (empty()) {
    std::stringstream ss;
    ss << "Sound:writeToFile() sound={" << prettyString()
       << "} fails to write sound to file since data is empty.";
    throw std::invalid_argument(ss.str());
  }

  try {
    const std::string path = dirname(listFileEntry_.audioFilePath_);
    dirCreateRecursive(path);
  } catch (std::exception& ex) {
    std::stringstream ss;
    ss << "Sound:writeToFile() sound={" << prettyString()
       << "} fails to create directory for file="
       << listFileEntry_.audioFilePath_ << " with error=" << ex.what();
    throw std::invalid_argument(ss.str());
  }

  try {
    saveSound(
        listFileEntry_.audioFilePath_,
        *getCpuData(),
        info_.samplerate,
        info_.channels,
        outputFormat_,
        outputSubformat_);
  } catch (std::exception& ex) {
    std::stringstream ss;
    ss << "Sound:writeToFile() sound={" << prettyString()
       << "} fails to create directory for file="
       << listFileEntry_.audioFilePath_ << " with error=" << ex.what();
    throw std::invalid_argument(ss.str());
  }
}

Sound Sound::getCopy() {
  Sound other = *this;
  if (!empty()) {
    if (backend_ == Backend::CPU) {
      other.cpuData_ = std::make_shared<std::vector<float>>(
          other.cpuData_->begin(), other.cpuData_->end());
    } else {
      other.gpuData_ = fl::Variable(this->gpuData_.array().copy(), false);
    }
  }
  return other;
}

size_t Sound::size() const {
  switch (backend_) {
    case Backend::GPU:
      return gpuData_.elements();
    case Backend::CPU:
      return cpuData_ ? cpuData_->size() : 0UL;
    case Backend::None:
      return 0UL;
  }
}

bool Sound::empty() const {
  return size() == 0;
}

Backend Sound::getBackend() const {
  return backend_;
}

fl::Variable Sound::getGpuData() {
  if (backend_ == Backend::CPU) {
    if (cpuData_ && cpuData_->size() > 0) {
      gpuData_ =
          fl::Variable(af::array(cpuData_->size(), cpuData_->data()), false);
      cpuData_.reset();
    } else {
      gpuData_ = fl::Variable();
    }
    backend_ = Backend::GPU;
  }
  return gpuData_;
}

std::shared_ptr<std::vector<float>> Sound::getCpuData() {
  if (backend_ == Backend::GPU) {
    if (gpuData_.elements() > 0) {
      cpuData_ = std::make_shared<std::vector<float>>(gpuData_.elements());
      gpuData_.host(cpuData_->data());
      gpuData_ = fl::Variable();
    } else {
      cpuData_.reset();
    }
    backend_ = Backend::CPU;
  }
  return cpuData_;
}

void Sound::setGpuData(fl::Variable gpuData) {
  cpuData_.reset();
  gpuData_ = gpuData;
  backend_ = Backend::GPU;
}

void Sound::setCpuData(std::shared_ptr<std::vector<float>> cpuData) {
  cpuData_ = cpuData;
  gpuData_ = fl::Variable();
  backend_ = Backend::CPU;
}

fl::Variable getInterval(fl::Variable all, const Interval& interval) {
  const Interval dataInterval(0, all.elements());
  const Interval intersection = intervalIntersection({dataInterval, interval});
  if (intersection == dataInterval) {
    return all;
  } else if (intervalSize(intersection) > 0) {
    return fl::Variable(
        all.array()(af::seq(intersection.first, intersection.second)), false);
  } else {
    return {};
  }
}

std::shared_ptr<std::vector<float>> getInterval(
    std::shared_ptr<std::vector<float>> all,
    const Interval& interval) {
  const Interval dataInterval(0, all->size());
  const Interval intersection = intervalIntersection({dataInterval, interval});
  if (intersection == dataInterval) {
    return all;
  } else if (intervalSize(intersection) > 0) {
    return std::make_shared<std::vector<float>>(
        all->begin() + intersection.first, all->begin() + intersection.second);
  } else {
    return {};
  }
}

Sound Sound::getInterval(const Interval& interval) {
  Sound soundInterval = *this;
  if (!empty()) {
    if (backend_ == Backend::CPU) {
      soundInterval.setCpuData(
          fl::app::asr::sfx::getInterval(cpuData_, interval));
    } else {
      soundInterval.setGpuData(
          fl::app::asr::sfx::getInterval(gpuData_, interval));
    }
  }
  return soundInterval;
}

std::vector<float> vectorDiff(
    const std::vector<float>& a,
    const std::vector<float>& b) {
  assert(a.size() == b.size());
  std::vector<float> diff(a.size());
  for (int i = 0; i < a.size(); ++i) {
    diff[i] = a[i] - b[i];
  }
  return diff;
}

Sound operator-(Sound lhs, Sound rhs) {
  if (lhs.size() != rhs.size()) {
    std::stringstream ss;
    ss << "operator-(Sound lhs.size()=" << lhs.size()
       << ", Sound rhs.size()=" << rhs.size() << ")"
       << " size mismatch error.";
    throw std::invalid_argument(ss.str());
  }
  if (lhs.empty()) {
    return lhs;
  } else if (lhs.getBackend() == Backend::CPU) {
    return Sound(std::make_shared<std::vector<float>>(
        vectorDiff(*lhs.getCpuData(), *rhs.getCpuData())));
  } else {
    return Sound(lhs.getGpuData() - rhs.getGpuData());
  }
}

float Sound::power() {
  float sumOfSquares = 0;
  if (empty()) {
    return 0.0;
  } else if (backend_ == Backend::CPU) {
    const std::vector<float>& sound = *getCpuData();
    for (float i : sound) {
      sumOfSquares += i * i;
    }
  } else {
    fl::Variable sound = getGpuData();
    fl::Variable soundSqr = sound * sound;
    af::sum(soundSqr.array()).host(&sumOfSquares);
  }
  return sumOfSquares / static_cast<float>(size());
}

float Sound::rootMeanSquare() {
  return std::sqrt(power());
}

Sound Sound::normalize() {
  if (empty()) {
    std::stringstream ss;
    ss << "Sound::normalize() failed since empty data={" << prettyString()
       << '}';
    throw std::invalid_argument(ss.str());
  }
  const float maxAbsVal = maxAbs();
  if (maxAbsVal <= 0) {
    std::stringstream ss;
    ss << "Sound::normalize() maxAbsVal=" << maxAbsVal << " data={"
       << prettyString() << '}'
       << " invalid maxAbsVal value for normalization.";
    throw std::invalid_argument(ss.str());
  }
  operator/=(maxAbsVal);
  return *this;
}

Sound Sound::normalizeIfHigh() {
  if (empty()) {
    std::stringstream ss;
    ss << "Sound::normalizeIfHigh() failed since empty data={" << prettyString()
       << '}';
    throw std::invalid_argument(ss.str());
  }
  const float maxAbsVal = maxAbs();
  if (maxAbsVal <= 0) {
    std::stringstream ss;
    ss << "Sound::normalizeIfHigh() maxAbsVal=" << maxAbsVal << " data={"
       << prettyString() << '}'
       << " invalid maxAbsVal value for normalization.";
    throw std::invalid_argument(ss.str());
  }
  if (maxAbsVal > 1.0) {
    operator/=(maxAbsVal);
  }
  return *this;
}

float snrDb(Sound signal, Sound noise) {
  const float signalRms = signal.rootMeanSquare();
  const float noiseRms = noise.rootMeanSquare();
  if (noiseRms <= 0) {
    std::stringstream ss;
    ss << "snrDb(signal, noise) signalRms=" << signalRms
       << " noiseRms=" << noiseRms << " signal={" << signal.prettyString()
       << "} noise={" << noise.prettyString() << '}'
       << " invalid noiseRms for SNR calculation.";
    throw std::invalid_argument(ss.str());
  }
  return 20.0 * log10(signalRms / noiseRms);
}

float Sound::maxAbs() {
  float maxAbsAmp = 0;
  if (!empty()) {
    if (backend_ == Backend::CPU) {
      const std::vector<float>& sound = *getCpuData();
      for (float i : sound) {
        maxAbsAmp = std::fmax(maxAbsAmp, std::fabs(i));
      }
    } else {
      af::max(fl::abs(gpuData_).array()).host(&maxAbsAmp);
    }
    return maxAbsAmp;
  }
}

Sound& Sound::reverb(Sound kernel) {
  auto reverbConv = std::make_shared<fl::Conv2D>(
      kernel.getGpuData(),
      /*sx=*/1,
      /*sy*/ 1,
      /*px=*/kernel.size() / 2);

  setGpuData(reverbConv->forward(getGpuData()));
  return *this;
}

Sound& Sound::addNoise(Sound noise, float snr) {
  if (size() != noise.size()) {
    std::stringstream ss;
    ss << "Sound::addNoise(noise.size()=" << noise.size() << ", snr=" << snr
       << ')' << " size()=" << size() << " size must be the same.";
    throw std::invalid_argument(ss.str());
  }
  if (empty()) {
    return *this;
  }
  const float noiseRms = noise.rootMeanSquare();
  const float signalRms = rootMeanSquare();
  if (noiseRms <= 0) {
    std::stringstream ss;
    ss << "Sound::addNoise(noise, snr=" << snr << ") noiseRms=" << noiseRms
       << " signalRms=" << signalRms << " INVALID noiseRms ";
    throw std::invalid_argument(ss.str());
  }
  if (signalRms <= 0) {
    std::stringstream ss;
    ss << "Sound::addNoise(noise, snr=" << snr << ") noiseRms=" << noiseRms
       << " signalRms=" << signalRms << " INVALID signalRms ";
    throw std::invalid_argument(ss.str());
  }
  const float adjustRms = signalRms / std::pow(10, snr / 20.0);
  const float multiplier = adjustRms / noiseRms;
  noise *= multiplier;
  *this += noise;
  return *this;
}

std::string intervalPrettyString(const Interval& interval) {
  std::stringstream ss;
  ss << "first=" << interval.first << " second=" << interval.second
     << " size=" << intervalSize(interval);
  return ss.str();
}

Interval intervalIntersection(const std::vector<Interval>& intervals) {
  Interval result = intervals.front();
  for (int i = 1; i < intervals.size(); i++) {
    if (intervals[i].first > result.second ||
        intervals[i].second < result.first) {
      return {0, 0};
    } else {
      result.first = std::max(result.first, intervals[i].first);
      result.second = std::min(result.second, intervals[i].second);
    }
  }
  return result;
}

int intervalSize(const Interval& interval) {
  return interval.second - interval.first;
}

Interval intervalShift(const Interval& interval, int shift) {
  return {interval.first + shift, interval.second + shift};
}

Sound& Sound::operator+=(Sound other) {
  if (size() != other.size()) {
    std::stringstream ss;
    ss << "Sound::operator+=(other.size()=" << other.size()
       << ") this.size()=" << size() << " size must be the same.";
    throw std::invalid_argument(ss.str());
  }
  if (!empty()) {
    setGpuData(getGpuData() + other.getGpuData());
  }
  return *this;
}

Sound& Sound::operator-=(Sound other) {
  if (size() != other.size()) {
    std::stringstream ss;
    ss << "Sound::operator-=(other.size()=" << other.size()
       << ") this.size()=" << size() << " size must be the same.";
    throw std::invalid_argument(ss.str());
  }
  if (!empty()) {
    setGpuData(getGpuData() - other.getGpuData());
  }
  return *this;
}

Sound& Sound::operator*=(float a) {
  if (!empty()) {
    setGpuData(getGpuData() * a);
  }
  return *this;
}

Sound& Sound::operator/=(float a) {
  return operator*=(1.0 / a);
}

std::string backendPrettyString(Backend backend) {
  switch (backend) {
    case Backend::GPU:
      return "GPU";
    case Backend::CPU:
      return "CPU";
    case Backend::None:
      return "None";
    default:
      return "Invalid_sfx_backend_type";
  }
}
namespace {
std::string strToLower(std::string s) {
  std::transform(
      s.begin(),
      s.end(),
      s.begin(),
      [](unsigned char c) { return std::tolower(c); } // correct
  );
  return s;
}
} // namespace

Backend backendFromString(const std::string& backend) {
  std::string lowerCase = strToLower(backend);

  if (lowerCase == "cpu") {
    return Backend::CPU;
  } else if (lowerCase == "gpu") {
    return Backend::GPU;
  } else {
    return Backend::None;
  }
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
