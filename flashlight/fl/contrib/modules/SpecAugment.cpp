/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>
#include <math.h>
#include <stdexcept>

#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/contrib/modules/SpecAugment.h"

namespace fl {

SpecAugment::SpecAugment(
    int tWarpW,
    int fMaskF,
    int nFMask,
    int tMaskT,
    float tMaskP,
    int nTMask,
    RawWavSpecAugmentConfig rawWaveConfig,
    MaskingStrategy mStrategy /* = MaskingStrategy::ZERO */)
    : timeWarpW_(tWarpW),
      freqMaskF_(fMaskF),
      numFreqMask_(nFMask),
      timeMaskT_(tMaskT),
      timeMaskP_(tMaskP),
      numTimeMask_(nTMask),
      maskStrategy_(mStrategy),
      useRawWav_(rawWaveConfig.useRawWav),
      rawWavNMels_(rawWaveConfig.nMels),
      rawWavLowFreqHz_(rawWaveConfig.lowFreqHz),
      rawWavHighFreqHz_(rawWaveConfig.highFreqHz),
      rawWavSampleRate_(rawWaveConfig.sampleRate),
      maxKernelSize_(rawWaveConfig.maxKernelSize) {
  if (numFreqMask_ > 0 && freqMaskF_ <= 0) {
    throw std::invalid_argument("invalid arguments for frequency masking.");
  }
  if (numTimeMask_ > 0 && timeMaskT_ <= 0) {
    throw std::invalid_argument("invalid arguments for time masking.");
  }
  if (numTimeMask_ > 0 && (timeMaskP_ <= 0 || timeMaskP_ > 1.0)) {
    throw std::invalid_argument("invalid arguments for time masking.");
  }
  if (useRawWav_ && (rawWavLowFreqHz_ < 0 || rawWavHighFreqHz_ < 0 || rawWavLowFreqHz_ >= rawWavHighFreqHz_)) {
    throw std::invalid_argument("invalid arguments for raw Wav high and low frequencies.");
  }
  if (useRawWav_ && rawWavNMels_ <= 0) {
    throw std::invalid_argument("invalid arguments for raw Wav nMels.");
  }
  rawWavPrecompute();
}

void SpecAugment::rawWavPrecompute() {
  if (useRawWav_ && lowPassFilters_.empty()) {
    auto mel2hz = [](float mel) {
      return 700.0 * (std::pow(10, (mel / 2595.0)) - 1.0);
    };
    auto hz2mel = [](float hz) {
      return 2595.0 * std::log10(1.0 + hz / 700.0);
    };
    float minMel = hz2mel(rawWavLowFreqHz_), maxMel = hz2mel(rawWavHighFreqHz_);
    // nMels intervals and nMels + 1 points
    float delta = (maxMel - minMel) / rawWavNMels_;
    float currentMel = minMel;
    // set transition band as half of lowest bin frequency size (left bin)
    // for lowest frequency set it to half of the right bin
    // cutoff frequency and transmision band are stored from 0 to 0.5 of sampling rate
    std::vector<float> transBandKhz(rawWavNMels_ + 1);
    for (int index = 0; index <= rawWavNMels_; index++) {
      cutoff_.push_back(mel2hz(currentMel) / rawWavSampleRate_);
      currentMel += delta;
      if (index > 0) {
        transBandKhz[index] = cutoff_[index - 1] / 4.;
      }
    }
    transBandKhz[0] = transBandKhz[1];
    ignoredLowPassFilters_ = 0;
    // compute filters for each frequency point, nMel + 1 low pass filters
    for (int fidx = 0; fidx < cutoff_.size(); fidx++) {
      int width = 2. / (1e-6 + transBandKhz[fidx]);
      if (width * 2 + 1 > maxKernelSize_) {
        FL_LOG(fl::INFO) << "SpecAugment raw wave: frequency " << cutoff_[fidx]
                         << " will be skipped for eval, too large kernel";
        lowPassFilters_.push_back(nullptr);
        ignoredLowPassFilters_++;
        continue;
      }
      af::array indexArr = af::iota(af::dim4(2 * width + 1));
      af::array blackmanWindow =
        0.42 - 0.5 * af::cos(M_PI * indexArr / width) + 0.08 * af::cos(2 * M_PI * indexArr / width);
      af::array denom = indexArr - width;
      // compute sinc with proper process for index = width
      af::array kernel = af::sin(2 * M_PI * cutoff_[fidx] * (indexArr - width));
      kernel(denom != 0) = kernel(denom != 0) / denom(denom != 0);
      kernel(denom == 0) = 2 * M_PI * cutoff_[fidx];
      kernel = kernel * blackmanWindow;
      // normalize kernel
      kernel = kernel / af::tile(af::sum(kernel), 2 * width + 1);
      // create low pass filter
      auto filter = std::make_shared<Conv2D>(Variable(kernel, false), 1, 1, PaddingMode::SAME, 0);
      filter->eval();
      lowPassFilters_.push_back(filter);
    }
    if (ignoredLowPassFilters_ >= lowPassFilters_.size()) {
      throw std::invalid_argument("All low pass filters are ignored, too huge kernel for all frequencies");
    }
  }
}

Variable SpecAugment::forward(const Variable& input) {
  if (input.isCalcGrad()) {
    throw std::invalid_argument(
        "input gradient calculation is not supported for SpecAugment.");
  }

  auto output = Variable(input.array(), false);
  if (!train_) {
    return output;
  }

  double replaceVal = (maskStrategy_ == MaskingStrategy::GLOBAL_MEAN)
      ? af::mean<double>(input.array())
      : 0.0;

  if (useRawWav_) {
    rawWavPrecompute();
    // input is expected T x C x B (mostly C=1)
    af::dim4 timeView = af::dim4(input.dims(0), input.dims(1) * input.dims(2) * input.dims(3));
    auto inputForFilter = fl::Variable(af::moddims(input.array(), timeView), false);
    for (int i = 0; i < numFreqMask_; ++i) {
      auto low = generateRandomInt(ignoredLowPassFilters_, rawWavNMels_);
      auto high = generateRandomInt(low, std::min(rawWavNMels_, low + freqMaskF_) + 1);
      if (high > low) {
        auto midLowWav = lowPassFilters_[high]->forward(inputForFilter);
        auto lowWav = lowPassFilters_[low]->forward(inputForFilter);
        output = output - fl::moddims(midLowWav - lowWav, input.dims());
      }
    }
  }

  auto& opArr = output.array();
  if (!useRawWav_) {
    auto numFreqChans = input.dims(1); // number of frequency channels
    if (numFreqChans < freqMaskF_) {
      throw std::runtime_error("Invalid input frequency channels");
    }
    for (int i = 0; i < numFreqMask_; ++i) {
      auto f = generateRandomInt(0, freqMaskF_);
      auto f0 = generateRandomInt(0, numFreqChans - f);
      opArr(af::span, af::seq(f0, f0 + f), af::span, af::span) = replaceVal;
    }
  }

  auto numTimeSteps = input.dims(0); // number of time steps
  // an upper bound on the time mask
  int T = std::min(timeMaskT_, static_cast<int>(numTimeSteps * timeMaskP_));
  if (T > 0) {
    for (int i = 0; i < numTimeMask_; ++i) {
      auto t = generateRandomInt(0, T);
      auto t0 = generateRandomInt(0, numTimeSteps - t);
      opArr(af::seq(t0, t0 + t), af::span, af::span, af::span) = replaceVal;
    }
  }
  return output;
}

int SpecAugment::generateRandomInt(int low, int high) {
  std::uniform_int_distribution<int> uniformDist(low, high - 1);
  return uniformDist(eng_);
}

std::string SpecAugment::prettyString() const {
  std::ostringstream ss;
  ss << "SpecAugment ( ";
  ss << "W: " << timeWarpW_ << ", ";
  ss << "F: " << freqMaskF_ << ", ";
  ss << "mF: " << numFreqMask_ << ", ";
  ss << "T: " << timeMaskT_ << ", ";
  ss << "p: " << timeMaskP_ << ", ";
  ss << "mT: " << numTimeMask_ << ", ";
  ss << "useRawWav: " << useRawWav_ << ", ";
  ss << "rawWavNMels: " << rawWavNMels_ << ", ";
  ss << "rawWavLowFreqHz: " << rawWavLowFreqHz_ << ", ";
  ss << "rawWavHighFreqHz: " << rawWavHighFreqHz_ << ", ";
  ss << "rawWavSampleRate: " << rawWavSampleRate_ << ", ";
  ss << "maxKernelSize: " << maxKernelSize_ << ", ";
  ss << " )";
  return ss.str();
}
} // namespace fl
