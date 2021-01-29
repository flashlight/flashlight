/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/augmentation/ReverbAndNoise.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

#include <glog/logging.h>
#include <mkl.h>

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"
#include "flashlight/app/asr/data/Sound.h"
#include "flashlight/lib/mkl/Conv.h"

namespace fl {
namespace app {
namespace asr {
namespace sfx {

ReverbAndNoise::ReverbAndNoise(
    const ReverbAndNoise::Config& conf,
    unsigned int seed /* = 0 */)
    : conf_(conf), rng_(seed) {
  rirFiles_ = loadListFile(conf.rirListFilePath_, "ReverbAndNoise");
  noiseFiles_ = loadListFile(conf.noiseListFilePath_, "ReverbAndNoise");
}

namespace {
/*
    Early reverberation component of the signal is composed of reflections
    within 0.05 seconds of the direct path signal (assumed to be the peak of
    the room impulse response). This function returns the energy in
    this early reverberation component of the signal.
    The input parameters to this function are the room impulse response, the
   signal and their sampling frequency respectively.
 */
float ComputeEarlyReverbEnergy(
    const std::vector<float>& rir,
    std::vector<float>& signal,
    float sampleRate) {
  auto maxItr = std::max_element(rir.begin(), rir.end());
  auto maxIdx = std::distance(rir.begin(), maxItr);
  int peak_index = rir[maxIdx];

  LOG(INFO) << "peak index is " << peak_index;

  const float sec_before_peak = 0.001;
  const float sec_after_peak = 0.05;
  int early_rir_start_index = peak_index - sec_before_peak * sampleRate;
  int early_rir_end_index = peak_index + sec_after_peak * sampleRate;
  if (early_rir_start_index < 0)
    early_rir_start_index = 0;
  if (early_rir_end_index > rir.size())
    early_rir_end_index = rir.size();

  int duration = early_rir_end_index - early_rir_start_index;
  std::vector<float> early_rir(
      rir.begin() + early_rir_start_index,
      rir.begin() + early_rir_start_index + duration);

  // TODO: padding signal with duration size)
  auto early_reverb = fl::lib::mkl::conv1D(early_rir, signal);

  return dotProduct(early_reverb, early_reverb) / early_reverb.size();
}

/**
 * This function is to add signal1 to signal2 starting at the offset of signal2
 * This will not extend the length of signal2.
 */
void AddVectorsWithOffset(
    const std::vector<float>& signal1,
    int offset,
    std::vector<float>& signal2) {
  const int addLength = std::min(signal2.size() - offset, signal1.size());
  for (int i = 0; i < addLength; ++i) {
    signal2[offset + i] += signal1[i];
  }
}

/*
   The noise will be scaled before the addition
   to match the given signal-to-noise ratio (SNR).
*/
void AddNoise(
    std::vector<float>& noise,
    float snrDb,
    float signalPower,
    std::vector<float>& signal) {
  float noisePower = dotProduct(noise, noise) / noise.size();
  if (noisePower == 0) {
    return;
  }
  float scaleFactor =
      std::sqrt(pow(10, -snrDb / 10) * signalPower / noisePower);

  LOG(INFO) << " signal.size()=" << signal.size()
            << " signalPower=" << signalPower
            << " noise.size()=" << noise.size() << " snrDb=" << snrDb
            << " noisePower=" << noisePower << " scaleFactor=" << scaleFactor;

  for (int i = 0; i < signal.size(); ++i) {
    signal[i] += (noise[i % (noise.size() - 1)] * scaleFactor);
  }
}

/*
   This is the core function to do reverberation on the given signal.
   The input parameters to this function are the room impulse response,
   the sampling frequency and the signal respectively.
   The length of the signal will be extended to (original signal length +
   rir length - 1) after the reverberation.
*/
float DoReverberation(
    std::vector<float>& rir,
    std::vector<float>& signal,
    float sampleRate) {
  float signalPower = ComputeEarlyReverbEnergy(rir, signal, sampleRate);

  const size_t inputSize = signal.size();
  const size_t outputSize = inputSize + rir.size() - 1;

  const size_t pad = rir.size();
  std::vector<float> paddedSignal(inputSize + pad * 2, 0);
  for (int i = 0; i < inputSize; ++i) {
    paddedSignal[pad + i] = signal[i];
  }

  auto augmented = fl::lib::mkl::conv1D(rir, paddedSignal);
  // signal = augmented without the left (past) padding.
  signal = std::vector<float>(outputSize, 0);
  for (int i = 0; i < signal.size(); ++i) {
    signal[i] = augmented[pad + i];
  }

  return signalPower;
}

} // namespace

// implemented as at https://kaldi-asr.org/doc/wav-reverberate_8cc_source.html
void ReverbAndNoise::apply(std::vector<float>& signal) {
  if (signal.empty() || rng_.random() >= conf_.proba_) {
    return;
  }
  const size_t origSize = signal.size();
  const float powerBeforeReverb = dotProduct(signal, signal) / origSize;

  auto curRirFileIdx = rng_.randInt(0, rirFiles_.size() - 1);
  auto rir = loadSound<float>(rirFiles_[curRirFileIdx]);
  float earlyEnergy = powerBeforeReverb;
  LOG(INFO) << "powerBeforeReverb=" << powerBeforeReverb;
  int shift_index = 0;
  if (!rir.empty()) {
    earlyEnergy = DoReverberation(rir, signal, conf_.sampleRate_);

    // if (shift_output) {
    //   // find the position of the peak of the impulse response
    //   // and shift the output waveform by this amount
    //   rir.Max(&shift_index);
    // }
  }
  LOG(INFO) << "earlyEnergy=" << earlyEnergy;

  const int nClips = rng_.randInt(conf_.nClipsMin_, conf_.nClipsMax_);
  for (int i = 0; i < nClips; ++i) {
    const auto curNoiseFileIdx = rng_.randInt(0, noiseFiles_.size() - 1);
    auto curNoise = loadSound<float>(noiseFiles_[curNoiseFileIdx]);
    if (!curNoise.empty()) {
      const float snrDb = rng_.uniform(conf_.minSnr_, conf_.maxSnr_);
      // Note that we choose SNR per noise.
      AddNoise(curNoise, snrDb, earlyEnergy, signal);
    }
  }

  float scaleFactor = conf_.volume_;
  if (scaleFactor <= 0) {
    const float powerAfterRevereb =
        dotProduct(signal, signal, origSize) / origSize;
    scaleFactor = std::sqrt(powerBeforeReverb / powerAfterRevereb);
    LOG(INFO) << " scaleFactor=" << scaleFactor
              << " powerAfterRevereb=" << powerAfterRevereb
              << " (powerBeforeReverb / powerAfterRevereb)="
              << (powerBeforeReverb / powerAfterRevereb);
  }
  scale(signal, scaleFactor);
  const float outputPower = dotProduct(signal, signal, origSize) / origSize;
  LOG(INFO) << "orig power=" << powerBeforeReverb
            << " output power=" << scaleFactor;
}

std::string ReverbAndNoise::prettyString() const {
  return "ReverbAndNoise{conf_=" + conf_.prettyString() + "}}";
}

std::string ReverbAndNoise::Config::prettyString() const {
  std::stringstream ss;
  // ss << " proba_=" << proba_ << " initialMin_=" << initialMin_
  //    << " initialMax_=" << initialMax_ << " rt60Min_=" << rt60Min_
  //    << " rt60Max_=" << rt60Max_ << " firstDelayMin_=" << firstDelayMin_
  //    << " firstDelayMax_=" << firstDelayMax_ << " repeat_=" << repeat_
  //    << " jitter_=" << jitter_ << " sampleRate_=" << sampleRate_;
  return ss.str();
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
