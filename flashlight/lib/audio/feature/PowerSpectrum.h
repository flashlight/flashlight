/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>

#include "flashlight/lib/audio/feature/Dither.h"
#include "flashlight/lib/audio/feature/FeatureParams.h"
#include "flashlight/lib/audio/feature/PreEmphasis.h"
#include "flashlight/lib/audio/feature/Windowing.h"

// Fwd decl
class fftw_plan_s;
typedef fftw_plan_s* fftw_plan;

namespace fl {
namespace lib {
namespace audio {

// Computes Power Spectrum features for a speech signal.

class PowerSpectrum {
 public:
  explicit PowerSpectrum(const FeatureParams& params);

  virtual ~PowerSpectrum();

  // input - input speech signal (T)
  // Returns - Power spectrum (Col Major : FEAT X FRAMESZ)
  virtual std::vector<float> apply(const std::vector<float>& input);

  // input - input speech signal (Col Major : T X BATCHSZ)
  // Returns - Output features (Col Major : FEAT X FRAMESZ X BATCHSZ)
  std::vector<float> batchApply(const std::vector<float>& input, int batchSz);

  virtual int outputSize(int inputSz);

  FeatureParams getFeatureParams() const;

 protected:
  FeatureParams featParams_;

  // Helper function which takes input as signal after dividing the signal into
  // frames. Main purpose of this function is to reuse it in MFSC, MFCC code
  std::vector<float> powSpectrumImpl(std::vector<float>& frames);

  void validatePowSpecParams() const;

 private:
  // The following classes are defined in the order they are applied
  Dither dither_;
  PreEmphasis preEmphasis_;
  Windowing windowing_;

  std::unique_ptr<fftw_plan> fftPlan_; // fftw_plan is an opque pointer type
  std::vector<double> inFftBuf_, outFftBuf_;
  std::mutex fftMutex_;
};
} // namespace audio
} // namespace lib
} // namespace fl
