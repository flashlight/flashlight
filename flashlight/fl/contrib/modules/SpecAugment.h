/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>

#include "flashlight/fl/nn/nn.h"

namespace fl {

/**
 * Implementation of SpecAugment: A Simple Data Augmentation Method
 * for Automatic Speech Recognition - https://arxiv.org/pdf/1904.08779.pdf
 *
 * We assume time axis is the 0th dimension, and freq axis is the 1st dimension
 * for the  input array
 *
 * Example policies        tWarpW    fMaskF    nFMask tMaskT    tMaskP   nTMask
 * LibriSpeech basic (LB)    80        27        1     100       1.0       1
 * LibriSpeech double (LD)   80        27        2     100       1.0       2
 * Switchboard mild (SM)     40        15        2      70       0.2       2
 * Switchboard strong (SS)   40        27        2      70       0.2       2
 *
 * Raw wave specAug is implemented with the low pass filter, for example
 * https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
 **/
struct RawWavSpecAugmentConfig {
  bool useRawWav;
  int nMels;
  int lowFreqHz;
  int highFreqHz;
  int sampleRate;
  int maxKernelSize;
};

class SpecAugment : public UnaryModule {
 public:
  enum class MaskingStrategy {
    ZERO = 0,
    GLOBAL_MEAN = 1,
    // TODO - add support for mean along time, freq axes
  };

  SpecAugment(
      int tWarpW,
      int fMaskF,
      int nFMask,
      int tMaskT,
      float tMaskP,
      int nTMask,
      RawWavSpecAugmentConfig rawWaveConfig,
      MaskingStrategy mStrategy = MaskingStrategy::ZERO
      );

  Variable forward(const Variable& input) override;

  FL_SAVE_LOAD_WITH_BASE(
      UnaryModule,
      timeWarpW_,
      freqMaskF_,
      numFreqMask_,
      timeMaskT_,
      timeMaskP_,
      numTimeMask_,
      maskStrategy_,
      fl::versioned(useRawWav_, 1),
      fl::versioned(rawWavNMels_, 1),
      fl::versioned(rawWavLowFreqHz_, 1),
      fl::versioned(rawWavHighFreqHz_, 1),
      fl::versioned(rawWavSampleRate_, 1),
      fl::versioned(maxKernelSize_, 1))

  std::string prettyString() const override;

 private:
  // Time Warping - NOT SUPPORTED CURRENTLY
  //  Use timeWarpW_ = 0 to disable this
  int timeWarpW_;

  // Frequency Masking
  //  Use freqMaskF_ = 0 to disable this
  int freqMaskF_;
  int numFreqMask_;

  // Time Masking
  //  Use timeMaskT_ = 0 to disable this
  int timeMaskT_;
  float timeMaskP_;
  int numTimeMask_;

  std::mt19937 eng_{0};
  MaskingStrategy maskStrategy_;

  // Raw wave input
  bool useRawWav_{false};
  int rawWavNMels_{80};
  int rawWavLowFreqHz_{0};
  int rawWavHighFreqHz_{8000};
  int rawWavSampleRate_{16000};
  int maxKernelSize_{20000};
  int ignoredLowPassFilters_;
  std::vector<float> cutoff_;
  std::vector<std::shared_ptr<Conv2D>> lowPassFilters_;

  int generateRandomInt(int low, int high);

  void rawWavPrecompute();

  af::array lowPassFilter(int freq, af::array wav);

  SpecAugment() = default;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::SpecAugment)
CEREAL_CLASS_VERSION(fl::SpecAugment, 1)
