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
 * Input is a raw wave, specAug frequency masking is implemented with the low
 * pass filter for example
 * https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
 * For frequency augmentation we emulate masking as it is done with filterbanks.
 * nMel param is used to create nMel bins in the frequency domain, and then we
 * mask these bins. bins are created on [lowFreqHz, highFreqHz] range. Time mask
 * is measured in number of input frames: there are sampleRate frames in 1s
 * audio, e.g. 50 frames for time masking of standard specAug corresponds to
 *8000 frames (in case of 16kHz audio) for time masking with raw wave specaug
 **/
class RawWavSpecAugment : public UnaryModule {
 public:
  enum class MaskingStrategy {
    ZERO = 0,
    GLOBAL_MEAN = 1,
    // TODO - add support for mean along time, freq axes
  };

  RawWavSpecAugment(
      int tWarpW,
      int fMaskF,
      int nFMask,
      int tMaskT,
      float tMaskP,
      int nTMask,
      int nMels = 80,
      int lowFreqHz = 0,
      int highFreqHz = 8000,
      int sampleRate = 16000,
      int maxKernelSize = 20000,
      MaskingStrategy mStrategy = MaskingStrategy::ZERO);

  Variable forward(const Variable& input) override;
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

  int rawWavNMels_;
  int rawWavLowFreqHz_;
  int rawWavHighFreqHz_;
  int rawWavSampleRate_;
  int maxKernelSize_;
  int ignoredLowPassFilters_;
  std::vector<float> cutoff_;
  std::vector<std::shared_ptr<Conv2D>> lowPassFilters_;

  int generateRandomInt(int low, int high);

  void precomputeFilters();

  af::array lowPassFilter(int freq, af::array wav);

  RawWavSpecAugment() = default;

  FL_SAVE_LOAD_DECLARE()
};

template <class Archive>
void RawWavSpecAugment::save(Archive& ar, const uint32_t /* version */) const {
  ar(cereal::base_class<Module>(this),
     timeWarpW_,
     freqMaskF_,
     numFreqMask_,
     timeMaskT_,
     timeMaskP_,
     numTimeMask_,
     maskStrategy_,
     rawWavNMels_,
     rawWavLowFreqHz_,
     rawWavHighFreqHz_,
     rawWavSampleRate_,
     maxKernelSize_);
}

template <class Archive>
void RawWavSpecAugment::load(Archive& ar, const uint32_t /* version */) {
  ar(cereal::base_class<Module>(this),
     timeWarpW_,
     freqMaskF_,
     numFreqMask_,
     timeMaskT_,
     timeMaskP_,
     numTimeMask_,
     maskStrategy_,
     rawWavNMels_,
     rawWavLowFreqHz_,
     rawWavHighFreqHz_,
     rawWavSampleRate_,
     maxKernelSize_);
  precomputeFilters();
}

} // namespace fl

CEREAL_REGISTER_TYPE(fl::RawWavSpecAugment)
