/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/runtime/SpeechStatMeter.h"
#include <array>

namespace fl {
namespace app {
namespace asr {
SpeechStatMeter::SpeechStatMeter() {
  reset();
}

void SpeechStatMeter::reset() {
  stats_.reset();
}

void SpeechStatMeter::add(
    const af::array& inputSizes,
    const af::array& targetSizes) {
  int64_t curInputSz = af::sum<int64_t>(inputSizes);
  int64_t curTargetSz = af::sum<int64_t>(targetSizes);

  stats_.totalInputSz_ += curInputSz;
  stats_.totalTargetSz_ += curTargetSz;

  stats_.maxInputSz_ =
      std::max(stats_.maxInputSz_, af::max<int64_t>(inputSizes));
  stats_.maxTargetSz_ =
      std::max(stats_.maxTargetSz_, af::max<int64_t>(targetSizes));

  stats_.numSamples_ += inputSizes.dims(1);
  stats_.numBatches_++;
}

void SpeechStatMeter::add(const SpeechStats& stats) {
  stats_.totalInputSz_ += stats.totalInputSz_;
  stats_.totalTargetSz_ += stats.totalTargetSz_;

  stats_.maxInputSz_ = std::max(stats_.maxInputSz_, stats.maxInputSz_);
  stats_.maxTargetSz_ = std::max(stats_.maxTargetSz_, stats.maxTargetSz_);

  stats_.numSamples_ += stats.numSamples_;
  stats_.numBatches_ += stats.numBatches_;
}

std::vector<int64_t> SpeechStatMeter::value() const {
  return stats_.toArray();
}

SpeechStats::SpeechStats() {
  reset();
}

void SpeechStats::reset() {
  totalInputSz_ = 0;
  totalTargetSz_ = 0;
  maxInputSz_ = 0;
  maxTargetSz_ = 0;
  numSamples_ = 0;
  numBatches_ = 0;
}

std::vector<int64_t> SpeechStats::toArray() const {
  std::vector<int64_t> arr(6);
  arr[0] = totalInputSz_;
  arr[1] = totalTargetSz_;
  arr[2] = maxInputSz_;
  arr[3] = maxTargetSz_;
  arr[4] = numSamples_;
  arr[5] = numBatches_;
  return arr;
}
} // namespace asr
} // namespace app
} // namespace fl
