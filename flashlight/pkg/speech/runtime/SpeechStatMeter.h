/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace pkg {
namespace speech {

struct SpeechStats {
  int64_t totalInputSz_;
  int64_t totalTargetSz_;
  int64_t maxInputSz_;
  int64_t maxTargetSz_;
  int64_t numSamples_;
  int64_t numBatches_;

  SpeechStats();
  void reset();
  std::vector<int64_t> toArray() const;
};

class SpeechStatMeter {
 public:
  SpeechStatMeter();
  void add(const af::array& inputSizes, const af::array& targetSizes);
  void add(const SpeechStats& stats);
  std::vector<int64_t> value() const;
  void reset();

 private:
  SpeechStats stats_;
};
} // namespace speech
} // namespace pkg
} // namespace fl
