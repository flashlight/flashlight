/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include <flashlight/flashlight.h>

namespace fl {
namespace task {
namespace asr {

// Convenience structs for serializing emissions and targets
struct EmissionUnit {
  std::vector<float> emission; // A column-major tensor with shape T x N.
  std::string sampleId;
  int nFrames;
  int nTokens;

  FL_SAVE_LOAD(emission, sampleId, nFrames, nTokens)

  EmissionUnit() : nFrames(0), nTokens(0) {}

  EmissionUnit(
      const std::vector<float>& emission,
      const std::string& sampleId,
      int nFrames,
      int nTokens)
      : emission(emission),
        sampleId(sampleId),
        nFrames(nFrames),
        nTokens(nTokens) {}
};

struct TargetUnit {
  std::vector<std::string> wordTargetStr; // Word targets in strings
  std::vector<int> tokenTarget; // Token targets in indices

  FL_SAVE_LOAD(wordTargetStr, tokenTarget)
};

using EmissionTargetPair = std::pair<EmissionUnit, TargetUnit>;
} // namespace asr
} // namespace task
} // namespace fl