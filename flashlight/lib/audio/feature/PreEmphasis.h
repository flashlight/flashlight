/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

namespace fl {
namespace lib {
namespace audio {

// Pre-emphasise the signal by applying the first order difference equation
//    s'(n) = s(n) - k * s(n-1)  where k in [0, 1)

class PreEmphasis {
 public:
  PreEmphasis(float alpha, int N);

  std::vector<float> apply(const std::vector<float>& input) const;

  void applyInPlace(std::vector<float>& input) const;

 private:
  float preemCoef_;
  int windowLength_;
};
} // namespace audio
} // namespace lib
} // namespace fl
