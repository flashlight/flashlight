/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

#include "flashlight/lib/audio/feature/FeatureParams.h"

namespace fl {
namespace lib {
namespace audio {

// Applies a given window on input
//    s'(n) = w(n) * s(n) where w(n) are the window coefficients

class Windowing {
 public:
  Windowing(int N, WindowType window);

  std::vector<float> apply(const std::vector<float>& input) const;

  void applyInPlace(std::vector<float>& input) const;

 private:
  int windowLength_;
  WindowType windowType_;
  std::vector<float> coefs_;
};
} // namespace audio
} // namespace lib
} // namespace fl
