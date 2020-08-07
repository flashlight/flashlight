/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/libraries/audio/feature/Dither.h"

#include <time.h>

namespace fl {
namespace lib {
namespace audio {

Dither::Dither(float ditherVal)
    : ditherVal_(ditherVal), rng_((ditherVal > 0.0) ? 123456 : time(nullptr)){};

std::vector<float> Dither::apply(const std::vector<float>& input) {
  auto output(input);
  applyInPlace(output);
  return output;
}

void Dither::applyInPlace(std::vector<float>& input) {
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  for (auto& i : input) {
    i += ditherVal_ * distribution(rng_);
  }
}
} // namespace audio
} // namespace lib
} // namespace fl
