/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/audio/feature/Dct.h"

#include <cmath>
#include <cstddef>
#include <numeric>

#include "flashlight/lib/audio/feature/SpeechUtils.h"

namespace fl {
namespace lib {
namespace audio {

Dct::Dct(int numfilters, int numceps)
    : numFilters_(numfilters),
      numCeps_(numceps),
      dctMat_(numfilters * numceps) {
  for (size_t f = 0; f < numFilters_; ++f) {
    for (size_t c = 0; c < numCeps_; ++c) {
      dctMat_[f * numCeps_ + c] = std::sqrt(2.0 / numFilters_) *
          std::cos(M_PI * c * (f + 0.5) / numFilters_);
    }
  }
}

std::vector<float> Dct::apply(const std::vector<float>& input) const {
  return cblasGemm(input, dctMat_, numCeps_, numFilters_);
}
} // namespace audio
} // namespace lib
} // namespace fl
