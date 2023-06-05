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

// Compute Discrete Cosine Transform
//    c(i) = sqrt(2/N)  SUM_j (m(j) * cos(pi * i * (j - 0.5)/ N))
//      where j in [1, N], m - log filterbank amplitudes

class Dct {
 public:
  Dct(int numfilters, int numceps);

  std::vector<float> apply(const std::vector<float>& input) const;

 private:
  int numFilters_; // Number of filterbank channels
  int numCeps_; // Number of cepstral coefficients
  std::vector<float> dctMat_; // Dct matrix
};
} // namespace audio
} // namespace lib
} // namespace fl
