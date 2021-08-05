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

// Re-scale the cepstral coefficients using liftering
//    c'(n) = c(n) * (1 + 0.5 * L * sin(pi * n/ L)) where L is lifterparam

class Ceplifter {
 public:
  Ceplifter(int numfilters, int lifterparam);

  std::vector<float> apply(const std::vector<float>& input) const;

  void applyInPlace(std::vector<float>& input) const;

 private:
  int numFilters_; // number of filterbank channels
  int lifterParam_; // liftering parameter
  std::vector<float> coefs_; // coefficients to scale cepstral coefficients
};
} // namespace audio
} // namespace lib
} // namespace fl
