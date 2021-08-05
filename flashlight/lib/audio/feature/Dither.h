/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>
#include <vector>

namespace fl {
namespace lib {
namespace audio {

// Dither the signal by adding small amount of random noise to the signal
//    s'(n) = s(n) + q * RND()  where RND() is uniformly distributed in [-1, 1)
//      and `q` is the dithering constant
// Similar to HTK, positive value of `q` causes the same noise signal to be
// added everytime and with negative value of `q`, noise is random and the same
// file may produce slightly different results in different trials

class Dither {
 public:
  explicit Dither(float ditherVal);

  std::vector<float> apply(const std::vector<float>& input);

  void applyInPlace(std::vector<float>& input);

 private:
  float ditherVal_;
  std::mt19937 rng_; // Standard mersenne_twister_engine
};
} // namespace audio
} // namespace lib
} // namespace fl
