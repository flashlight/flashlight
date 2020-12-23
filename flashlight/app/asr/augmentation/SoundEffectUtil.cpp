/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/augmentation/SoundEffectUtil.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fl {
namespace app {
namespace asr {
namespace sfx {

std::string randomPolicyToString(RandomPolicy policy) {
  switch (policy) {
    case RandomPolicy::WITH_REPLACEMENT:
      return "with_replacment";
    case RandomPolicy::WITHOUT_REPLACEMENT:
      return "without_replacment";
    default:
      throw std::invalid_argument("PolicyToString() invalid policy");
  }
}

RandomPolicy stringToRandomPolicy(const std::string& policy) {
  if (policy == "with_replacment") {
    return RandomPolicy::WITH_REPLACEMENT;
  } else if (policy == "without_replacment") {
    return RandomPolicy::WITHOUT_REPLACEMENT;
  } else {
    throw std::invalid_argument("StringToPolicy() invalid policy=" + policy);
  }
}

RandomNumberGenerator::RandomNumberGenerator(int seed /* = 0 */)
    : randomEngine_(seed), uniformDist_(0, 1) {}

int RandomNumberGenerator::randInt(int a, int b) {
  if (a > b) {
    std::swap(a, b);
  }
  return randomEngine_() % (b - a + 1) + a;
}

float RandomNumberGenerator::random() {
  return uniformDist_(randomEngine_);
}

float RandomNumberGenerator::uniform(float a, float b) {
  return a + (b - a) * uniformDist_(randomEngine_);
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
