/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/augmentation/TimeStretch.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>

#include <sox.h>

#include "flashlight/pkg/speech/augmentation/SoxWrapper.h"

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

TimeStretch::TimeStretch(
    const TimeStretch::Config& config,
    unsigned int seed /* =0 */)
    : conf_(config),
      rng_(seed),
      sox_(SoxWrapper::instance(config.sampleRate_)) {
  FL_SOX_CHECK(stretchEffect_ = sox_find_effect("stretch"));
}

void TimeStretch::apply(std::vector<float>& signal) {
  if (rng_.random() >= conf_.proba_) {
    return;
  }
  const float factor = rng_.uniform(conf_.minFactor_, conf_.maxFactor_);
  sox_effect_t* e = sox_create_effect(stretchEffect_);
  char* args[] = {(char*)std::to_string(factor).c_str()};
  FL_SOX_CHECK(sox_effect_options(e, 1, args));
  sox_->applyAndFreeEffect(signal, e);
}

std::string TimeStretch::Config::prettyString() const {
  std::stringstream ss;
  ss << "TimeStretch::Config{minFactor_=" << minFactor_
     << " maxFactor_=" << maxFactor_ << " proba_=" << proba_
     << " sampleRate_=" << sampleRate_ << '}';
  return ss.str();
}

std::string TimeStretch::prettyString() const {
  std::stringstream ss;
  ss << "TimeStretch{config={" << conf_.prettyString() << '}';
  return ss.str();
};

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
