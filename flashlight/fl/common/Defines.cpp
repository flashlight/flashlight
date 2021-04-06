/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <string>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Types.h"

namespace fl {

OptimLevel OptimMode::getOptimLevel() {
  return optimLevel_;
}

void OptimMode::setOptimLevel(OptimLevel level) {
  optimLevel_ = level;
}

OptimMode& OptimMode::get() {
  static OptimMode optimMode;
  return optimMode;
}

OptimLevel OptimMode::toOptimLevel(const std::string& in) {
  auto l = kStringToOptimLevel.find(in);
  if (l == kStringToOptimLevel.end()) {
    throw std::invalid_argument(
        "OptimMode::toOptimLevel - no matching "
        "optim level for given string.");
  }
  return l->second;
}

const std::unordered_map<std::string, OptimLevel>
    OptimMode::kStringToOptimLevel = {
        {"DEFAULT", OptimLevel::DEFAULT},
        {"O1", OptimLevel::O1},
        {"O2", OptimLevel::O2},
        {"O3", OptimLevel::O3},
};

} // namespace fl
