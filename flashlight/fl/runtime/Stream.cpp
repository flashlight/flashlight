/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/Stream.h"

namespace fl {
namespace runtime {

void Stream::relativeSync(
  const std::unordered_set<const Stream*>& waitOns) const {
  for (const auto* waitOn : waitOns) {
    this->relativeSync(*waitOn);
  }
}

} // namespace runtime
} // namespace fl
