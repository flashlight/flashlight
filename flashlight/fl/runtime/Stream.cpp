/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/Stream.h"

namespace fl {

void Stream::relativeSync(
  const std::unordered_set<const Stream*>& waitOns) const {
  for (const auto* waitOn : waitOns) {
    this->relativeSync(*waitOn);
  }
}

} // namespace fl
