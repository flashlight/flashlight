/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/SynchronousStream.h"

namespace fl {

X64Device& SynchronousStream::device() {
  return device_;
}

const X64Device& SynchronousStream::device() const {
  return device_;
}

void SynchronousStream::relativeSync(const SynchronousStream& waitOn) const {
  waitOn.sync();
}

} // namespace fl
