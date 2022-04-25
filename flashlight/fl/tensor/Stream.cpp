/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Stream.h"

namespace fl {

std::packaged_task<void()> Stream::sync() const {
  return impl_->sync();
}

StreamType Stream::type() const {
  return impl_->type();
}

} // namespace fl
