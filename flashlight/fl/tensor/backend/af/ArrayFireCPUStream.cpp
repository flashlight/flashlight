/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireCPUStream.h"

#include <af/device.h>

namespace fl {

std::shared_ptr<ArrayFireCPUStream> ArrayFireCPUStream::create() {
  // TODO `std::make_shared` requires a public constructor, which could be
  // abused and lead to unregistered stream. However, it has one internal
  // allocation and is more cache-friendly than `std::shared_ptr`.
  const auto rawStreamPtr = new ArrayFireCPUStream();
  const auto stream = std::shared_ptr<ArrayFireCPUStream>(rawStreamPtr);
  rawStreamPtr->device_.addStream(stream);
  return stream;
}

void ArrayFireCPUStream::sync() const {
  af::sync(af::getDevice());
}

} // namespace fl
