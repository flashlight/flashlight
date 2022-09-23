/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/trace/TracerBase.h"

#include <utility>

namespace fl {

TracerBase::TracerBase(std::unique_ptr<std::ostream> stream)
    : ostream_(std::move(stream)) {}

void TracerBase::enableTracer(bool val) {
  tracingEnabled_ = val;
}

bool TracerBase::tracingEnabled() const {
  return tracingEnabled_;
}

const std::ostream& TracerBase::getStream() const {
  if (ostream_) {
    return *ostream_;
  } else {
    throw std::runtime_error("TracerBase::getStream() - stream is null.");
  }
}

std::ostream& TracerBase::getStream() {
  if (ostream_) {
    return *ostream_;
  } else {
    throw std::runtime_error("TracerBase::getStream() - stream is null.");
  }
}

void TracerBase::setStream(std::unique_ptr<std::ostream> stream) {
  ostream_ = std::move(stream);
}

} // namespace fl
