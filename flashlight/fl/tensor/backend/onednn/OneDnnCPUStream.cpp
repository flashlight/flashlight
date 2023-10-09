/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/onednn/OneDnnCPUStream.h"

#include <stdexcept>

namespace fl {

OneDnnCPUStream::OneDnnCPUStream(const dnnl::engine& engine) {
  stream_ = std::make_unique<dnnl::stream>(engine);
}

std::shared_ptr<OneDnnCPUStream> OneDnnCPUStream::create(
    const dnnl::engine& engine) {
  if (engine.get_kind() != dnnl::engine::kind::cpu) {
    throw std::invalid_argument("OneDnnCPUStream expects a CPU engine");
  }
  const auto rawStreamPtr = new OneDnnCPUStream(engine);
  const auto stream = std::shared_ptr<OneDnnCPUStream>(rawStreamPtr);
  rawStreamPtr->device_.addStream(stream);
  return stream;
}

void OneDnnCPUStream::sync() const {
  stream_->wait();
}

dnnl::stream& OneDnnCPUStream::handle() {
  return *stream_;
}

const dnnl::stream& OneDnnCPUStream::handle() const {
  return *stream_;
}

} // namespace fl
