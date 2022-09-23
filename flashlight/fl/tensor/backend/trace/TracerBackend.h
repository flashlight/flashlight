/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <ostream>

#include "flashlight/fl/tensor/backend/trace/DefaultTracer.h"
#include "flashlight/fl/tensor/backend/trace/TracerBackendBase.h"

namespace fl {

/**
 * A backend that wraps another backend and provides tracing info on top of it.
 */
template <typename T>
class TracerBackend : public TracerBackendBase {
 public:
  TracerBackend() = default;
  ~TracerBackend() override = default;

  TensorBackend& backend() const override {
    // TODO: T128751983 add a static_assert here that this method exists via
    // sfinae
    return T::getInstance();
  }

  static TracerBackendBase& getInstance() {
    static TracerBackend backend;
    return backend;
  }
};

} // namespace fl
