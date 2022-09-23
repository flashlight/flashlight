/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>
#include <ostream>
#include <utility>

#include "flashlight/fl/tensor/backend/trace/DefaultTracer.h"
#include "flashlight/fl/tensor/backend/trace/TracerBackendBase.h"

namespace fl {

/**
 * A backend that wraps another backend and provides tracing info on top of it.
 */
template <typename T, typename U>
class TracerBackend : public TracerBackendBase {
  static inline std::once_flag createSingleton_;

 public:
  // TracerBackend(TensorCreatorFunc f) : TracerBackendBase(std::move(f)) {}
  TracerBackend() = default;
  ~TracerBackend() override = default;

  TensorBackend& backend() const override {
    // TODO: T128751983 add a static_assert here that this method exists via
    // sfinae
    return U::getInstance();
  }

  static inline TracerBackendBase& getInstance() {
    std::call_once(createSingleton_, []() {
      instance_ = std::make_unique<TracerBackend<T, U>>();
    });
    return TracerBackendBase::getInstance();
  }
};

} // namespace fl
