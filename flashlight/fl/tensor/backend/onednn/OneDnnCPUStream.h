/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "flashlight/fl/runtime/SynchronousStream.h"

#include <dnnl.hpp>

namespace fl {

/**
 * An abstraction for OneDNN's CPU Stream with controlled creation methods.
 */
class OneDnnCPUStream : public SynchronousStream {
  std::unique_ptr<dnnl::stream> stream_; // stored as a pointer to satisfy `sync() const`

  // internal constructor used to create the native OneDNN stream.
  explicit OneDnnCPUStream(const dnnl::engine& engine);

 public:
  /**
   * Creates an OneDnnCPUStream on given engine and automatically register it
   * with the active x64 device from DeviceManager.
   *
   * @param[in] engine is the cpu engine on which the stream will be created.
   * @return a shared pointer to the created OneDnnCPUStream.
   * @throws invalid_argument if given engine is not a CPU engine.
   */
  static std::shared_ptr<OneDnnCPUStream> create(const dnnl::engine& engine);

  void sync() const override;

  /**
   * Gets the underlying OneDNN stream.
   *
   * @return the underlying OneDNN stream.
   */
  const dnnl::stream& handle() const;

  /**
   * Gets the underlying OneDNN stream.
   *
   * @return the underlying OneDNN stream.
   */
  dnnl::stream& handle();
};
} // namespace fl
