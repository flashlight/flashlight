/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <memory>
#include <stdexcept>
#include <utility>

namespace fl {

// TODO{runtime} - have a pointer back to a device to manage this
/**
 * A runtime type for various stream types.
 */
enum class StreamType { CUDA };

struct StreamImpl {
  virtual ~StreamImpl() = default;
  virtual std::packaged_task<void()> sync() const = 0;
  virtual StreamType type() const = 0;
};

/**
 * A stub of a interface for a computation stream. NB: this is a stub of an
 * abstraction that will be more comprehensively-defined later and eventually
 * moved into a small Flashlight runtime.
 */
class Stream {
  std::unique_ptr<StreamImpl> impl_;

 public:
  explicit Stream(std::unique_ptr<StreamImpl> impl) : impl_(std::move(impl)) {}
  ~Stream() = default;

  /**
   * Block the calling thread until all computation enqueued to the stream at
   * calling time is complete.
   */
  std::packaged_task<void()> sync() const;

  /**
   * Return a runtime stream type for this stream.
   */
  StreamType type() const;

  /**
   * Get the underlying StreamImpl for this stream. Throws if the specified type
   * does not match the underlying stream's StreamType.
   */
  template <typename T>
  const T& impl() const {
    if (T::streamType != type()) {
      throw std::invalid_argument(
          "fl::Stream::impl() - given stream type doesn't match "
          "impl stream type.");
    }
    return *(static_cast<T*>(impl_.get()));
  }

  // No copies allowed, only moves
  Stream(const Stream& s) = delete;
  Stream& operator=(const Stream& s) = delete;
};

} // namespace fl
