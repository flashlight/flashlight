/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <future>
#include <unordered_set>

namespace fl {

class Device;

namespace runtime {

enum class StreamType {
  CUDA,
};

/**
 * An abstraction that represents a sequence of computations that must happen
 * synchronously on a specific device. It focuses on synchronization of the
 * computations, while being agnostic to the computations themselves.
 */
class Stream {
 public:
  Stream() = default;
  virtual ~Stream() = default;

  // no copy/move
  Stream(const Stream&) = delete;
  Stream(Stream&&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream& operator=(const Stream&&) = delete;

  /**
   * Get the underlying implementation of this stream.
   *
   * Throws invalid_argument if the specified type does not match the actual
   * derived stream type.
   *
   * @return an immutable reference to the specified stream type.
   */
  template <typename T>
  const T& impl() const {
    if (T::type != type()) {
      throw std::invalid_argument(
          "[fl::Stream::impl] "
          "specified stream type doesn't match actual stream type.");
    }
    return *(static_cast<const T*>(this));
  }

  /**
   * Returns the type of this stream.
   *
   * @return a enum denoting stream type.
   */
  virtual StreamType type() const = 0;

  /**
   * Return the owner device of this stream.
   *
   * @return a reference to the owner device of this stream.
   */
  virtual const Device& device() const = 0;

  /**
   * Synchronize w.r.t. all tasks on this stream.
   *
   * @return a future representing the completion of all tasks on this stream.
   */
  virtual std::future<void> sync() const = 0;

  /**
   * Synchronize future tasks on this stream w.r.t. current tasks on given
   * stream, i.e., the former can only start after the completion of the latter.
   * NOTE this function may or may not block the calling thread.
   *
   * @param[in] waitOn the stream to perform relative synchronization against.
   */
  virtual void relativeSync(const Stream& waitOn) const = 0;

  /**
   * Synchronize future tasks on this stream w.r.t. current tasks on all given
   * stream, i.e., the former can only start after the completion of the latter.
   * NOTE this function may or may not block the calling thread.
   *
   * @param[in] waitOns the streams to perform relative synchronization against.
   */
  virtual void relativeSync(
    const std::unordered_set<const Stream*>& waitOns) const;
};

/**
 * A trait for some generic stream functionalities.
 *
 * REQUIRED definition in derived class:
 *   static StreamType type;
 */
template <typename Derived>
class StreamTrait : public Stream {
 public:
  // prevent name hiding
  using Stream::relativeSync;

  // A specialized relativeSync for streams of the same type.
  virtual void relativeSync(const Derived& waitOn) const = 0;

  StreamType type() const override {
    return Derived::type;
  }

  virtual void relativeSync(const Stream& waitOn) const override {
    switch (waitOn.type()) {
      case Derived::type: relativeSync(waitOn.impl<Derived>()); break;
      default:
        throw std::runtime_error(
          "[Stream::relativeSync] Unsupported for different types of streams");
    }
  }
};

} // namespace runtime
} // namespace fl
