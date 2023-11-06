/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/memory.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "flashlight/fl/tensor/backend/af/mem/MemoryManagerDeviceInterface.h"

namespace fl {

namespace {

const size_t kDefaultLogFlushInterval = 50;

} // namespace

/**
 * An interface for defining memory managers purely in C++.
 *
 * The ArrayFire memory management API is defined using C callbacks; this class
 * provides a thin layer of abstraction over this callbacks and acts as an
 * adapter between derived C++ class implementations and the ArrayFire C API. In
 * particular:
 * - Each instance has an associated af_memory_manager whose payload is a
 *   pointer to `this` which allows callbacks to call C++ class methods after
 *   casting.
 * - Provides logging functions and a logging mode which logs all function calls
 *   from ArrayFire and all relevant arguments. Only virtual base class methods
 *   that have derived implementations are eligible for logging.
 * - The `MemoryManagerInstaller` provides an interface for setting implemented
 *   memory managers as the active ArrayFire memory managers by setting relevant
 *   callbacks on construction.
 *
 * For documentation of virtual methods, see [ArrayFire's memory
 * header](https://git.io/Jv7do) for full specifications on when these methods
 * are called by ArrayFire and the JIT.
 */
class MemoryManagerAdapter {
 public:
  /**
   * Constructs a MemoryManagerAdapter.
   *
   * @param[in] deviceInterface a pointer to a `MemoryManagerDeviceInterface`.
   * Function pointers on the interface will be defined once the memory manager
   * is installed.
   * @param[in] logStream a pointer to an output stream to use for logging. All
   * function calls to overridden base class methods by ArrayFire will be logged
   * to the resulting stream in conjunction with passed arguments. If a valid
   * output stream is passed, the memory manager will initialize with logging
   * enabled. This argument is optional - passing no argument disables logging
   * for the memory manager by default.
   */
  explicit MemoryManagerAdapter(
      std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface,
      std::ostream* logStream = nullptr);
  virtual ~MemoryManagerAdapter();

  // Standard API methods - see ArrayFire's af/memory.h header for docs.
  virtual void initialize() = 0;
  virtual void shutdown() = 0;
  virtual void* alloc(
      bool userLock,
      const unsigned ndims,
      dim_t* dims,
      const unsigned elSize) = 0;
  virtual size_t allocated(void* ptr) = 0;
  virtual void unlock(void* ptr, bool userLock) = 0;
  virtual void signalMemoryCleanup() = 0;
  virtual void printInfo(
      const char* msg,
      const int device,
      std::ostream* ostream = &std::cout) = 0;
  virtual void userLock(const void* ptr) = 0;
  virtual void userUnlock(const void* ptr) = 0;
  virtual bool isUserLocked(const void* ptr) = 0;
  virtual float getMemoryPressure() = 0;
  virtual bool jitTreeExceedsMemoryPressure(size_t bytes) = 0;
  virtual void addMemoryManagement(int device) = 0;
  virtual void removeMemoryManagement(int device) = 0;

  virtual size_t getMemStepSize();
  virtual void setMemStepSize(size_t size);

  /**
   * Logs information to the `MemoryManagerAdapters`'s log stream. If logging
   * mode is enabled, function calls to virtual base class methods are logged.
   *
   * @param[in] fname the name of the function to be logged (or some arbitrary
   * prefix string)
   * @param[in] vs variadic list of arguments (of `int` type) to be appended in
   * a space-delimited fashion after the fname
   */
  template <typename... Values>
  void log(std::string fname, Values... vs);

  /**
   * Sets the log stream for a memory manager base.
   *
   * @param[in] logStream the output stream to set.
   */
  void setLogStream(std::ostream* logStream);

  /**
   * Returns the log stream for a memory manager base.
   *
   * @return the manager's log stream.
   */
  std::ostream* getLogStream() const;

  /**
   * Sets the logging mode for the memory manager base. If disabled, no logs are
   * written. If enabled, all function calls to virtual base class methods are
   * logged.
   *
   * @param[in] log bool determinig whether logging is enabled.
   */
  void setLoggingEnabled(bool log);

  /**
   * Sets a number of lines after which the adapter's temporary logging buffer
   * gets flushed to the user-supplied output stream. Default value is 50.
   *
   * @param[in] interval the number of lines after which to flush the temporary
   * log buffer. Supplied interval must be greater than 1.
   */
  void setLogFlushInterval(size_t interval);

  /**
   * Returns the ArrayFire handle for this memory manager.
   *
   * @return the `af_memory_manager` handle associated with this class.
   */
  af_memory_manager getHandle() const;

  // Native and device memory management functions
  const std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface;

 protected:
  // AF memory manager entity containing relevant function pointers
  af_memory_manager interface_;

 private:
  // Logging components
  bool loggingEnabled_{false};
  std::ostream* logStream_;
  std::stringstream logStreamBuffer_;
  size_t logStreamBufferSize_{0}; // in number of lines
  size_t logFlushInterval_{kDefaultLogFlushInterval};
};

template <typename... Values>
void MemoryManagerAdapter::log(std::string fname, Values... vs) {
  if (loggingEnabled_) {
    if (!logStream_) {
      throw std::runtime_error(
          "MemoryManagerAdapter::log: cannot write to logStream_"
          " - stream is invalid or uninitialized");
    }
    logStreamBuffer_ << fname << " ";
    int unpack[]{0, (logStreamBuffer_ << std::to_string(vs) << " ", 0)...};
    static_cast<void>(unpack);
    logStreamBuffer_ << '\n';
    logStreamBufferSize_++;
    // Decide whether or not to flush
    if (logStreamBufferSize_ == logFlushInterval_) {
      *logStream_ << logStreamBuffer_.str();
      logStreamBuffer_.str(""); // clear the log buffer.
      logStreamBufferSize_ = 0;
    }
  }
}

}; // namespace fl
