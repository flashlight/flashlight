/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/memory.h>

#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "flashlight/memory/MemoryManagerDeviceInterface.h"

namespace fl {

namespace {

const size_t kDefaultLogFlushInterval = 50;

} // namespace

/**
 * An interface for defining memory managers purely in C++
 */
class MemoryManagerAdapter {
 public:
  explicit MemoryManagerAdapter(
      std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface,
      std::ostream* logStream = nullptr);
  virtual ~MemoryManagerAdapter();
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
  virtual void printInfo(const char* msg, const int device) = 0;
  virtual void userLock(const void* ptr) = 0;
  virtual void userUnlock(const void* ptr) = 0;
  virtual bool isUserLocked(const void* ptr) = 0;
  virtual float getMemoryPressure() = 0;
  virtual bool jitTreeExceedsMemoryPressure(size_t bytes) = 0;
  virtual void addMemoryManagement(int device) = 0;
  virtual void removeMemoryManagement(int device) = 0;

  // Native and device memory management functions
  const std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface;

  // Logging functions
  template <typename... Values>
  void log(std::string fname, Values... vs);
  void setLogStream(std::ostream* logStream);
  void setLoggingEnabled(bool log);
  void setLogFlushInterval(size_t interval);

  const af_memory_manager getHandle() const;

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
      logStreamBufferSize_ = 0;
    }
  }
}

}; // namespace fl
