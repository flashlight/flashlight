/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/memory/MemoryManagerAdapter.h"

#include <stdexcept>
#include <utility>

#include "flashlight/fl/common/Utils.h"

namespace fl {

MemoryManagerAdapter::MemoryManagerAdapter(
    std::shared_ptr<MemoryManagerDeviceInterface> itf,
    std::ostream* logStream)
    : deviceInterface(itf), logStream_(logStream) {
  if (!itf) {
    throw std::invalid_argument(
        "MemoryManagerAdapter::MemoryManagerAdapter - "
        "memory manager device interface is null");
  }
  if (logStream_) {
    loggingEnabled_ = true;
  }

  // Create handle and set payload to point to this instance
  AF_CHECK(af_create_memory_manager(&interface_));
  AF_CHECK(af_memory_manager_set_payload(interface_, (void*)this));
}

MemoryManagerAdapter::~MemoryManagerAdapter() {
  // Flush the log buffer and log stream
  if (logStream_) {
    *logStream_ << logStreamBuffer_.str();
    logStream_->flush();
  }

  if (interface_) {
    af_release_memory_manager(interface_); // nothrow
  }
}

void MemoryManagerAdapter::setLogStream(std::ostream* logStream) {
  logStream_ = logStream;
}

void MemoryManagerAdapter::setLoggingEnabled(bool log) {
  loggingEnabled_ = log;
}

void MemoryManagerAdapter::setLogFlushInterval(size_t interval) {
  if (interval < 1) {
    throw std::invalid_argument(
        "MemoryManagerAdapter::setLogFlushInterval - "
        "flush interval must be great than zero.");
  }
  logFlushInterval_ = interval;
}

af_memory_manager MemoryManagerAdapter::getHandle() const {
  return interface_;
}

size_t MemoryManagerAdapter::getMemStepSize() {
  return -1; //  -1 denotes stepsize is not used by the custom memory manager
}

void MemoryManagerAdapter::setMemStepSize(size_t size) {}

} // namespace fl
