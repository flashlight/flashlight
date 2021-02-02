/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/memory/MemoryManagerAdapter.h"

#include <stdexcept>
#include <utility>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {

const size_t MemoryManagerAdapter::kDefaultLogFlushInterval;
const size_t MemoryManagerAdapter::kLogStatsMask;
const size_t MemoryManagerAdapter::kLogEveryOperationMask;

namespace {

constexpr const char* kMemLogFile = "FL_MEM_LOG_FILE";
constexpr const char* kMemLogFlushInterval = "FL_MEM_LOG_FLUSH_INTERVAL";
constexpr const char* kMemLogStats = "FL_MEM_LOG_STATS";
constexpr const char* kMemLogEveryOperation = "FL_MEM_LOG_OPS";

/**
 * Return the value of the named environment variable interprested as unsigend
 * long integer. Return the defaultVal on failure to read the environment
 * variable as an integer.
 */
size_t getEnvAsSize(const char* name, size_t defaultVal) {
  const char* env = std::getenv(name);
  if (env) {
    try {
      return std::stoul(env);
    } catch (std::exception& ex) {
      FL_LOG(fl::ERROR) << "Invalid environment variable=" << name
                        << " value=" << env;
    }
  }
  return defaultVal;
}

/**
 * Return true when the named environment variable is read as non zero integer.
 * Return false when the value is zero. Return the defaultVal on failure to read
 * the environment variable as an integer.
 */
bool getEnvAsBool(const char* name, bool defaultVal) {
  return getEnvAsSize(name, defaultVal ? 1 : 0) != 0;
}

std::unique_ptr<std::ofstream> logFile;
} // namespace

void MemoryManagerAdapter::configFromEnvironmentVariables() {
  const char* filename = std::getenv(kMemLogFile);
  if (filename) {
    logFile = std::make_unique<std::ofstream>(filename);
    if (logFile && *logFile) {
      setLogFlushInterval(getEnvAsSize(
          kMemLogFlushInterval,
          MemoryManagerAdapter::kDefaultLogFlushInterval));
      size_t enableMask = 0;
      if (getEnvAsBool(kMemLogStats, false)) {
        enableMask |= MemoryManagerAdapter::kLogStatsMask;
      }
      if (getEnvAsBool(kMemLogEveryOperation, false)) {
        enableMask |= MemoryManagerAdapter::kLogEveryOperationMask;
      }
      setLoggingEnabled(enableMask);
      setLogStream(logFile.get());

      FL_LOG(fl::INFO) << "MemoryManagerAdapter log filename=" << filename
                       << " enableMask=" << enableMask;

    } else {
      FL_LOG(fl::ERROR) << "Failed to open memory log file=" << filename;
    }
  }
}

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
  } else {
    configFromEnvironmentVariables();
  }

  // Create handle and set payload to point to this instance
  AF_CHECK(af_create_memory_manager(&interface_));
  AF_CHECK(af_memory_manager_set_payload(interface_, (void*)this));
}

MemoryManagerAdapter::~MemoryManagerAdapter() {
  // Flush the log buffer and log stream
  if (logStream_ && *logStream_) {
    *logStream_ << logStreamBuffer_.str();
    logStream_->flush();
  }

  if (interface_) {
    af_release_memory_manager(interface_); // nothrow
  }
}

void MemoryManagerAdapter::setLogStream(std::ostream* logStream) {
  std::cout << "setLogStream(logStream= " << logStream << ")" << std::endl;
  logStream_ = logStream;
}

void MemoryManagerAdapter::setLoggingEnabled(size_t log) {
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
