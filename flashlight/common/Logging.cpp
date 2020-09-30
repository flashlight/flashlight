/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <thread>
#include <utility>

#include "flashlight/flashlight/common/Logging.h"
#include "flashlight/flashlight/common/Utils.h"

namespace fl {
LogLevel Logging::maxLoggingLevel_ = DEFUALT_MAX_LOGGING_LEVEL;
int VerboseLogging::maxLoggingLevel_ = DEFUALT_MAX_VERBOSE_LOGGING_LEVEL;

namespace {
// Constatnts for ANSI terminal colors.
constexpr const char* RED = "\033[0;31m";
constexpr const char* GREEN = "\033[0;32m";
constexpr const char* YELLOW = "\033[0;33m";
constexpr const char* NO_COLOR = "\033[0m";

#ifdef _WIN32
constexpr const char* kSeparator = "\\";
#else
constexpr const char* kSeparator = "/";
#endif

std::string getFileName(const std::string& path) {
  const size_t separatorIndex = path.rfind(kSeparator, path.length());
  if (separatorIndex == std::string::npos) {
    return path;
  }
  return path.substr(separatorIndex + 1, path.length() - separatorIndex);
}

void addContext(
    const char* fullPath,
    int lineNumber,
    std::stringstream* outputStream) {
  // report only the last threadIdNumDigits of the thread ID for succinctness
  // and compatibility with glog.
  constexpr size_t threadIdNumDigits = 5;
  std::stringstream ss;
  ss << std::this_thread::get_id();
  const std::string threadId = ss.str();

  (*outputStream) << dateTimeWithMicroSeconds() << ' '
                  << threadId.substr(threadId.size() - threadIdNumDigits) << ' '
                  << getFileName(fullPath) << ':' << lineNumber << ' ';
}

} // namespace

Logging::Logging(LogLevel level, const char* fullPath, int lineNumber)
    : level_(level), outputStreamPtr_(&std::cout) {
  if (level_ <= Logging::maxLoggingLevel_) {
    switch (level_) {
      case LogLevel::INFO:
        stringStream_ << GREEN << "I";
        break;
      case LogLevel::WARNING:
        stringStream_ << YELLOW << "W";
        break;
      case LogLevel::ERROR:
        outputStreamPtr_ = &std::cerr;
        stringStream_ << RED << "E";
        break;
      case LogLevel::FATAL:
        outputStreamPtr_ = &std::cerr;
        stringStream_ << RED << "F";
        break;
      default:
        outputStreamPtr_ = &std::cerr;
        stringStream_ << RED << "Invalid log level ";
    };
    addContext(fullPath, lineNumber, &stringStream_);
    stringStream_ << NO_COLOR;
  }
}

Logging::~Logging() {
  if (level_ <= Logging::maxLoggingLevel_) {
    stringStream_ << std::endl;
    (*outputStreamPtr_) << stringStream_.str();
    outputStreamPtr_->flush();
    if (level_ == LogLevel::FATAL) {
      exit(-1);
    }
  }
}

void Logging::setMaxLoggingLevel(LogLevel maxLoggingLevel) {
  Logging::maxLoggingLevel_ = maxLoggingLevel;
}

Logging&& operator<<(Logging&& log, const std::string& s) {
  return std::move(log.print(s));
}

Logging&& operator<<(Logging&& log, const char* s) {
  return std::move(log.print(s));
}

Logging&& operator<<(Logging&& log, const void* s) {
  return std::move(log.print(s));
}

Logging&& operator<<(Logging&& log, char c) {
  return std::move(log.print(c));
}

Logging&& operator<<(Logging&& log, unsigned char u) {
  return std::move(log.print(u));
}

Logging&& operator<<(Logging&& log, int i) {
  return std::move(log.print(i));
}

Logging&& operator<<(Logging&& log, unsigned int u) {
  return std::move(log.print(u));
}

Logging&& operator<<(Logging&& log, long l) {
  return std::move(log.print(l));
}

Logging&& operator<<(Logging&& log, long long l) {
  return std::move(log.print(l));
}

Logging&& operator<<(Logging&& log, unsigned long u) {
  return std::move(log.print(u));
}

Logging&& operator<<(Logging&& log, unsigned long long u) {
  return std::move(log.print(u));
}

Logging&& operator<<(Logging&& log, float f) {
  return std::move(log.print(f));
}

Logging&& operator<<(Logging&& log, double d) {
  return std::move(log.print(d));
}

Logging&& operator<<(Logging&& log, bool b) {
  return std::move(log.print(b));
}

VerboseLogging::VerboseLogging(int level, const char* fullPath, int lineNumber)
    : level_(level) {
  if (level_ <= VerboseLogging::maxLoggingLevel_) {
    stringStream_ << "vlog(" << level_ << ") ";
    addContext(fullPath, lineNumber, &stringStream_);
  }
}

VerboseLogging::~VerboseLogging() {
  if (level_ <= VerboseLogging::maxLoggingLevel_) {
    stringStream_ << std::endl;
    std::cout << stringStream_.str();
    std::cout.flush();
  }
}

void VerboseLogging::setMaxLoggingLevel(int maxLoggingLevel) {
  VerboseLogging::maxLoggingLevel_ = maxLoggingLevel;
}

VerboseLogging&& operator<<(VerboseLogging&& log, const std::string& s) {
  return std::move(log.print(s));
}

VerboseLogging&& operator<<(VerboseLogging&& log, const char* s) {
  return std::move(log.print(s));
}

VerboseLogging&& operator<<(VerboseLogging&& log, const void* s) {
  return std::move(log.print(s));
}

VerboseLogging&& operator<<(VerboseLogging&& log, char c) {
  return std::move(log.print(c));
}

VerboseLogging&& operator<<(VerboseLogging&& log, unsigned char u) {
  return std::move(log.print(u));
}

VerboseLogging&& operator<<(VerboseLogging&& log, int i) {
  return std::move(log.print(i));
}

VerboseLogging&& operator<<(VerboseLogging&& log, unsigned int u) {
  return std::move(log.print(u));
}

VerboseLogging&& operator<<(VerboseLogging&& log, long l) {
  return std::move(log.print(l));
}

VerboseLogging&& operator<<(VerboseLogging&& log, unsigned long u) {
  return std::move(log.print(u));
}

VerboseLogging&& operator<<(VerboseLogging&& log, float f) {
  return std::move(log.print(f));
}

VerboseLogging&& operator<<(VerboseLogging&& log, double d) {
  return std::move(log.print(d));
}

VerboseLogging&& operator<<(VerboseLogging&& log, bool b) {
  return std::move(log.print(b));
}

} // namespace fl
