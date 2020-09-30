/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

/**
 * Logging is a light, multi-level, compile time filterable, logging
 * infrastructure that is similar to glog in output format. It defines two
 * logging macros, one for any logging and the other for more verbose logging.
 * Compile time filter is applied separately to each of the two.
 *
 * Output format:
 * LMMDD HH:MM:SS.uuuuuu tid filename:##] Log message ...
 *  L: Log level (Fatal, Critical, Error, Warning, Info)
 * MMDD: month, day
 * HH:MM:SS.uuuuuu: time (24-hour format) with micro-seconds
 * tid: thread ID
 * filename:## the basename of the source file and line number of the LOG
 * message
 *
 * LOG use examples:
 *   LOG(INFO) << "foo bar n=" << 42;
 * Output example:
 *   I0206 10:42:21.047293 87072 Logging.h:15 foo bar n=42
 * Note that LOG(level) only prints when level is <= from value set to
 * Logging::setMaxLoggingLevel(level)
 *
 * VLOG use example:
 *   VLOG(1) << "foo bar n=" << 42;
 * Output example:
 *   vlog(1)0206 10:42:21.005439 87072 Logging.h:23 foo bar n=42
 * Note that VLOG(level) only prints when level is <= from value set to
 * VerboseLogging::setMaxLoggingLevel(level)
 *
 */

#pragma once

#include <signal.h>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace fl {
enum LogLevel {
  DISABLE_LOGGING, // use only for when calling setMaxLoggingLevel() or
  // setting DEFUALT_MAX_LOGGING_LEVEL.
  FATAL,
  ERROR,
  WARNING,
  INFO,
};

// DEFUALT_MAX_LOGGING_LEVEL is used for globaly limit LOG(level).
constexpr LogLevel DEFUALT_MAX_LOGGING_LEVEL = LogLevel::INFO;
// MAX_VERBOSE_LOGGING_LEVEL values are based on the values used in VLOG()
// and can be any value, but expected reasonable values are: 0..10
// for print none and print all respectively.
constexpr int DEFUALT_MAX_VERBOSE_LOGGING_LEVEL = 1;

#define LOG(level) Logging(level, __FILE__, __LINE__)
#define VLOG(level) VerboseLogging(level, __FILE__, __LINE__)

#define IFLOG(level) if (Logging::ifLog(level))
#define IFVLOG(level) if (VerboseLogging::ifLog(level))

class Logging {
 public:
  Logging(LogLevel level, const char* filename, int lineNumber);
  ~Logging();

  // Prints t to stdout along with context and sensible font color.
  template <typename T>
  Logging&& print(T& t) {
    if (level_ <= Logging::maxLoggingLevel_) {
      stringStream_ << t;
    }
    return std::move(*this);
  }

  // Overrides DEFUALT_MAX_LOGGING_LEVEL value.
  static void setMaxLoggingLevel(LogLevel maxLoggingLevel);

  static bool ifLog(LogLevel level) {
    return (maxLoggingLevel_ >= level);
  }

 private:
  static LogLevel maxLoggingLevel_;
  const LogLevel level_;
  std::stringstream stringStream_;
  std::ostream* outputStreamPtr_;
};

class VerboseLogging {
 public:
  VerboseLogging(int level, const char* filename, int lineNumber);
  ~VerboseLogging();

  // Prints t to stdout along with logging level and context.
  template <typename T>
  VerboseLogging&& print(T& t) {
    if (level_ <= VerboseLogging::maxLoggingLevel_) {
      stringStream_ << t;
    }
    return std::move(*this);
  }

  // Overrides DEFUALT_MAX_VERBOSE_LOGGING_LEVEL value.
  static void setMaxLoggingLevel(int maxLoggingLevel);

  static bool ifLog(int level) {
    return (maxLoggingLevel_ >= level);
  }

 private:
  static int maxLoggingLevel_;
  const int level_;
  std::stringstream stringStream_;
};

// Can't use template here since the compiler will try resolve
// to all kind of other existing function before it considers
// instantiating a template.
Logging&& operator<<(Logging&& log, const std::string& s);
Logging&& operator<<(Logging&& log, const char* s);
Logging&& operator<<(Logging&& log, const void* s);
Logging&& operator<<(Logging&& log, char c);
Logging&& operator<<(Logging&& log, unsigned char u);
Logging&& operator<<(Logging&& log, int i);
Logging&& operator<<(Logging&& log, unsigned int u);
Logging&& operator<<(Logging&& log, long l);
Logging&& operator<<(Logging&& log, long long l);
Logging&& operator<<(Logging&& log, unsigned long u);
Logging&& operator<<(Logging&& log, unsigned long long u);
Logging&& operator<<(Logging&& log, float f);
Logging&& operator<<(Logging&& log, double d);
Logging&& operator<<(Logging&& log, bool b);

VerboseLogging&& operator<<(VerboseLogging&& log, const std::string& s);
VerboseLogging&& operator<<(VerboseLogging&& log, const char* s);
VerboseLogging&& operator<<(VerboseLogging&& log, const void* s);
VerboseLogging&& operator<<(VerboseLogging&& log, char c);
VerboseLogging&& operator<<(VerboseLogging&& log, unsigned char u);
VerboseLogging&& operator<<(VerboseLogging&& log, int i);
VerboseLogging&& operator<<(VerboseLogging&& log, unsigned int u);
VerboseLogging&& operator<<(VerboseLogging&& log, long l);
VerboseLogging&& operator<<(VerboseLogging&& log, unsigned long u);
VerboseLogging&& operator<<(VerboseLogging&& log, float f);
VerboseLogging&& operator<<(VerboseLogging&& log, double d);
VerboseLogging&& operator<<(VerboseLogging&& log, bool b);

} // namespace fl
