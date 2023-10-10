/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "flashlight/fl/common/Defines.h"

/**
 * \defgroup logging Logging Library
 *
 * Logging is a light, multi-level, compile-time-filterable, logging
 * framework that is similar to [glog](https://github.com/google/glog) in output
 * format. It defines two logging macros, one for conventional logging and the
 * other for verbose logging. Compile time filtering is applied separately to
 * each of the two.
 *
 * Output format:
 * \code
 * LMMDD HH:MM:SS.uuuuuu tid filename:##] Log message ...
 * \endcode
 *
 * Where:
 * \code
 * L: Log level {Fatal, Critical, Error, Warning, Info}
 * MMDD: month, day
 * HH:MM:SS.uuuuuu: time (24-hour format) with micro-seconds
 * tid: thread ID
 * filename:## the basename of the source file and line number of the FL_LOG
 * message
 *
 * FL_LOG use examples:
 * \code
 *   FL_LOG(INFO) << "foo bar n=" << 42;
 * \endcode
 * Output example:
 *   I0206 10:42:21.047293 87072 Logging.h:15 foo bar n=42
 * Note that FL_LOG(level) only prints when level is <= from value set to
 * Logging::setMaxLoggingLevel(level)
 *
 * FL_VLOG use example:
 * \code
 *   FL_VLOG(1) << "foo bar n=" << 42;
 * \endcode
 * Output example:
 * \code
 *   vlog(1)0206 10:42:21.005439 87072 Logging.h:23 foo bar n=42
 * \endcode
 * Note that FL_VLOG(level) only prints when level is <= from value set to
 * \code
 * VerboseLogging::setMaxLoggingLevel(level)
 * \endcode
 *
 * Gives output:
 *
 * \code
 * vlog(1) 0206 10:42:21.005439 87072 Logging.h:23 foo bar n=42
 * \endcode
 *
 * Note that `VLOG(level)` only prints when level is less than or equal to the
 * value set to `VerboseLogging`
 */
namespace fl {

/**
 * Initialize all logging components including stacktraces and signal handling.
 */
FL_API void initLogging();

/// \ingroup logging
enum class LogLevel {
  DISABLED, // use only for when calling setMaxLoggingLevel() or
  // setting DEFAULT_MAX_FL_LOGGING_LEVEL.
  FATAL,
  ERROR,
  WARNING,
  INFO,
};

/**
 * Gets the `LogLevel` for a given string. Throws if invalid.
 *
 * \ingroup logging
 */
FL_API LogLevel logLevelValue(const std::string& level);

/**
 * Gets string representation of a given `LogLevel`.
 *
 * \ingroup logging
 */
FL_API std::string logLevelName(LogLevel level);

/**
 *  Used to globally limit `FL_LOG(level)`.
 *
 * \ingroup logging
 */
constexpr LogLevel DEFAULT_MAX_FL_LOGGING_LEVEL = LogLevel::INFO;
/**
 * `MAX_VERBOSE_FL_LOGGING_LEVEL` values are based on the values used in
 * `FL_VLOG()` and can be any value, but expected reasonable values are: 0..10
 * for print none and print all respectively.
 *
 * \ingroup logging
 */
constexpr int DEFAULT_MAX_VERBOSE_FL_LOGGING_LEVEL = 0;

/**
 * Write to log output for a given `LogLevel`.
 *
 * \ingroup logging
 */
#define FL_LOG(level) fl::Logging(level, __FILE__, __LINE__)

/**
 * Write to verbose log output for a given verbose logging level.
 *
 * \ingroup logging
 */
#define FL_VLOG(level) fl::VerboseLogging(level, __FILE__, __LINE__)

// Optimization macros that allow to run code only we are going to log it.
#define IF_LOG(level) if (fl::Logging::ifLog(level))
#define IF_VLOG(level) if (fl::VerboseLogging::ifLog(level))

/// \ingroup logging
#define FL_LOG_IF(level, exp) \
  if (exp)                    \
  fl::Logging(level, __FILE__, __LINE__)
/// \ingroup logging
#define FL_VLOG_IF(level, exp) \
  if (exp)                     \
  fl::VerboseLogging(level, __FILE__, __LINE__)

class FL_API Logging {
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

  // Overrides DEFAULT_MAX_FL_LOGGING_LEVEL value.
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

class FL_API VerboseLogging {
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

  // Overrides DEFAULT_MAX_VERBOSE_FL_LOGGING_LEVEL value.
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
FL_API Logging&& operator<<(Logging&& log, const std::string& s);
FL_API Logging&& operator<<(Logging&& log, const char* s);
FL_API Logging&& operator<<(Logging&& log, const void* s);
FL_API Logging&& operator<<(Logging&& log, char c);
FL_API Logging&& operator<<(Logging&& log, unsigned char u);
FL_API Logging&& operator<<(Logging&& log, int i);
FL_API Logging&& operator<<(Logging&& log, unsigned int u);
FL_API Logging&& operator<<(Logging&& log, long l);
FL_API Logging&& operator<<(Logging&& log, long long l);
FL_API Logging&& operator<<(Logging&& log, unsigned long u);
FL_API Logging&& operator<<(Logging&& log, unsigned long long u);
FL_API Logging&& operator<<(Logging&& log, float f);
FL_API Logging&& operator<<(Logging&& log, double d);
FL_API Logging&& operator<<(Logging&& log, bool b);

// Catch all designed mostly for <iomanip> stuff.
template <typename T>
Logging&& operator<<(Logging&& log, const T& t) {
  return log.print(t);
}

FL_API VerboseLogging&& operator<<(VerboseLogging&& log, const std::string& s);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, const char* s);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, const void* s);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, char c);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, unsigned char u);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, int i);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, unsigned int u);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, long l);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, unsigned long u);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, float f);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, double d);
FL_API VerboseLogging&& operator<<(VerboseLogging&& log, bool b);

// Catch all designed mostly for <iomanip> stuff.
template <typename T>
VerboseLogging&& operator<<(VerboseLogging&& log, const T& t) {
  return log.print(t);
}

} // namespace fl
