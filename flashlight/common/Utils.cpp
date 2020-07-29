/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <stdexcept>

#include "flashlight/common/Utils.h"

namespace fl {

bool allClose(
    const af::array& a,
    const af::array& b,
    double absTolerance /* = 1e-5 */) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.dims() != b.dims()) {
    return false;
  }
  if (a.isempty() && b.isempty()) {
    return true;
  }
  return af::max<double>(af::abs(a - b)) < absTolerance;
}

std::string dateTimeWithMicroSeconds() {
  std::chrono::system_clock::time_point highResTime =
      std::chrono::high_resolution_clock::now();
  const time_t secondsSinceEpoc =
      std::chrono::system_clock::to_time_t(highResTime);
  const struct tm* timeinfo = localtime(&secondsSinceEpoc);

  // Formate date and time to the seconds as:
  // MMDD HH MM SS
  // 1231 08:42:42
  constexpr size_t bufferSize = 50;
  char buffer[bufferSize];
  const size_t nWrittenBytes = std::strftime(buffer, 30, "%m%d %T", timeinfo);
  if (!nWrittenBytes) {
    return "getTime() failed to format time";
  }

  const std::chrono::system_clock::time_point timeInSecondsResolution =
      std::chrono::system_clock::from_time_t(secondsSinceEpoc);
  const auto usec = std::chrono::duration_cast<std::chrono::microseconds>(
      highResTime - timeInSecondsResolution);

  // Add msec and usec.
  std::snprintf(
      buffer + nWrittenBytes,
      bufferSize - nWrittenBytes,
      ".%06ld",
      usec.count());

  return buffer;
}

size_t divRoundUp(size_t numerator, size_t denominator) {
  if (!numerator) {
    return 0;
  }
  if (!denominator) {
    throw std::invalid_argument(
        std::string("divRoundUp() zero denominator error"));
  }
  return (numerator + denominator - 1) / denominator;
}

namespace {
std::string prettyStringMemorySizeUnits(size_t size) {
  if (size == SIZE_MAX) {
    return "SIZE_MAX";
  }
  std::stringstream ss;

  bool isFirst = true;
  while (size) {
    size_t shift = 0;
    const char* unit = "";
    if (size >= (1L << 40)) { // >= 8TB
      shift = 40;
      unit = "TB";
    } else if (size >= (1L << 30)) { // >= 8G B
      shift = 30;
      unit = "GB";
    } else if (size >= (1L << 20)) { // >= 8M B
      shift = 20;
      unit = "MB";
    } else if (size >= (1L << 10)) { // >= 8K B
      shift = 10;
      unit = "KB";
    }
    if (size > 0) {
      if (!isFirst) {
        ss << '+';
      }
      isFirst = false;
      size_t nUnits = size >> shift;
      ss << nUnits << unit;
      size -= (nUnits << shift);
    }
  }

  return ss.str();
}

std::string prettyStringCountUnits(size_t count) {
  if (count == SIZE_MAX) {
    return "SIZE_MAX";
  }
  std::stringstream ss;

  bool isFirst = true;
  while (count) {
    size_t magnitude = 1;
    const char* unit = "";
    if (count >= 1e12) {
      magnitude = 1e12;
      unit = "t";
    } else if (count >= 1e9) {
      magnitude = 1e9;
      unit = "b";
    } else if (count >= 1e6) {
      magnitude = 1e6;
      unit = "m";
    } else if (count >= 1e3) {
      magnitude = 1e3;
      unit = "k";
    }
    if (count > 0) {
      if (!isFirst) {
        ss << '+';
      }
      isFirst = false;
      size_t nUnits = count / magnitude;
      ss << nUnits << unit;
      count -= (nUnits * magnitude);
    }
  }

  return ss.str();
}
} // namespace

std::string prettyStringMemorySize(size_t size) {
  if (size == SIZE_MAX) {
    return "SIZE_MAX";
  }
  std::stringstream ss;
  ss << size;
  if (size >= (1UL << 13)) {
    ss << '(' << prettyStringMemorySizeUnits(size) << ')';
  }

  return ss.str();
}

std::string prettyStringCount(size_t count) {
  if (count == SIZE_MAX) {
    return "SIZE_MAX";
  }
  std::stringstream ss;
  ss << count;

  if (count >= 1e3) { // >= 10 thousand
    ss << '(' << prettyStringCountUnits(count) << ')';
  }
  return ss.str();
}

} // namespace fl
