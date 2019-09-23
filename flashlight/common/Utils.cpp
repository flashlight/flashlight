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

#include "flashlight/common/Utils.h"

#include <unordered_map>

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
af::dtype stringToAfType(const std::string& typeName) {
  std::unordered_map<std::string, af::dtype> strToType = {
      {"f32", f32},
      {"c32", c32},
      {"f64", f64},
      {"c64", c64},
      {"b8", b8},
      {"s32", s32},
      {"u32", u32},
      {"u8", u8},
      {"s64", s64},
      {"u64", u64},
      {"s16", s16},
      {"u16", u16},
      {"f16", f16},
  };
  if (strToType.find(typeName) != strToType.end()) {
    return strToType[typeName];
  }
  throw std::invalid_argument("Invalid inputs type: " + typeName);
}

std::string afTypeToString(const af::dtype& type) {
  u_map<af::dtype, std::string> typeToStr = {
      {f32, "f32"},
      {c32, "c32"},
      {f64, "f64"},
      {c64, "c64"},
      {b8, "b8"},
      {s32, "s32"},
      {u32, "u32"},
      {u8, "u8"},
      {s64, "s64"},
      {u64, "u64"},
      {s16, "s16"},
      {u16, "u16"},
      {f16, "f16"},
  };
  if (typeToStr.find(type) != typeToStr.end()) {
    return typeToStr[type];
  }
  throw std::invalid_argument("Invalid inputs type: " + type);
}

} // namespace fl
