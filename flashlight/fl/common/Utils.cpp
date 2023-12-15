/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/Utils.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

bool f16Supported() {
  return defaultTensorBackend().isDataTypeSupported(fl::dtype::f16);
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

std::string getEnvVar(
    const std::string& key,
    const std::string& dflt /*= "" */) {
  char* val = getenv(key.c_str());
  return val ? std::string(val) : dflt;
}

} // namespace fl
