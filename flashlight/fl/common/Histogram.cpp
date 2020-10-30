/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/Histogram.h"

#include <iomanip>
#include <stdexcept>

namespace fl {

void shortFormatCount(std::stringstream& ss, size_t count) {
  constexpr size_t stringLen = 5;
  if (count >= 10e13) { // >= 10 trillion
    ss << std::setw(stringLen - 1) << (count / (size_t)10e12) << 't';
  } else if (count >= 10e10) { // >= 10 billion
    ss << std::setw(stringLen - 1) << (count / (size_t)10e9) << 'b';
  } else if (count >= 10e7) { // >= 10 million
    ss << std::setw(stringLen - 1) << (count / (size_t)10e6) << 'm';
  } else if (count >= 10e4) { // >= 10 thousand
    ss << std::setw(stringLen - 1) << (count / (size_t)10e3) << 'k';
  } else {
    ss << std::setw(stringLen) << count;
  }
}

void shortFormatMemory(std::stringstream& ss, size_t size) {
  constexpr size_t stringLen = 5;
  if (size >= (1L << 43)) { // >= 8TB
    ss << std::setw(stringLen - 1) << (size >> 40) << "T";
  } else if (size >= (1L << 33)) { // >= 8G B
    ss << std::setw(stringLen - 1) << (size >> 30) << "G";
  } else if (size >= (1L << 23)) { // >= 8M B
    ss << std::setw(stringLen - 1) << (size >> 20) << "M";
  } else if (size >= (1L << 13)) { // >= 8K B
    ss << std::setw(stringLen - 1) << (size >> 10) << "K";
  } else {
    ss << std::setw(stringLen) << size;
  }
}

} // namespace fl
