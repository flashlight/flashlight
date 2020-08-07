/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

#include "flashlight/libraries/sequence/criterion/Defines.h"
using fl::lib::seq::CriterionScaleMode;

namespace fl {
namespace lib {
namespace cpu {

/// Check CUDA header for docs.
template <class Float>
struct CriterionUtils {
  static void batchTargetSize(
      int B,
      int L,
      int maxSize,
      const int* target,
      int* targetSize);

  static void computeScale(
      int B,
      int T,
      int N,
      CriterionScaleMode scaleMode,
      const int* targetSize,
      Float* scale);
};

/// Zeroes `count * sizeof(T)` device bytes
template <typename T>
void setZero(T* ptr, size_t count) {
  std::memset(ptr, 0, count * sizeof(T));
}

} // namespace cpu
} // namespace lib
} // namespace fl
