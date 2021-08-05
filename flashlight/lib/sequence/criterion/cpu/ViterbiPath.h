/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

namespace fl {
namespace lib {
namespace cpu {

/// Check CUDA header for docs.
template <class Float>
struct ViterbiPath {
  static size_t getWorkspaceSize(int B, int T, int N);

  static void compute(
      int B,
      int T,
      int N,
      const Float* input,
      const Float* trans,
      int* path,
      void* workspace);
};

} // namespace cpu
} // namespace lib
} // namespace fl
