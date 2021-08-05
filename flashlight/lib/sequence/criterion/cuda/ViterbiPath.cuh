/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>

namespace fl {
namespace lib {
namespace cuda {

/// Computes max likelihood path using Viterbi algorithm.
template <class Float>
struct ViterbiPath {
  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   */
  static size_t getWorkspaceSize(int B, int T, int N);

  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   * input: [B][T][N] input frames from network
   * trans: [N][N] transition matrix
   * path: [B][T] (out) Viterbi path
   * workspace: (in/out) internal workspace
   * stream: CUDA stream
   */
  static void compute(
      int B,
      int T,
      int N,
      const Float* input,
      const Float* trans,
      int* path,
      void* workspace,
      cudaStream_t stream);
};

} // namespace cuda
} // namespace lib
} // namespace fl
