/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/CriterionUtils.h"

#include <flashlight/common/cuda.h>

#include "libraries/audio/criterion/cuda/CriterionUtils.cuh"
#include "libraries/audio/criterion/cuda/ViterbiPath.cuh"

using CriterionUtils = fl::lib::cuda::CriterionUtils<float>;
using ViterbiPath = fl::lib::cuda::ViterbiPath<float>;

namespace fl {
namespace task {
namespace asr {

af::array viterbiPath(const af::array& input, const af::array& trans) {
  auto B = input.dims(2);
  auto T = input.dims(1);
  auto N = input.dims(0);

  if (N != trans.dims(0) || N != trans.dims(1)) {
    throw std::invalid_argument("viterbiPath: mismatched dims");
  } else if (input.type() != f32) {
    throw std::invalid_argument("viterbiPath: input must be float32");
  } else if (trans.type() != f32) {
    throw std::invalid_argument("viterbiPath: trans must be float32");
  }

  af::array path(T, B, s32);
  af::array workspace(ViterbiPath::getWorkspaceSize(B, T, N), u8);

  {
    fl::DevicePtr inputRaw(input);
    fl::DevicePtr transRaw(trans);
    fl::DevicePtr pathRaw(path);
    fl::DevicePtr workspaceRaw(workspace);

    ViterbiPath::compute(
        B,
        T,
        N,
        static_cast<const float*>(inputRaw.get()),
        static_cast<const float*>(transRaw.get()),
        static_cast<int*>(pathRaw.get()),
        workspaceRaw.get(),
        fl::cuda::getActiveStream());
  }

  return path;
}

af::array getTargetSizeArray(const af::array& target, int maxSize) {
  int B = target.dims(1);
  int L = target.dims(0);

  af::array targetSize(B, s32);

  {
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);

    CriterionUtils::batchTargetSize(
        B,
        L,
        maxSize,
        static_cast<const int*>(targetRaw.get()),
        static_cast<int*>(targetSizeRaw.get()),
        fl::cuda::getActiveStream());
  }

  return targetSize;
}
} // namespace asr
} // namespace task
} // namespace fl
