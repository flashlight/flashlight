/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include <stdexcept>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/TensorBackend.h"

#include <flashlight/lib/sequence/criterion/cuda/CriterionUtils.cuh>
#include <flashlight/lib/sequence/criterion/cuda/ViterbiPath.cuh>

using CriterionUtils = fl::lib::cuda::CriterionUtils<float>;
using ViterbiPath = fl::lib::cuda::ViterbiPath<float>;

namespace fl::pkg::speech {

Tensor viterbiPath(const Tensor& input, const Tensor& trans) {
  if (input.ndim() != 3) {
    throw std::invalid_argument(
        "Criterion viterbiPath expects input of shape {N, T, B}");
  }
  if (trans.ndim() != 2) {
    throw std::invalid_argument(
        "Criterion viterbiPath expects trans of shape {N, N}");
  }

  auto B = input.dim(2);
  auto T = input.dim(1);
  auto N = input.dim(0);

  if (N != trans.dim(0) || N != trans.dim(1)) {
    throw std::invalid_argument("viterbiPath: mismatched dims");
  } else if (input.type() != fl::dtype::f32) {
    throw std::invalid_argument("viterbiPath: input must be float32");
  } else if (trans.type() != fl::dtype::f32) {
    throw std::invalid_argument("viterbiPath: trans must be float32");
  }

  Tensor path({T, B}, fl::dtype::s32);
  Tensor workspace(
      {static_cast<long long>(ViterbiPath::getWorkspaceSize(B, T, N))},
      fl::dtype::u8);

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
        input.stream().impl<CUDAStream>().handle());
  }

  return path;
}

Tensor getTargetSizeArray(const Tensor& target, int maxSize) {
  int B = target.dim(1);
  int L = target.dim(0);

  Tensor targetSize({B}, fl::dtype::s32);

  {
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);

    CriterionUtils::batchTargetSize(
        B,
        L,
        maxSize,
        static_cast<const int*>(targetRaw.get()),
        static_cast<int*>(targetSizeRaw.get()),
        target.stream().impl<CUDAStream>().handle());
  }

  return targetSize;
}
} // namespace fl
