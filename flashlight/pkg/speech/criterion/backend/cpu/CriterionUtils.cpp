/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include <flashlight/lib/sequence/criterion/cpu/CriterionUtils.h>
#include <flashlight/lib/sequence/criterion/cpu/ViterbiPath.h>

using CriterionUtils = fl::lib::cpu::CriterionUtils<float>;
using ViterbiPath = fl::lib::cpu::ViterbiPath<float>;

namespace fl {
namespace pkg {
namespace speech {

Tensor viterbiPath(const Tensor& input, const Tensor& trans) {
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

  auto inputVec = input.toHostVector<float>();
  auto transVec = trans.toHostVector<float>();
  std::vector<int> pathVec(B * T);
  std::vector<uint8_t> workspaceVec(ViterbiPath::getWorkspaceSize(B, T, N));

  ViterbiPath::compute(
      B,
      T,
      N,
      inputVec.data(),
      transVec.data(),
      pathVec.data(),
      workspaceVec.data());

  return Tensor::fromVector({T, B}, pathVec);
}

Tensor getTargetSizeArray(const Tensor& target, int maxSize) {
  int B = target.dim(1);
  int L = target.dim(0);

  auto targetVec = target.toHostVector<int>();
  std::vector<int> targetSizeVec(B);

  CriterionUtils::batchTargetSize(
      B, L, maxSize, targetVec.data(), targetSizeVec.data());

  return Tensor::fromVector(targetSizeVec);
}
} // namespace speech
} // namespace pkg
} // namespace fl
