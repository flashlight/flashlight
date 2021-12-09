/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/FullConnectionCriterion.h"
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include "flashlight/lib/sequence/criterion/cpu/FullConnectionCriterion.h"
#include "flashlight/pkg/runtime/common/DistributedUtils.h"

using fl::Variable;
using FCC = fl::lib::cpu::FullConnectionCriterion<float>;

namespace {
// By passing shared_ptr<Context> we avoid copies from forward to backward.
struct Context {
  std::vector<float> transVec;
  std::vector<uint8_t> workspaceVec;
};
} // namespace

namespace fl {
namespace pkg {
namespace speech {

static void backward(
    std::vector<Variable>& inputs,
    const Variable& gradVar,
    int B,
    int T,
    int N,
    const std::shared_ptr<Context>& ctx) {
  if (gradVar.type() != f32) {
    throw std::invalid_argument("FCC: grad must be float32");
  }

  auto gradVec = gradVar.toHostVector<float>();
  std::vector<float> inputGradVec(B * T * N);
  std::vector<float> transGradVec(N * N);

  FCC::backward(
      B,
      T,
      N,
      ctx->transVec.data(),
      gradVec.data(),
      inputGradVec.data(),
      transGradVec.data(),
      ctx->workspaceVec.data());

  Tensor inputGrad = Tensor::fromVector({N, T, B}, inputGradVec);
  Tensor transGrad = Tensor::fromVector({N, N}, transGradVec);

  inputs[0].addGrad(Variable(inputGrad, false));
  inputs[1].addGrad(Variable(transGrad, false));
}

Variable FullConnectionCriterion::forward(
    const Variable& inputVar,
    const Variable& targetVar) {
  const auto& transVar = param(0);
  int B = inputVar.dims(2);
  int T = inputVar.dims(1);
  int N = inputVar.dims(0);

  if (N != transVar.dims(0)) {
    throw std::invalid_argument("FCC: input dim doesn't match N");
  } else if (inputVar.type() != f32) {
    throw std::invalid_argument("FCC: input must be float32");
  } else if (targetVar.type() != s32) {
    throw std::invalid_argument("FCC: target must be int32");
  }

  const auto& targetSize = getTargetSizeArray(targetVar.array(), T);
  auto ctx = std::make_shared<Context>();
  auto inputVec = inputVar.toHostVector<float>();
  auto targetVec = targetVar.toHostVector<int>();
  auto targetSizeVec = targetSize.toHostVector<int>();
  ctx->transVec = transVar.toHostVector<float>();
  std::vector<float> lossVec(B);
  ctx->workspaceVec.assign(FCC::getWorkspaceSize(B, T, N), 0);

  FCC::forward(
      B,
      T,
      N,
      scaleMode_,
      inputVec.data(),
      targetSizeVec.data(),
      ctx->transVec.data(),
      lossVec.data(),
      ctx->workspaceVec.data());

  return Variable(
      Tensor(B, lossVec.data()),
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Variable>& inputs, const Variable& gradVar) mutable {
        backward(inputs, gradVar, B, T, N, ctx);
      });
}
} // namespace speech
} // namespace pkg
} // namespace fl
