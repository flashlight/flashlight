/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/ForceAlignmentCriterion.h"
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include <flashlight/lib/sequence/criterion/cpu/ForceAlignmentCriterion.h>

using fl::Variable;
using FAC = fl::lib::cpu::ForceAlignmentCriterion<float>;

namespace {
// By passing shared_ptr<Context> we avoid copies from forward to backward.
struct Context {
  std::vector<int> targetVec;
  std::vector<int> targetSizeVec;
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
    int L,
    const std::shared_ptr<Context>& ctx) {
  if (gradVar.type() != fl::dtype::f32) {
    throw std::invalid_argument("FAC: grad must be float32");
  }

  auto gradVec = gradVar.tensor().toHostVector<float>();
  std::vector<float> inputGradVec(B * T * N);
  std::vector<float> transGradVec(N * N);

  FAC::backward(
      B,
      T,
      N,
      L,
      ctx->targetVec.data(),
      ctx->targetSizeVec.data(),
      gradVec.data(),
      inputGradVec.data(),
      transGradVec.data(),
      ctx->workspaceVec.data());

  auto inputGrad = Tensor::fromVector({N, T, B}, inputGradVec);
  auto transGrad = Tensor::fromVector({N, N}, transGradVec);

  inputs[0].addGrad(Variable(inputGrad, false));
  inputs[1].addGrad(Variable(transGrad, false));
}

Variable ForceAlignmentCriterion::forward(
    const Variable& inputVar,
    const Variable& targetVar) {
  const auto& transVar = param(0);
  int B = inputVar.dim(2);
  int T = inputVar.dim(1);
  int N = inputVar.dim(0);
  int L = targetVar.dim(0);

  if (N != transVar.dim(0)) {
    throw std::invalid_argument(
        "ForceAlignmentCriterion(cpu)::forward: input dim doesn't match N");
  } else if (inputVar.type() != fl::dtype::f32) {
    throw std::invalid_argument(
        "ForceAlignmentCriterion(cpu)::forward: input must be float32");
  } else if (targetVar.type() != fl::dtype::s32) {
    throw std::invalid_argument(
        "ForceAlignmentCriterion(cpu)::forward: target must be int32");
  }

  const auto& targetSize = getTargetSizeArray(targetVar.tensor(), T);
  auto ctx = std::make_shared<Context>();
  auto inputVec = inputVar.tensor().toHostVector<float>();
  ctx->targetVec = targetVar.tensor().toHostVector<int>();
  ctx->targetSizeVec = targetSize.toHostVector<int>();
  auto transVec = transVar.tensor().toHostVector<float>();
  std::vector<float> lossVec(B);
  ctx->workspaceVec.assign(FAC::getWorkspaceSize(B, T, N, L), 0);

  FAC::forward(
      B,
      T,
      N,
      L,
      scaleMode_,
      inputVec.data(),
      ctx->targetVec.data(),
      ctx->targetSizeVec.data(),
      transVec.data(),
      lossVec.data(),
      ctx->workspaceVec.data());

  return Variable(
      Tensor::fromVector(lossVec),
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Variable>& inputs, const Variable& gradVar) {
        backward(inputs, gradVar, B, T, N, L, ctx);
      });
}

Tensor ForceAlignmentCriterion::viterbiPath(
    const Tensor& input,
    const Tensor& target) {
  const Tensor& trans = param(0).tensor();
  int N = input.dim(0); // Number of output tokens
  int T = input.dim(1); // Utterance length
  int B = input.dim(2); // Batchsize
  int L = target.dim(0); // Target length

  if (N != trans.dim(0)) {
    throw std::invalid_argument("FAC: input dim doesn't match N:");
  } else if (input.type() != fl::dtype::f32) {
    throw std::invalid_argument("FAC: input must be float32");
  } else if (target.type() != fl::dtype::s32) {
    throw std::invalid_argument("FAC: target must be int32");
  }
  const Tensor targetSize = getTargetSizeArray(target, T);
  std::shared_ptr<Context> ctx = std::make_shared<Context>();
  std::vector<float> inputVec = input.toHostVector<float>();
  ctx->targetVec = target.toHostVector<int>();
  ctx->targetSizeVec = targetSize.toHostVector<int>();
  std::vector<float> transVec = trans.toHostVector<float>();
  std::vector<float> lossVec(B);
  ctx->workspaceVec.assign(FAC::getWorkspaceSize(B, T, N, L), 0);
  std::vector<int> bestPaths(B * T);
  FAC::viterbi(
      B,
      T,
      N,
      L,
      inputVec.data(),
      ctx->targetVec.data(),
      ctx->targetSizeVec.data(),
      transVec.data(),
      bestPaths.data(),
      ctx->workspaceVec.data());
  return Tensor::fromVector({T, B}, bestPaths);
}
} // namespace speech
} // namespace pkg
} // namespace fl
