/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/pkg/speech/criterion/ConnectionistTemporalClassificationCriterion.h"

#include <stdexcept>

#include <flashlight/lib/sequence/criterion/cpu/ConnectionistTemporalClassificationCriterion.h>

using CTC = fl::lib::cpu::ConnectionistTemporalClassificationCriterion<float>;

namespace {

using namespace fl;

struct CTCContext {
  std::vector<int> targetVec;
  std::vector<int> targetSizeVec;
  std::vector<uint8_t> workspaceVec;
};

Tensor logSoftmax(const Tensor& input, const int dim) {
  Tensor maxvals = fl::amax(input, {dim}, /* keepDims = */ true);
  Shape tiledims(std::vector<Dim>(input.ndim(), 1));
  if (dim > 3) {
    throw std::invalid_argument("logSoftmax: Dimension must be less than 3");
  }
  tiledims[dim] = input.dim(dim);
  // Compute log softmax.
  // Subtracting then adding maxvals is for numerical stability.
  auto result = input -
      fl::tile(fl::log(fl::sum(
                   fl::exp(input - fl::tile(maxvals, tiledims)),
                   {dim},
                   /* keepDims = */ true)) +
                   maxvals,
               tiledims);
  fl::eval(result);
  return result;
};

} // namespace

namespace fl::pkg::speech {

ConnectionistTemporalClassificationCriterion::
    ConnectionistTemporalClassificationCriterion(
        fl::lib::seq::CriterionScaleMode
            scalemode /* = fl::lib::seq::CriterionScaleMode::NONE */)
    : scaleMode_(scalemode) {}

std::unique_ptr<Module> ConnectionistTemporalClassificationCriterion::clone()
    const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'ConnectionistTemporalClassificationCriterion'");
}

Tensor ConnectionistTemporalClassificationCriterion::viterbiPath(
    const Tensor& input,
    const Tensor& inputSize /* = Tensor() */) {
  Tensor bestpath, maxvalues;
  fl::max(maxvalues, bestpath, input, 0);
  return bestpath;
}

Tensor ConnectionistTemporalClassificationCriterion::viterbiPathWithTarget(
    const Tensor& input,
    const Tensor& target,
    const Tensor& inputSizes /* = Tensor() */,
    const Tensor& targetSizes /* = Tensor() */
) {
  if (input.ndim() != 3) {
    throw std::invalid_argument(
        "ConnectionistTemporalClassificationCriterion::viterbiPathWithTarget: "
        "expected input of shape {N x T x B}");
  }
  int N = input.dim(0);
  int T = input.dim(1);
  int B = input.dim(2);
  int L = target.dim(0);

  const Tensor targetSize = getTargetSizeArray(target, T);
  std::shared_ptr<CTCContext> ctx = std::make_shared<CTCContext>();
  Tensor softmax = ::logSoftmax(input, 0);
  std::vector<float> inputVec = softmax.toHostVector<float>();
  ctx->targetVec = target.toHostVector<int>();
  ctx->targetSizeVec = targetSize.toHostVector<int>();
  ctx->workspaceVec.assign(CTC::getWorkspaceSize(B, T, N, L), 0);
  std::vector<int> bestPaths(B * T);
  CTC::viterbi(
      B,
      T,
      N,
      L,
      inputVec.data(),
      ctx->targetVec.data(),
      ctx->targetSizeVec.data(),
      bestPaths.data(),
      ctx->workspaceVec.data());
  Tensor result =
      Tensor::fromBuffer({T, B}, bestPaths.data(), MemoryLocation::Host);
  return result;
}

std::string ConnectionistTemporalClassificationCriterion::prettyString() const {
  return "ConnectionistTemporalClassificationCriterion";
}

void ConnectionistTemporalClassificationCriterion::validate(
    const Variable& input,
    const Variable& target) {
  if (input.isEmpty()) {
    throw std::invalid_argument("CTC: Input cannot be empty");
  }
  if (target.ndim() < 2) {
    throw std::invalid_argument(
        "CTC: Incorrect dimensions for target. Expected {L, B}, got " +
        target.shape().toString());
  }
  if (input.ndim() < 3) {
    throw std::invalid_argument(
        "CTC: Incorrect dimensions for input. Expected {N, T, B}, got " +
        input.shape().toString());
  }
  if (input.dim(2) != target.dim(1)) {
    throw std::invalid_argument(
        "CTC: Batchsize mismatch for input and target with dims " +
        input.shape().toString() + " and " + target.shape().toString() +
        ", respectively");
  }
}
} // namespace fl
