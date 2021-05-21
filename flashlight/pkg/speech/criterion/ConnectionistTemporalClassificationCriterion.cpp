/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "flashlight/pkg/speech/criterion/ConnectionistTemporalClassificationCriterion.h"

#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/lib/sequence/criterion/cpu/ConnectionistTemporalClassificationCriterion.h"

using CTC = fl::lib::cpu::ConnectionistTemporalClassificationCriterion<float>;
using namespace fl::ext;

namespace {

struct CTCContext {
  std::vector<int> targetVec;
  std::vector<int> targetSizeVec;
  std::vector<uint8_t> workspaceVec;
};

af::array logSoftmax(const af::array& input, const int dim) {
  af::array maxvals = max((input), dim);
  af::dim4 tiledims(1, 1, 1, 1);
  if (dim > 3) {
    throw std::invalid_argument("logSoftmax: Dimension must be less than 3");
  }
  tiledims[dim] = input.dims(dim);
  // Compute log softmax.
  // Subtracting then adding maxvals is for numerical stability.
  auto result = input -
      tile(log(sum(exp(input - tile(maxvals, tiledims)), dim)) + maxvals,
           tiledims);
  fl::eval(result);
  return result;
};

} // namespace

namespace fl {
namespace app {
namespace asr {

ConnectionistTemporalClassificationCriterion::
    ConnectionistTemporalClassificationCriterion(
        fl::lib::seq::CriterionScaleMode
            scalemode /* = fl::lib::seq::CriterionScaleMode::NONE */)
    : scaleMode_(scalemode) {}

af::array ConnectionistTemporalClassificationCriterion::viterbiPath(
    const af::array& input,
    const af::array& inputSize /* = af::array() */) {
  af::array bestpath, maxvalues;
  af::max(maxvalues, bestpath, input, 0);
  return af::moddims(bestpath, bestpath.dims(1), bestpath.dims(2));
}

af::array ConnectionistTemporalClassificationCriterion::viterbiPathWithTarget(
    const af::array& input,
    const af::array& target,
    const af::array& inputSizes /* = af::array() */,
    const af::array& targetSizes /* = af::array() */
) {
  int N = input.dims(0);
  int T = input.dims(1);
  int B = input.dims(2);
  int L = target.dims(0);

  const af::array targetSize = getTargetSizeArray(target, T);
  std::shared_ptr<CTCContext> ctx = std::make_shared<CTCContext>();
  af::array softmax = ::logSoftmax(input, 0);
  std::vector<float> inputVec = afToVector<float>(softmax);
  ctx->targetVec = afToVector<int>(target);
  ctx->targetSizeVec = afToVector<int>(targetSize);
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
  af::array result(T, B, bestPaths.data());
  return result;
}

std::string ConnectionistTemporalClassificationCriterion::prettyString() const {
  return "ConnectionistTemporalClassificationCriterion";
}

void ConnectionistTemporalClassificationCriterion::validate(
    const Variable& input,
    const Variable& target) {
  if (input.isempty()) {
    throw(af::exception("CTC: Input cannot be empty"));
  }
  if (target.numdims() > 2) {
    throw(af::exception(
        "CTC: Incorrect dimensions for target. Expected dim4(L, B)"));
  }
  if (input.numdims() > 3) {
    throw(af::exception(
        "CTC: Incorrect dimensions for input. Expected dim4(N, T, B)"));
  }
  if (input.dims(2) != target.dims(1)) {
    throw(af::exception("CTC: Batchsize mismatch for input and target"));
  }
}
} // namespace asr
} // namespace app
} // namespace fl
