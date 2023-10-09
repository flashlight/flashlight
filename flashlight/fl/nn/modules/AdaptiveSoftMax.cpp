/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/modules/AdaptiveSoftMax.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

AdaptiveSoftMax::AdaptiveSoftMax(
    int inputSize,
    const std::vector<int>& cutoff,
    float divValue)
    : UnaryModule(), cutoff_(cutoff), divValue_(divValue) {
  if (cutoff_.empty()) {
    throw std::invalid_argument("invalid cutoff for AdaptiveSoftMaxLoss");
  }

  int outputSize = cutoff_[0] + cutoff_.size() - 1;

  auto head = kaimingUniform(
      {outputSize, inputSize}, inputSize /* fanIn */, fl::dtype::f32, true);
  params_.push_back(head);

  int denominator = 1;
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    denominator *= divValue_;
    int hiddenSize = inputSize / denominator;
    auto tail1 = kaimingUniform(
        {hiddenSize, inputSize}, inputSize /* fanIn */, fl::dtype::f32, true);
    auto tail2 = kaimingUniform(
        {cutoff_[i + 1] - cutoff_[i], hiddenSize},
        hiddenSize /* fanIn */,
        fl::dtype::f32,
        true);

    params_.push_back(tail1);
    params_.push_back(tail2);
  }
}

Variable AdaptiveSoftMax::getFullLogProb(
    const Variable& inputs,
    const Variable& headOutput) const {
  auto outputSize = cutoff_[cutoff_.size() - 1];
  auto batchSize = inputs.dim(1);
  Tensor output({outputSize, batchSize}, inputs.type());

  output(
      fl::range(0, cutoff_[0] + static_cast<long long>(cutoff_.size()) - 1)) =
      headOutput.tensor();

  for (int i = cutoff_.size() - 2; i >= 0; i--) {
    auto tailOutput = matmul(params_[1 + i * 2], inputs);
    tailOutput = matmul(params_[2 + i * 2], tailOutput);
    auto idx = i + cutoff_[0];
    tailOutput = logSoftmax(tailOutput, 0) +
        tileAs(headOutput(fl::range(idx, idx + 1)), tailOutput);
    output(fl::range(cutoff_[i], cutoff_[i + 1])) = tailOutput.tensor();
  }

  return Variable(output, false);
}

Variable AdaptiveSoftMax::forward(const Variable& inputs) {
  // input -- [C_in, .. , N]
  // return -- [C_out, .. , N]
  auto inputSize = inputs.dim(0);
  if (inputSize != params_[0].dim(1)) {
    throw std::invalid_argument("invalid input dimension for AdaptiveSoftMax");
  }

  auto inputsFlattened = moddims(inputs, {inputSize, -1});
  auto headOutput = logSoftmax(matmul(params_[0], inputsFlattened), 0);

  auto ret = getFullLogProb(inputsFlattened, headOutput);

  Shape outDims = inputs.shape();
  outDims[0] = ret.dim(0);
  return moddims(ret, outDims);
}

Variable AdaptiveSoftMax::predict(const Variable& inputs) const {
  // input -- [C, .. , N]
  // return -- [1, .. , N]
  auto inputSize = inputs.dim(0);
  if (inputSize != params_[0].dim(1)) {
    throw std::invalid_argument(
        "invalid input dimension for AdaptiveSoftMaxLoss");
  }

  auto inputsFlattened = moddims(inputs, {inputSize, -1});
  auto headOutput = matmul(params_[0], inputsFlattened);
  Tensor maxValue, prediction;
  fl::max(maxValue, prediction, headOutput.tensor(), 0);

  auto notInShortlist = (prediction >= cutoff_[0]);
  Variable ret = Variable(prediction, false);
  if (fl::any(notInShortlist).asScalar<bool>()) {
    headOutput = logSoftmax(headOutput, 0);
    auto logProbTailPositions = getFullLogProb(
        inputsFlattened(fl::span, notInShortlist),
        headOutput(fl::span, notInShortlist));
    Tensor maxValueTailPositions, predictionTailPositions;
    fl::max(
        maxValueTailPositions,
        predictionTailPositions,
        logProbTailPositions.tensor(),
        0);
    ret.tensor()(notInShortlist) = predictionTailPositions;
  }

  Shape outDims = inputs.shape();
  outDims[0] = 1;
  return moddims(ret, outDims);
}

std::vector<int> AdaptiveSoftMax::getCutoff() const {
  return cutoff_;
}

std::unique_ptr<Module> AdaptiveSoftMax::clone() const {
  return std::make_unique<AdaptiveSoftMax>(*this);
}

std::string AdaptiveSoftMax::prettyString() const {
  std::ostringstream ss;
  ss << "Adaptive Softmax (";
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    ss << cutoff_[i] << ", ";
  }
  ss << cutoff_[cutoff_.size() - 1] << ")";
  return ss.str();
}

} // namespace fl
