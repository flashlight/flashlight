/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/modules/AdaptiveSoftMax.h"

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
      af::dim4(outputSize, inputSize),
      inputSize /* fanIn */,
      af::dtype::f32,
      true);
  params_.push_back(head);

  int denominator = 1;
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    denominator *= divValue_;
    int hiddenSize = inputSize / denominator;
    auto tail1 = kaimingUniform(
        af::dim4(hiddenSize, inputSize),
        inputSize /* fanIn */,
        af::dtype::f32,
        true);
    auto tail2 = kaimingUniform(
        af::dim4(cutoff_[i + 1] - cutoff_[i], hiddenSize),
        hiddenSize /* fanIn */,
        af::dtype::f32,
        true);

    params_.push_back(tail1);
    params_.push_back(tail2);
  }
}

Variable AdaptiveSoftMax::getFullLogProb(
    const Variable& inputs,
    const Variable& headOutput) const {
  auto outputSize = cutoff_[cutoff_.size() - 1];
  auto batchSize = inputs.dims(1);
  af::array output(af::dim4(outputSize, batchSize), inputs.type());

  output.rows(0, cutoff_[0] + cutoff_.size() - 2) = headOutput.array();

  for (int i = cutoff_.size() - 2; i >= 0; i--) {
    auto tailOutput = matmul(params_[1 + i * 2], inputs);
    tailOutput = matmul(params_[2 + i * 2], tailOutput);
    tailOutput = logSoftmax(tailOutput, 0) +
        tileAs(headOutput.row(i + cutoff_[0]), tailOutput);
    output.rows(cutoff_[i], cutoff_[i + 1] - 1) = tailOutput.array();
  }

  return Variable(output, false);
}

Variable AdaptiveSoftMax::forward(const Variable& inputs) {
  // input -- [C_in, .. , N]
  // return -- [C_out, .. , N]
  auto inputSize = inputs.dims(0);
  if (inputSize != params_[0].dims(1)) {
    throw std::invalid_argument("invalid input dimension for AdaptiveSoftMax");
  }

  auto inputsFlattened = moddims(inputs, af::dim4(inputSize, -1, 1, 1));
  auto headOutput = logSoftmax(matmul(params_[0], inputsFlattened), 0);

  auto ret = getFullLogProb(inputsFlattened, headOutput);
  return moddims(
      ret,
      af::dim4(ret.dims(0), inputs.dims(1), inputs.dims(2), inputs.dims(3)));
}

Variable AdaptiveSoftMax::predict(const Variable& inputs) const {
  // input -- [C, .. , N]
  // return -- [1, .. , N]
  auto inputSize = inputs.dims(0);
  if (inputSize != params_[0].dims(1)) {
    throw std::invalid_argument(
        "invalid input dimension for AdaptiveSoftMaxLoss");
  }

  auto inputsFlattened = moddims(inputs, af::dim4(inputSize, -1, 1, 1));
  auto headOutput = matmul(params_[0], inputsFlattened);
  af::array maxValue, prediction;
  af::max(maxValue, prediction, headOutput.array(), 0);

  auto notInShortlist = (prediction >= cutoff_[0]);
  Variable ret = Variable(prediction, false);
  if (af::anyTrue<bool>(notInShortlist)) {
    headOutput = logSoftmax(headOutput, 0);
    auto logProbTailPositions = getFullLogProb(
        inputsFlattened(af::span, notInShortlist),
        headOutput(af::span, notInShortlist));
    af::array maxValueTailPositions, predictionTailPositions;
    af::max(
        maxValueTailPositions,
        predictionTailPositions,
        logProbTailPositions.array(),
        0);
    ret.array()(notInShortlist) = predictionTailPositions;
  }
  return moddims(
      ret,
      af::dim4(ret.dims(0), inputs.dims(1), inputs.dims(2), inputs.dims(3)));
}

std::vector<int> AdaptiveSoftMax::getCutoff() const {
  return cutoff_;
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
