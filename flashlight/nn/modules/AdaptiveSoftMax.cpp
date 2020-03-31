/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <vector>

#include "flashlight/autograd/Functions.h"
#include "flashlight/nn/Init.h"
#include "flashlight/nn/modules/AdaptiveSoftMax.h"

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

  // TODO add this back once initialization has landed
  //auto head = kaimingUniform(outputSize, inputSize);
  //params_.push_back(head);

  //int denominator = 1;
  //for (int i = 0; i < cutoff_.size() - 1; i++) {
    //denominator *= divValue_;
    //int hiddenSize = inputSize / denominator;
    //auto tail1 = kaimingUniform(hiddenSize, inputSize);
    //auto tail2 = kaimingUniform(cutoff_[i + 1] - cutoff_[i], hiddenSize);
    //params_.push_back(tail1);
    //params_.push_back(tail2);
  //}
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
  auto headOutput = logSoftmax(matmul(params_[0], inputsFlattened), 0);
  af::array maxValue, prediction;
  af::max(maxValue, prediction, headOutput.array(), 0);

  auto notInShortlist = (prediction >= cutoff_[0]);
  Variable ret;
  if (!af::anyTrue<bool>(notInShortlist)) {
    ret = Variable(prediction, false);
  } else {
    auto logProb = getFullLogProb(inputs, headOutput);
    af::max(maxValue, prediction, logProb.array(), 0);
    ret = Variable(prediction, false);
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
