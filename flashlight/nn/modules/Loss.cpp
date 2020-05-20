/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "flashlight/nn/modules/Loss.h"
#include <stdexcept>
#include <vector>

#include "flashlight/autograd/Functions.h"

namespace fl {

Variable MeanSquaredError::forward(
    const Variable& inputs,
    const Variable& targets) {
  auto df = inputs - targets;
  auto res = mean(flat(df * df), {0});
  return res;
}

std::string MeanSquaredError::prettyString() const {
  return "MeanSquaredError";
}

Variable MeanAbsoluteError::forward(
    const Variable& inputs,
    const Variable& targets) {
  auto df = inputs - targets;
  return mean(flat(fl::abs(df)), {0});
}

std::string MeanAbsoluteError::prettyString() const {
  return "MeanAbsoluteError";
}

Variable BinaryCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets) {
  return mean(flat(binaryCrossEntropy(inputs, targets)), {0});
}

Variable BinaryCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets,
    const Variable& weights) {
  return mean(flat(weights * binaryCrossEntropy(inputs, targets)), {0});
}

std::string BinaryCrossEntropy::prettyString() const {
  return "BinaryCrossEntropy";
}

Variable CategoricalCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets) {
  return categoricalCrossEntropy(inputs, targets, reduction_, ignoreIndex_);
}

std::string CategoricalCrossEntropy::prettyString() const {
  return "CategoricalCrossEntropy";
}

AdaptiveSoftMaxLoss::AdaptiveSoftMaxLoss(
    std::shared_ptr<AdaptiveSoftMax> activation,
    ReduceMode reduction)
    : BinaryModule(), activation_(activation), reduction_(reduction) {
  params_ = activation_->params();
}

void AdaptiveSoftMaxLoss::setTargets(
    const Variable& targets,
    std::vector<af::array>& masks,
    std::vector<Variable>& targetChunks,
    std::vector<int>& cutoff) const {
  auto headTarget = Variable(targets.array(), false);
  headTarget = headTarget && (targets < cutoff[0]);
  targetChunks[0] = headTarget;

  for (int i = 0; i < cutoff.size() - 1; i++) {
    auto mask1 = targets >= cutoff[i];
    auto mask2 = targets < cutoff[i + 1];
    auto mask = mask1 && mask2;
    auto maskArray = where(mask.array());
    masks[i] = maskArray;
    targetChunks[0] = targetChunks[0] + mask * (cutoff[0] + i);
    auto tailTarget = targets(maskArray) - cutoff[i];
    targetChunks[i + 1] = tailTarget;
  }
}

Variable AdaptiveSoftMaxLoss::forward(
    const Variable& input,
    const Variable& targets) {
  if (input.numdims() != 2) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss only supports 2D inputs");
  }
  if (targets.numdims() != 1) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss only supports 1D targets");
  }
  auto cutoff = activation_->getCutoff();

  std::vector<af::array> masks(cutoff.size() - 1);
  std::vector<Variable> targetChunks(cutoff.size());
  setTargets(targets, masks, targetChunks, cutoff);

  // Head forward
  auto headOutput = matmul(params_[0], input);
  auto res = categoricalCrossEntropy(
      logSoftmax(headOutput, 0), targetChunks[0], ReduceMode::SUM);

  // Tail forwawrd
  for (int i = 0; i < cutoff.size() - 1; i++) {
    if (masks[i].dims(0) == 0) {
      continue;
    }

    auto selectedInput = embedding(Variable(masks[i], false), input);
    auto tailOutput = matmul(params_[1 + i * 2], selectedInput);
    tailOutput = matmul(params_[2 + i * 2], tailOutput);
    res = res +
        categoricalCrossEntropy(
              logSoftmax(tailOutput, 0), targetChunks[i + 1], ReduceMode::SUM);
  }
  if (reduction_ == ReduceMode::MEAN) {
    res = res / input.dims(1);
  }
  return res;
}

std::shared_ptr<AdaptiveSoftMax> AdaptiveSoftMaxLoss::getActivation() const {
  return activation_;
};

void AdaptiveSoftMaxLoss::setParams(const Variable& var, int position) {
  Module::setParams(var, position);
  activation_->setParams(var, position);
}

std::string AdaptiveSoftMaxLoss::prettyString() const {
  std::ostringstream ss;
  auto cutoff = activation_->getCutoff();
  ss << "Adaptive Softmax (";
  for (int i = 0; i < cutoff.size() - 1; i++) {
    ss << cutoff[i] << ", ";
  }
  ss << cutoff[cutoff.size() - 1] << ")";
  return ss.str();
}

} // namespace fl
