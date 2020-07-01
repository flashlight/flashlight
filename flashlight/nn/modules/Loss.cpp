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
    ReduceMode reduction,
    int ignoreIndex)
    : BinaryModule(),
      activation_(activation),
      reduction_(reduction),
      ignoreIndex_(ignoreIndex) {
  params_ = activation_->params();
}

Variable AdaptiveSoftMaxLoss::cast(
    const Variable& input,
    const af::dim4& outDims,
    const af::array& indices) {
  if (input.elements() != indices.elements()) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: input, indices mismatch");
  }
  af::array output = af::constant(0, outDims, input.type());
  output(indices) = af::flat(input.array());
  auto inputDims = input.dims();

  auto gradFunc = [indices, inputDims](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    af::array gradArray = grad_output.array()(indices);
    auto grad = Variable(af::moddims(gradArray, inputDims), false);
    inputs[0].addGrad(grad);
  };
  return Variable(output, {input.withoutData()}, gradFunc);
}

Variable AdaptiveSoftMaxLoss::forward(
    const Variable& inputs,
    const Variable& targets) {
  // inputs: N x T x B x 1
  // targets: T x B x 1 x 1
  if (inputs.dims(1) != targets.dims(0)) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: length mismatch");
  } else if (inputs.dims(2) != targets.dims(1)) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: batch size mismatch");
  }

  auto N = inputs.dims(0);
  auto T = inputs.dims(1);
  auto B = inputs.dims(2);
  auto cutoff = activation_->getCutoff();

  auto input = moddims(inputs, af::dim4(N, T * B));
  auto target = moddims(targets, af::dim4(T * B));

  auto headOutput = matmul(params_[0], input);
  auto headTarget = Variable(target.array(), false) * (target < cutoff[0]);
  auto res = Variable(af::constant(0, T * B), true);

  // Tail forwawrd
  for (int i = 0; i < cutoff.size() - 1; i++) {
    auto mask = (target >= cutoff[i]) && (target < cutoff[i + 1]);
    if (!af::anyTrue<bool>(mask.array())) {
      continue;
    }

    auto indicesArray = af::where(mask.array());
    headTarget = headTarget + mask * (cutoff[0] + i);
    auto tailTarget = target(indicesArray) - cutoff[i];
    auto selectedInput = embedding(Variable(indicesArray, false), input);
    auto tailOutput = matmul(params_[1 + i * 2], selectedInput);
    tailOutput = matmul(params_[2 + i * 2], tailOutput);
    auto localLoss = categoricalCrossEntropy(
        logSoftmax(tailOutput, 0), tailTarget, ReduceMode::NONE, ignoreIndex_);
    res = res + cast(localLoss, res.dims(), indicesArray);
  }

  // Head forward
  res = res +
      categoricalCrossEntropy(
            logSoftmax(headOutput, 0),
            headTarget,
            ReduceMode::NONE,
            ignoreIndex_);

  // Reduce
  if (reduction_ == ReduceMode::NONE) {
    return moddims(res, targets.dims());
  }
  res = sum(res, {0});
  if (reduction_ == ReduceMode::MEAN) {
    auto denominator = af::count<int>(target.array() != ignoreIndex_);
    res = res / denominator;
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
