/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Loss.h"
#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

Variable MeanSquaredError::forward(
    const Variable& inputs,
    const Variable& targets) {
  if (inputs.shape() != targets.shape()) {
    throw std::invalid_argument(
        "MeanSquaredError::forward - inputs and targets are of different"
        " sizes: {inputs: " +
        inputs.shape().toString() + ", targets: " + targets.shape().toString() +
        "}");
  }

  auto df = inputs - targets;
  auto res = mean(flat(df * df), {0});
  return res;
}

std::unique_ptr<Module> MeanSquaredError::clone() const {
  return std::make_unique<MeanSquaredError>(*this);
}

std::string MeanSquaredError::prettyString() const {
  return "MeanSquaredError";
}

Variable MeanAbsoluteError::forward(
    const Variable& inputs,
    const Variable& targets) {
  if (inputs.shape() != targets.shape()) {
    throw std::invalid_argument(
        "MeanAbsoluteError::forward - inputs and targets are of different"
        " sizes: {inputs: " +
        inputs.shape().toString() + ", targets: " + targets.shape().toString() +
        "}");
  }

  auto df = inputs - targets;
  return mean(flat(fl::abs(df)), {0});
}

std::unique_ptr<Module> MeanAbsoluteError::clone() const {
  return std::make_unique<MeanAbsoluteError>(*this);
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

std::unique_ptr<Module> BinaryCrossEntropy::clone() const {
  return std::make_unique<BinaryCrossEntropy>(*this);
}

std::string BinaryCrossEntropy::prettyString() const {
  return "BinaryCrossEntropy";
}

Variable CategoricalCrossEntropy::forward(
    const Variable& inputs,
    const Variable& targets) {
  return categoricalCrossEntropy(inputs, targets, reduction_, ignoreIndex_);
}

std::unique_ptr<Module> CategoricalCrossEntropy::clone() const {
  return std::make_unique<CategoricalCrossEntropy>(*this);
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
    const Shape& outDims,
    const Tensor& indices) {
  if (input.elements() != indices.elements()) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: input, indices mismatch");
  }
  Tensor output = fl::full(outDims, 0, input.type());
  output(indices) = input.tensor().flatten();
  auto inputDims = input.shape();

  auto gradFunc = [indices, inputDims](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    Tensor gradTensor = grad_output.tensor()(indices);
    auto grad = Variable(fl::reshape(gradTensor, inputDims), false);
    inputs[0].addGrad(grad);
  };
  return Variable(output, {input.withoutData()}, gradFunc);
}

Variable AdaptiveSoftMaxLoss::forward(
    const Variable& inputs,
    const Variable& targets) {
  // inputs: N x T x B
  // targets: T x B
  if (inputs.ndim() != 3) {
    throw std::invalid_argument(
        "AdaptiveSoftMaxLoss::forward expects input tensor with "
        "3 dimensions in N x T x B ordering.");
  }
  if (targets.ndim() != 2) {
    throw std::invalid_argument(
        "AdaptiveSoftMaxLoss::forward expects target tensor with "
        "2 dimensions in T x B ordering.");
  }
  if (inputs.dim(1) != targets.dim(0)) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: length mismatch");
  } else if (inputs.dim(2) != targets.dim(1)) {
    throw std::invalid_argument("AdaptiveSoftMaxLoss: batch size mismatch");
  }

  auto N = inputs.dim(0);
  auto T = inputs.dim(1);
  auto B = inputs.dim(2);
  auto cutoff = activation_->getCutoff();

  auto input = moddims(inputs, {N, T * B});
  auto target = moddims(targets, {T * B});

  auto headOutput = matmul(params_[0], input);
  auto headTarget = Variable(target.tensor(), false) * (target < cutoff[0]);
  // TODO: check the type of res
  auto res = Variable(fl::full({T * B}, 0, fl::dtype::f32), true);

  // Tail forwawrd
  for (int i = 0; i < cutoff.size() - 1; i++) {
    auto mask = (target >= cutoff[i]) && (target < cutoff[i + 1]);
    if (!fl::any(mask.tensor()).scalar<char>()) {
      continue;
    }

    auto indicesArray = fl::nonzero(mask.tensor());
    headTarget =
        headTarget + (mask * (cutoff[0] + i)).astype(headTarget.type());
    auto tailTarget = target(indicesArray) - cutoff[i];
    auto selectedInput = embedding(Variable(indicesArray, false), input);
    auto tailOutput = matmul(params_[1 + i * 2], selectedInput);
    tailOutput = matmul(params_[2 + i * 2], tailOutput);
    auto localLoss = categoricalCrossEntropy(
        logSoftmax(tailOutput, 0), tailTarget, ReduceMode::NONE, ignoreIndex_);
    res = res + cast(localLoss, res.shape(), indicesArray);
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
    return moddims(res, targets.shape());
  }
  res = sum(res, {0});
  if (reduction_ == ReduceMode::MEAN) {
    auto denominator =
        fl::countNonzero(target.tensor() != ignoreIndex_).scalar<unsigned>();
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

std::unique_ptr<Module> AdaptiveSoftMaxLoss::clone() const {
  return std::make_unique<AdaptiveSoftMaxLoss>(*this);
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
