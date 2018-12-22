/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "AdaptiveSoftMaxLoss.h"

#include <flashlight/autograd/Functions.h>
#include <flashlight/nn/Init.h>

namespace fl {

AdaptiveSoftMaxLoss::AdaptiveSoftMaxLoss(
    int input_size,
    const std::vector<int>& cutoff,
    float div_value,
    ReduceMode reduction)
    : BinaryModule(),
      cutoff_(cutoff),
      reduction_(reduction),
      divValue_(div_value) {
  if (cutoff_.empty()) {
    throw std::invalid_argument("invalid cutoff for AdaptiveSoftMaxLoss");
  }

  int output_size = cutoff_[0] + cutoff_.size() - 1;

  auto head = kaimingUniform(output_size, input_size);
  params_.push_back(head);

  int denominator = 1;
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    denominator *= divValue_;
    int hidden_size = input_size / denominator;
    auto tail1 = kaimingUniform(hidden_size, input_size);
    auto tail2 = kaimingUniform(cutoff_[i + 1] - cutoff_[i], hidden_size);
    params_.push_back(tail1);
    params_.push_back(tail2);
  }
}

void AdaptiveSoftMaxLoss::setTargets(
    const Variable& targets,
    std::vector<af::array>& masks,
    std::vector<Variable>& target_chunks) const {
  auto head_target = Variable(targets.array(), false);
  head_target = head_target && (targets < cutoff_[0]);
  target_chunks[0] = head_target;

  for (int i = 0; i < cutoff_.size() - 1; i++) {
    auto mask1 = targets >= cutoff_[i];
    auto mask2 = targets < cutoff_[i + 1];
    auto mask = mask1 && mask2;
    auto mask_array = where(mask.array());
    masks[i] = mask_array;
    target_chunks[0] = target_chunks[0] + mask * (cutoff_[0] + i);
    auto tail_target = targets(mask_array) - cutoff_[i];
    target_chunks[i + 1] = tail_target;
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

  std::vector<af::array> masks(cutoff_.size() - 1);
  std::vector<Variable> target_chunks(cutoff_.size());
  setTargets(targets, masks, target_chunks);

  // Head forward
  auto head_output = matmul(params_[0], input);
  auto res = categoricalCrossEntropy(
      logSoftmax(head_output, 0), target_chunks[0], ReduceMode::SUM);

  // Tail forwawrd
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    if (masks[i].dims(0) == 0) {
      continue;
    }

    auto selected_input = embedding(Variable(masks[i], false), input);
    auto tail_output = matmul(params_[1 + i * 2], selected_input);
    tail_output = matmul(params_[2 + i * 2], tail_output);
    res =
        res +
        categoricalCrossEntropy(
            logSoftmax(tail_output, 0), target_chunks[i + 1], ReduceMode::SUM);
  }
  if (reduction_ == ReduceMode::MEAN) {
    res = res / input.dims(1);
  }
  return res;
}

Variable AdaptiveSoftMaxLoss::getFullLogProb(
    const Variable& inputs,
    const Variable& head_output) const {
  auto output_size = cutoff_[cutoff_.size() - 1];
  auto batch_size = inputs.dims(1);
  af::array output(af::dim4(output_size, batch_size), inputs.type());

  output.rows(0, cutoff_[0] + cutoff_.size() - 2) = head_output.array();

  for (int i = cutoff_.size() - 2; i >= 0; i--) {
    auto tail_output = matmul(params_[1 + i * 2], inputs);
    tail_output = matmul(params_[2 + i * 2], tail_output);
    tail_output = logSoftmax(tail_output, 0) +
        tileAs(head_output.row(i + cutoff_[0]), tail_output);
    output.rows(cutoff_[i], cutoff_[i + 1] - 1) = tail_output.array();
  }

  return Variable(output, false);
}

Variable AdaptiveSoftMaxLoss::getLogProb(const Variable& inputs) const {
  // input -- [C_in, .. , N]
  // return -- [C_out, .. , N]
  auto input_size = inputs.dims(0);
  if (input_size != params_[0].dims(1)) {
    throw std::invalid_argument(
        "invalid input dimension for AdaptiveSoftMaxLoss");
  }

  auto inputs_flattened = moddims(inputs, af::dim4(input_size, -1, 1, 1));
  auto head_output = logSoftmax(matmul(params_[0], inputs_flattened), 0);

  auto ret = getFullLogProb(inputs_flattened, head_output);
  return moddims(
      ret,
      af::dim4(ret.dims(0), inputs.dims(1), inputs.dims(2), inputs.dims(3)));
}

Variable AdaptiveSoftMaxLoss::predict(const Variable& inputs) const {
  // input -- [C, .. , N]
  // return -- [1, .. , N]
  auto input_size = inputs.dims(0);
  if (input_size != params_[0].dims(1)) {
    throw std::invalid_argument(
        "invalid input dimension for AdaptiveSoftMaxLoss");
  }

  auto inputs_flattened = moddims(inputs, af::dim4(input_size, -1, 1, 1));
  auto head_output = logSoftmax(matmul(params_[0], inputs_flattened), 0);
  af::array max_value, prediction;
  af::max(max_value, prediction, head_output.array(), 0);

  auto not_in_shortlist = (prediction >= cutoff_[0]);
  Variable ret;
  if (!af::anyTrue<bool>(not_in_shortlist)) {
    ret = Variable(prediction, false);
  } else {
    auto log_prob = getFullLogProb(inputs, head_output);
    af::max(max_value, prediction, log_prob.array(), 0);
    ret = Variable(prediction, false);
  }

  return moddims(
      ret,
      af::dim4(ret.dims(0), inputs.dims(1), inputs.dims(2), inputs.dims(3)));
}

std::string AdaptiveSoftMaxLoss::prettyString() const {
  std::ostringstream ss;
  ss << "Adaptive Softmax (";
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    ss << cutoff_[i] << ", ";
  }
  ss << cutoff_[cutoff_.size() - 1] << ")";
  return ss.str();
}

} // namespace fl
