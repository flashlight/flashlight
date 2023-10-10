/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/nn/modules/Module.h"

#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Module::Module() = default;

Module::Module(const std::vector<Variable>& params)
    : params_(params.begin(), params.end()) {}

Variable Module::param(int position) const {
  if (!(position >= 0 && position < params_.size())) {
    throw std::out_of_range("Module param index out of range");
  }
  return params_[position];
}

void Module::setParams(const Variable& var, int position) {
  if (!(position >= 0 && position < params_.size())) {
    throw std::out_of_range("Module param index out of range");
  }
  params_[position] = var;
}

std::vector<Variable> Module::copyParams() const {
  std::vector<Variable> params;
  params.reserve(params_.size());
  for (const auto& param : params_) {
    params.emplace_back(param.copy());
  }
  return params;
}

void Module::train() {
  train_ = true;
  for (auto& param : params_) {
    param.setCalcGrad(true);
  }
}

void Module::zeroGrad() {
  for (auto& param : params_) {
    param.zeroGrad();
  }
}

void Module::eval() {
  train_ = false;
  for (auto& param : params_) {
    param.setCalcGrad(false);
  }
}

std::vector<Variable> Module::params() const {
  return params_;
}

int Module::numParamTensors() const {
  return static_cast<int>(params_.size());
}

std::vector<Variable> Module::operator()(const std::vector<Variable>& input) {
  return this->forward(input);
}

UnaryModule::UnaryModule() = default;

UnaryModule::UnaryModule(const std::vector<Variable>& params)
    : Module(params) {}

std::vector<Variable> UnaryModule::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("UnaryModule expects only one input");
  }
  return {forward(inputs[0])};
}

Variable UnaryModule::operator()(const Variable& input) {
  return this->forward(input);
}

BinaryModule::BinaryModule() = default;

BinaryModule::BinaryModule(const std::vector<Variable>& params)
    : Module(params) {}

std::vector<Variable> BinaryModule::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("BinaryModule expects two inputs");
  }
  return {forward(inputs[0], inputs[1])};
}

Variable BinaryModule::operator()(
    const Variable& input1,
    const Variable& input2) {
  return this->forward(input1, input2);
}

} // namespace fl
