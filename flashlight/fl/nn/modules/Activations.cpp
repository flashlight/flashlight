/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "flashlight/fl/nn/modules/Activations.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

Sigmoid::Sigmoid() = default;

Variable Sigmoid::forward(const Variable& input) {
  return sigmoid(input);
}

std::string Sigmoid::prettyString() const {
  return "Sigmoid";
}

Log::Log() = default;

Variable Log::forward(const Variable& input) {
  return log(input);
}

std::string Log::prettyString() const {
  return "Log";
}

Tanh::Tanh() = default;

Variable Tanh::forward(const Variable& input) {
  return tanh(input);
}

std::string Tanh::prettyString() const {
  return "Tanh";
}

HardTanh::HardTanh() = default;

Variable HardTanh::forward(const Variable& input) {
  return clamp(input, -1.0, 1.0);
}

std::string HardTanh::prettyString() const {
  return "HardTanh";
}

ReLU::ReLU() = default;

Variable ReLU::forward(const Variable& input) {
  return max(input, 0.0);
}

std::string ReLU::prettyString() const {
  return "ReLU";
}

ReLU6::ReLU6() = default;

Variable ReLU6::forward(const Variable& input) {
  return clamp(input, 0.0, 6.0);
}

std::string ReLU6::prettyString() const {
  return "ReLU6";
}

LeakyReLU::LeakyReLU(double slope) : mSlope_(slope) {}

Variable LeakyReLU::forward(const Variable& input) {
  return max(input, mSlope_ * input);
}

std::string LeakyReLU::prettyString() const {
  return "LeakyReLU (" + std::to_string(mSlope_) + ")";
}

PReLU::PReLU(const Variable& w) : UnaryModule({w}) {}

PReLU::PReLU(int size, double value) {
  auto w = constant(value, size, 1);
  params_ = {w};
}

Variable PReLU::forward(const Variable& input) {
  auto mask = input >= 0.0;
  return (input * mask) + (input * !mask * tileAs(params_[0], input));
}

std::string PReLU::prettyString() const {
  return "PReLU";
}

ELU::ELU(double alpha) : mAlpha_(alpha) {}

Variable ELU::forward(const Variable& input) {
  auto mask = input >= 0.0;
  return (mask * input) + (!mask * mAlpha_ * (exp(input) - 1));
}

std::string ELU::prettyString() const {
  return "ELU (" + std::to_string(mAlpha_) + ")";
}

ThresholdReLU::ThresholdReLU(double threshold) : mThreshold_(threshold) {}

Variable ThresholdReLU::forward(const Variable& input) {
  auto mask = input >= mThreshold_;
  return input * mask;
}

std::string ThresholdReLU::prettyString() const {
  return "ThresholdReLU (" + std::to_string(mThreshold_) + ")";
}

GatedLinearUnit::GatedLinearUnit(int dim) : dim_(dim) {}

Variable GatedLinearUnit::forward(const Variable& input) {
  return gatedlinearunit(input, dim_);
}

std::string GatedLinearUnit::prettyString() const {
  return "GatedLinearUnit (" + std::to_string(dim_) + ")";
}

LogSoftmax::LogSoftmax(int dim /* = 0 */) : dim_(dim) {}

Variable LogSoftmax::forward(const Variable& input) {
  return logSoftmax(input, dim_);
}

std::string LogSoftmax::prettyString() const {
  return "LogSoftmax (" + std::to_string(dim_) + ")";
}

Swish::Swish(double beta /* = 1.0 */) : beta_(beta) {}

Variable Swish::forward(const Variable& input) {
  return swish(input, beta_);
}

std::string Swish::prettyString() const {
  return "Swish (" + std::to_string(beta_) + ")";
}

} // namespace fl
