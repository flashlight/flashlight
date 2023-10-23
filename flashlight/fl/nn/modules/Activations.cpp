/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Activations.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

Sigmoid::Sigmoid() = default;

Variable Sigmoid::forward(const Variable& input) {
  return sigmoid(input);
}

std::unique_ptr<Module> Sigmoid::clone() const {
  return std::make_unique<Sigmoid>(*this);
}

std::string Sigmoid::prettyString() const {
  return "Sigmoid";
}

Log::Log() = default;

Variable Log::forward(const Variable& input) {
  return log(input);
}

std::unique_ptr<Module> Log::clone() const {
  return std::make_unique<Log>(*this);
}

std::string Log::prettyString() const {
  return "Log";
}

Tanh::Tanh() = default;

Variable Tanh::forward(const Variable& input) {
  return tanh(input);
}

std::unique_ptr<Module> Tanh::clone() const {
  return std::make_unique<Tanh>(*this);
}

std::string Tanh::prettyString() const {
  return "Tanh";
}

HardTanh::HardTanh() = default;

Variable HardTanh::forward(const Variable& input) {
  return clamp(input, -1.0, 1.0);
}

std::unique_ptr<Module> HardTanh::clone() const {
  return std::make_unique<HardTanh>(*this);
}

std::string HardTanh::prettyString() const {
  return "HardTanh";
}

ReLU::ReLU() = default;

Variable ReLU::forward(const Variable& input) {
  return max(input, 0.0);
}

std::unique_ptr<Module> ReLU::clone() const {
  return std::make_unique<ReLU>(*this);
}

std::string ReLU::prettyString() const {
  return "ReLU";
}

ReLU6::ReLU6() = default;

Variable ReLU6::forward(const Variable& input) {
  return clamp(input, 0.0, 6.0);
}

std::unique_ptr<Module> ReLU6::clone() const {
  return std::make_unique<ReLU6>(*this);
}

std::string ReLU6::prettyString() const {
  return "ReLU6";
}

LeakyReLU::LeakyReLU(double slope) : mSlope_(slope) {}

Variable LeakyReLU::forward(const Variable& input) {
  return max(input, mSlope_ * input);
}

std::unique_ptr<Module> LeakyReLU::clone() const {
  return std::make_unique<LeakyReLU>(*this);
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

std::unique_ptr<Module> PReLU::clone() const {
  return std::make_unique<PReLU>(*this);
}

std::string PReLU::prettyString() const {
  return "PReLU";
}

ELU::ELU(double alpha) : mAlpha_(alpha) {}

Variable ELU::forward(const Variable& input) {
  auto mask = input >= 0.0;
  return (mask * input) + (!mask * mAlpha_ * (exp(input) - 1));
}

std::unique_ptr<Module> ELU::clone() const {
  return std::make_unique<ELU>(*this);
}

std::string ELU::prettyString() const {
  return "ELU (" + std::to_string(mAlpha_) + ")";
}

ThresholdReLU::ThresholdReLU(double threshold) : mThreshold_(threshold) {}

Variable ThresholdReLU::forward(const Variable& input) {
  auto mask = input >= mThreshold_;
  return input * mask;
}

std::unique_ptr<Module> ThresholdReLU::clone() const {
  return std::make_unique<ThresholdReLU>(*this);
}

std::string ThresholdReLU::prettyString() const {
  return "ThresholdReLU (" + std::to_string(mThreshold_) + ")";
}

GatedLinearUnit::GatedLinearUnit(int dim) : dim_(dim) {}

Variable GatedLinearUnit::forward(const Variable& input) {
  return gatedlinearunit(input, dim_);
}

std::unique_ptr<Module> GatedLinearUnit::clone() const {
  return std::make_unique<GatedLinearUnit>(*this);
}

std::string GatedLinearUnit::prettyString() const {
  return "GatedLinearUnit (" + std::to_string(dim_) + ")";
}

LogSoftmax::LogSoftmax(int dim /* = 0 */) : dim_(dim) {}

Variable LogSoftmax::forward(const Variable& input) {
  return logSoftmax(input, dim_);
}

std::unique_ptr<Module> LogSoftmax::clone() const {
  return std::make_unique<LogSoftmax>(*this);
}

std::string LogSoftmax::prettyString() const {
  return "LogSoftmax (" + std::to_string(dim_) + ")";
}

Swish::Swish(double beta /* = 1.0 */) : beta_(beta) {}

Variable Swish::forward(const Variable& input) {
  return swish(input, beta_);
}

std::unique_ptr<Module> Swish::clone() const {
  return std::make_unique<Swish>(*this);
}

std::string Swish::prettyString() const {
  return "Swish (" + std::to_string(beta_) + ")";
}

} // namespace fl
