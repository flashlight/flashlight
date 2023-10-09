/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/RNN.h"

#include <cmath>
#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

RNN::RNN(
    int input_size,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional /* false */,
    float drop_prob /* = 0.0 */)
    : inputSize_(input_size),
      hiddenSize_(hidden_size),
      numLayers_(num_layers),
      mode_(mode),
      bidirectional_(bidirectional),
      dropProb_(drop_prob) {
  initialize();
}

RNN::RNN(const RNN& other)
    : Module(other.copyParams()),
      inputSize_(other.inputSize_),
      hiddenSize_(other.hiddenSize_),
      numLayers_(other.numLayers_),
      mode_(other.mode_),
      bidirectional_(other.bidirectional_),
      dropProb_(other.dropProb_) {
  train_ = other.train_;
}

RNN& RNN::operator=(const RNN& other) {
  params_ = other.copyParams();
  train_ = other.train_;
  inputSize_ = other.inputSize_;
  hiddenSize_ = other.hiddenSize_;
  numLayers_ = other.numLayers_;
  mode_ = other.mode_;
  bidirectional_ = other.bidirectional_;
  dropProb_ = other.dropProb_;
  return *this;
}

void RNN::initialize() {
  int64_t n_params = detail::getNumRnnParams(
      inputSize_, hiddenSize_, numLayers_, mode_, bidirectional_);

  double stdv = std::sqrt(1.0 / (double)hiddenSize_);
  auto w = uniform({n_params}, -stdv, stdv, fl::dtype::f32, true);
  params_ = {w};
}

std::vector<Variable> RNN::forward(const std::vector<Variable>& inputs) {
  if (inputs.empty() || inputs.size() > 3) {
    throw std::invalid_argument("Invalid inputs size");
  }

  const auto& input = inputs[0];
  const auto& hiddenState = inputs.size() >= 2 ? inputs[1] : Variable();
  const auto& cellState = inputs.size() == 3 ? inputs[2] : Variable();

  float dropProb = train_ ? dropProb_ : 0.0;
  auto rnnRes =
      rnn(input,
          hiddenState.astype(input.type()),
          cellState.astype(input.type()),
          params_[0].astype(input.type()),
          hiddenSize_,
          numLayers_,
          mode_,
          bidirectional_,
          dropProb);

  std::vector<Variable> output(1, std::get<0>(rnnRes));
  if (inputs.size() >= 2) {
    output.push_back(std::get<1>(rnnRes));
  }
  if (inputs.size() == 3) {
    output.push_back(std::get<2>(rnnRes));
  }
  return output;
}

Variable RNN::forward(const Variable& input) {
  return forward(std::vector<Variable>{input}).front();
}

Variable RNN::operator()(const Variable& input) {
  return forward(input);
}

std::tuple<Variable, Variable> RNN::forward(
    const Variable& input,
    const Variable& hidden_state) {
  auto res = forward(std::vector<Variable>{input, hidden_state});
  return std::make_tuple(res[0], res[1]);
}

std::tuple<Variable, Variable> RNN::operator()(
    const Variable& input,
    const Variable& hidden_state) {
  return forward(input, hidden_state);
}

std::tuple<Variable, Variable, Variable> RNN::forward(
    const Variable& input,
    const Variable& hidden_state,
    const Variable& cell_state) {
  auto res = forward(std::vector<Variable>{input, hidden_state, cell_state});
  return std::make_tuple(res[0], res[1], res[2]);
}

std::tuple<Variable, Variable, Variable> RNN::operator()(
    const Variable& input,
    const Variable& hidden_state,
    const Variable& cell_state) {
  return forward(input, hidden_state, cell_state);
}

std::unique_ptr<Module> RNN::clone() const {
  return std::make_unique<RNN>(*this);
}

std::string RNN::prettyString() const {
  std::ostringstream ss;
  switch (mode_) {
    case RnnMode::RELU:
      ss << "RNN (relu)";
      break;
    case RnnMode::TANH:
      ss << "RNN (tanh)";
      break;
    case RnnMode::LSTM:
      ss << "LSTM";
      break;
    case RnnMode::GRU:
      ss << "GRU";
      break;
    default:
      break;
  }
  int output_size = bidirectional_ ? 2 * hiddenSize_ : hiddenSize_;
  ss << " (" << inputSize_ << "->" << output_size << ")";
  if (numLayers_ > 1) {
    ss << " (" << numLayers_ << "-layer)";
  }
  if (bidirectional_) {
    ss << " (bidirectional)";
  }
  if (dropProb_ > 0) {
    ss << " (dropout=" << dropProb_ << ")";
  }
  return ss.str();
}

} // namespace fl
