/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/LocationAttention.h"
#include "flashlight/pkg/speech/criterion/attention/Utils.h"

namespace fl::pkg::speech {

SimpleLocationAttention::SimpleLocationAttention(int convKernel) {
  Sequential pa;
  pa.add(Conv2D(1, 1, 1, convKernel, 1, 1, -1, -1));
  pa.add(Reorder({2, 0, 1, 3}));
  pa.add(ReLU());
  add(std::move(pa));
}

std::unique_ptr<Module> SimpleLocationAttention::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'SimpleLocationAttention'");
}

std::pair<Variable, Variable> SimpleLocationAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& prevAttn,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int U = state.dim(1);
  if (U > 1) {
    throw std::invalid_argument(
        prettyString() + " only works on single step forward");
  }

  int T = xEncoded.dim(1);
  int B = xEncoded.dim(2);

  // [1, seqlen, batchsize]
  auto innerProd = matmulTN(state, xEncoded);

  if (!prevAttn.isEmpty()) {
    auto addAttn = moddims(
        module(0)->forward({moddims(prevAttn, {1, T, 1, B})}).front(),
        {1, T, B});
    innerProd = innerProd + addAttn;
  }

  if (!logAttnWeight.isEmpty()) {
    if (logAttnWeight.shape() != innerProd.shape()) {
      throw std::invalid_argument(
          "SimpleLocationAttention: logAttnWeight has wong dimentions");
    }
    innerProd = innerProd + logAttnWeight;
  }
  if (!xEncodedSizes.isEmpty()) {
    innerProd = maskAttention(innerProd, xEncodedSizes);
  }
  // [1, seqlen, batchsize]
  auto attention = softmax(innerProd, 1);
  // [hiddendim, 1, batchsize]
  auto summaries = matmulNT(xEncoded, attention);
  return std::make_pair(attention, summaries);
}

std::string SimpleLocationAttention::prettyString() const {
  return "SimpleLocationBasedAttention";
}

LocationAttention::LocationAttention(int encDim, int convKernel) {
  Sequential pa;
  pa.add(Conv2D(1, encDim, 1, convKernel, 1, 1, -1, -1));
  pa.add(Reorder({2, 0, 1, 3}));
  pa.add(ReLU());
  add(std::move(pa));
}

std::unique_ptr<Module> LocationAttention::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'LocationAttention'");
}

std::pair<Variable, Variable> LocationAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& prevAttn,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int U = state.dim(1);
  if (U > 1) {
    throw std::invalid_argument(
        prettyString() + " only works on single step forward");
  }

  int H = xEncoded.dim(0);
  int T = xEncoded.dim(1);
  int B = xEncoded.dim(2);

  auto innerProd = matmulTN(state, xEncoded);

  if (!prevAttn.isEmpty()) {
    auto addAttn = moddims(
        module(0)->forward({moddims(prevAttn, {1, T, 1, B})}).front(),
        {H, T, B});
    innerProd = innerProd + matmulTN(state, addAttn);
  }

  if (!logAttnWeight.isEmpty()) {
    if (logAttnWeight.shape() != innerProd.shape()) {
      throw std::invalid_argument(
          "LocationAttention: logAttnWeight has wong dimentions");
    }
    innerProd = innerProd + logAttnWeight;
  }
  if (!xEncodedSizes.isEmpty()) {
    innerProd = maskAttention(innerProd, xEncodedSizes);
  }
  // [1, seqlen, batchsize]
  auto attention = softmax(innerProd, 1);
  // [hiddendim, 1, batchsize]
  auto summaries = matmulNT(xEncoded, attention);
  return std::make_pair(attention, summaries);
}

std::string LocationAttention::prettyString() const {
  return "LocationBasedAttention";
}

NeuralLocationAttention::NeuralLocationAttention(
    int encDim,
    int attnDim,
    int convChannel,
    int convKernel) {
  add(Linear(encDim, attnDim));
  add(Linear(encDim, attnDim, false));
  Sequential pa;
  pa.add(Conv2D(1, convChannel, 1, convKernel, 1, 1, -1, -1));
  pa.add(Reorder({2, 0, 1, 3}));
  pa.add(Linear(convChannel, attnDim, false));
  add(std::move(pa));
  add(Tanh());
  add(Linear(attnDim, 1, false));
}

std::unique_ptr<Module> NeuralLocationAttention::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'NeuralLocationAttention'");
}

std::pair<Variable, Variable> NeuralLocationAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& prevAttn,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int U = state.dim(1);
  if (U > 1) {
    throw std::invalid_argument(
        prettyString() + " only works on single step forward");
  }

  int T = xEncoded.dim(1);
  int B = xEncoded.dim(2);

  auto Hx = module(0)->forward({xEncoded}).front();
  auto tileHy = tile(module(1)->forward({state}).front(), {1, T, 1});

  // [1, seqlen, batchsize]
  auto hidden = Hx + tileHy;
  if (!prevAttn.isEmpty()) {
    auto addAttn = moddims(
        module(2)->forward({moddims(prevAttn, {1, T, 1, B})}).front(),
        {-1, T, B});
    hidden = hidden + addAttn;
  }
  hidden = module(3)->forward({hidden}).front();
  auto nnOut = module(4)->forward({hidden}).front();

  if (!logAttnWeight.isEmpty()) {
    if (logAttnWeight.shape() != nnOut.shape()) {
      throw std::invalid_argument(
          "NeuralLocationAttention: logAttnWeight has wong dimentions");
    }
    nnOut = nnOut + logAttnWeight;
  }

  if (!xEncodedSizes.isEmpty()) {
    nnOut = maskAttention(nnOut, xEncodedSizes);
  }
  // [1, seqlen, batchsize]
  auto attention = softmax(nnOut, 1);
  // [hiddendim, 1, batchsize]
  auto summaries = matmulNT(xEncoded, attention);
  return std::make_pair(attention, summaries);
}

std::string NeuralLocationAttention::prettyString() const {
  return "NeuralLocationBasedAttention";
}
} // namespace fl
