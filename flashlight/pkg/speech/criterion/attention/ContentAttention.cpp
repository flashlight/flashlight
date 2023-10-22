/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/ContentAttention.h"

#include <cmath>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/Utils.h"

namespace fl::pkg::speech {

std::unique_ptr<Module> ContentAttention::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'ContentAttention'");
}

std::pair<Variable, Variable> ContentAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& /* unused */,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int dim = xEncoded.dim(0);
  if (dim != (1 + ((keyValue_) ? 1 : 0)) * state.dim(0)) {
    throw std::invalid_argument(
        "ContentAttention: Invalid dimension for content attention");
  }
  auto keys = keyValue_ ? xEncoded(fl::range(0, dim / 2)) : xEncoded;
  auto values = keyValue_ ? xEncoded(fl::range(dim / 2, dim)) : xEncoded;
  // [targetlen, seqlen, batchsize]
  auto innerProd = matmulTN(state, keys) / std::sqrt(state.dim(0));
  if (!logAttnWeight.isEmpty()) {
    if (logAttnWeight.shape() != innerProd.shape()) {
      throw std::invalid_argument(
          "ContentAttention: logAttnWeight has wong dimentions");
    }
    innerProd = innerProd + logAttnWeight;
  }
  Tensor padMask;
  if (!xEncodedSizes.isEmpty()) {
    innerProd = maskAttention(innerProd, xEncodedSizes);
  }
  // [targetlen, seqlen, batchsize]
  auto attention = softmax(innerProd, 1);
  // [hiddendim, targetlen, batchsize]
  auto summaries = matmulNT(values, attention);
  return std::make_pair(attention, summaries);
}

std::string ContentAttention::prettyString() const {
  return "ContentBasedAttention";
}

NeuralContentAttention::NeuralContentAttention(int dim, int layers /* = 1 */) {
  Sequential net;
  net.add(ReLU());
  for (int i = 1; i < layers; i++) {
    net.add(Linear(dim, dim));
    net.add(ReLU());
  }
  net.add(Linear(dim, 1));
  add(std::move(net));
}

std::unique_ptr<Module> NeuralContentAttention::clone() const {
  throw std::runtime_error(
      "Cloning is unimplemented in Module 'NeuralContentAttention'");
}

std::pair<Variable, Variable> NeuralContentAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& /* unused */,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int U = state.dim(1);
  int H = xEncoded.dim(0);
  int T = xEncoded.dim(1);
  int B = xEncoded.dim(2);

  auto tileHx = tile(moddims(xEncoded, {H, 1, T, B}), {1, U, 1, 1});
  auto tileHy = tile(moddims(state, {H, U, 1, B}), {1, 1, T, 1});
  // [hiddendim, targetlen, seqlen, batchsize]
  auto hidden = tileHx + tileHy;
  // [targetlen, seqlen, batchsize]
  auto nnOut = moddims(module(0)->forward({hidden}).front(), {U, T, B});
  if (!logAttnWeight.isEmpty()) {
    if (logAttnWeight.shape() != nnOut.shape()) {
      throw std::invalid_argument(
          "ContentAttention: logAttnWeight has wong dimentions");
    }
    nnOut = nnOut + logAttnWeight;
  }

  if (!xEncodedSizes.isEmpty()) {
    nnOut = maskAttention(nnOut, xEncodedSizes);
  }
  // [targetlen, seqlen, batchsize]
  auto attention = softmax(nnOut, 1);
  // [hiddendim, targetlen, batchsize]
  auto summaries = matmulNT(xEncoded, attention);
  return std::make_pair(attention, summaries);
}

std::string NeuralContentAttention::prettyString() const {
  return "NeuralContentBasedAttention";
}
} // namespace fl
