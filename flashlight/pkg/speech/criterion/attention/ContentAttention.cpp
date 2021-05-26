/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/ContentAttention.h"
#include <cmath>
#include "flashlight/pkg/speech/criterion/attention/Utils.h"

namespace fl {
namespace pkg {
namespace speech {

std::pair<Variable, Variable> ContentAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& /* unused */,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int dim = xEncoded.dims(0);
  if (dim != (1 + ((keyValue_) ? 1 : 0)) * state.dims(0)) {
    throw std::invalid_argument(
        "ContentAttention: Invalid dimension for content attention");
  }
  auto keys = keyValue_ ? xEncoded(af::seq(0, dim / 2 - 1)) : xEncoded;
  auto values = keyValue_ ? xEncoded(af::seq(dim / 2, dim - 1)) : xEncoded;
  // [targetlen, seqlen, batchsize]
  auto innerProd = matmulTN(state, keys) / std::sqrt(state.dims(0));
  if (!logAttnWeight.isempty()) {
    if (logAttnWeight.dims() != innerProd.dims()) {
      throw std::invalid_argument(
          "ContentAttention: logAttnWeight has wong dimentions");
    }
    innerProd = innerProd + logAttnWeight;
  }
  af::array padMask;
  if (!xEncodedSizes.isempty()) {
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
  add(net);
}

std::pair<Variable, Variable> NeuralContentAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& /* unused */,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int U = state.dims(1);
  int H = xEncoded.dims(0);
  int T = xEncoded.dims(1);
  int B = xEncoded.dims(2);

  auto tileHx = tile(moddims(xEncoded, {H, 1, T, B}), {1, U, 1, 1});
  auto tileHy = tile(moddims(state, {H, U, 1, B}), {1, 1, T, 1});
  // [hiddendim, targetlen, seqlen, batchsize]
  auto hidden = tileHx + tileHy;
  // [targetlen, seqlen, batchsize]
  auto nnOut = moddims(module(0)->forward({hidden}).front(), {U, T, B});
  if (!logAttnWeight.isempty()) {
    if (logAttnWeight.dims() != nnOut.dims()) {
      throw std::invalid_argument(
          "ContentAttention: logAttnWeight has wong dimentions");
    }
    nnOut = nnOut + logAttnWeight;
  }

  if (!xEncodedSizes.isempty()) {
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
} // namespace speech
} // namespace pkg
} // namespace fl
