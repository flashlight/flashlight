/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/MultiHeadAttention.h"
#include "flashlight/pkg/speech/criterion/attention/Utils.h"

#include <cmath>

namespace fl {
namespace pkg {
namespace speech {

MultiHeadContentAttention::MultiHeadContentAttention(
    int dim,
    int numHeads /* = 8 */,
    bool keyValue /* = false */,
    bool splitInput /* = false */)
    : numHeads_(numHeads), keyValue_(keyValue), splitInput_(splitInput) {
  if (splitInput && dim % numHeads != 0) {
    throw std::invalid_argument("Invalid dimensions");
  }

  if (!splitInput) {
    add(Linear(dim, dim)); // query
    add(Linear(dim, dim)); // key
    add(Linear(dim, dim)); // value
  }
  add(Linear(dim, dim));
}

std::pair<Variable, Variable> MultiHeadContentAttention::forwardBase(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& /* unused */,
    const Variable& logAttnWeight,
    const Variable& xEncodedSizes) {
  int hEncode = xEncoded.dims(0);
  int T = xEncoded.dims(1);
  int hState = state.dims(0);
  int U = state.dims(1);
  int B = state.dims(2);
  auto hiddenDim = hState / numHeads_;
  if (hEncode != (1 + keyValue_) * hState) {
    throw std::invalid_argument("Invalid input encoder dimension");
  }

  auto xEncodedKey =
      keyValue_ ? xEncoded(af::seq(0, hEncode / 2 - 1)) : xEncoded;
  auto xEncodedValue =
      keyValue_ ? xEncoded(af::seq(hEncode / 2, hEncode - 1)) : xEncoded;

  auto query = splitInput_ ? state : module(0)->forward({state})[0];
  auto key = splitInput_ ? xEncodedKey : module(1)->forward({xEncodedKey})[0];
  auto value =
      splitInput_ ? xEncodedValue : module(2)->forward({xEncodedValue})[0];

  query = moddims(reorder(query, 1, 0, 2), {U, hiddenDim, B * numHeads_});
  key = moddims(reorder(key, 1, 0, 2), {T, hiddenDim, B * numHeads_});
  value = moddims(reorder(value, 1, 0, 2), {T, hiddenDim, B * numHeads_});

  // [U, T, B * numHeads_]
  auto innerProd =
      matmulNT(query, key) / std::sqrt(static_cast<float>(hiddenDim));

  if (!logAttnWeight.isempty()) {
    auto tiledLogAttnWeight = tile(logAttnWeight, {1, 1, numHeads_});
    if (tiledLogAttnWeight.dims() != innerProd.dims()) {
      throw std::invalid_argument(
          "MultiHeadContentAttention: logAttnWeight has wong dimentions");
    }
    innerProd = innerProd + tiledLogAttnWeight;
  }

  if (!xEncodedSizes.isempty()) {
    innerProd = maskAttention(
        innerProd,
        moddims(
            tile(xEncodedSizes, {numHeads_, 1}), af::dim4(1, B * numHeads_)));
  }

  // [U, T, B * numHeads_]
  auto attention = softmax(innerProd, 1);
  // [U, hiddendim, B * numHeads_]
  auto summaries = matmul(attention, value);
  // [hiddendim * numHeads_, U, B];
  summaries = reorder(moddims(summaries, {U, hState, B}), 1, 0, 2);

  auto out_summaries = modules().back()->forward({summaries}).front();

  // [U * numHeads_, T, B]
  attention = moddims(
      reorder(moddims(attention, {U, T, numHeads_, B}), 0, 2, 1, 3),
      {U * numHeads_, T, B});
  return std::make_pair(attention, out_summaries);
}

std::string MultiHeadContentAttention::prettyString() const {
  return "MultiHeadContentAttention";
}
} // namespace speech
} // namespace pkg
} // namespace fl
