/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include "flashlight/nn/Utils.h"

#include "flashlight/autograd/Utils.h"
#include "flashlight/common/Utils.h"

namespace fl {

bool allParamsClose(
    const Module& a,
    const Module& b,
    double absTolerance /* = 1e-5 */) {
  if (a.params().size() != b.params().size()) {
    return false;
  }
  const auto aParams = a.params();
  const auto bParams = b.params();
  for (int p = 0; p < aParams.size(); ++p) {
    if (!allClose(aParams[p], bParams[p], absTolerance)) {
      return false;
    }
  }
  return true;
}

namespace detail {
int64_t getNumRnnParams(
    int input_size,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional) {
  int bidir_mul = (bidirectional ? 2 : 1);

  int64_t i = input_size;
  int64_t h = hidden_size;
  int64_t n = num_layers;
  int64_t b = bidir_mul;

  int64_t n_params =
      /* hidden-to-hidden */
      h * h * n +
      /* hidden biases */
      h * n +
      /* input-to-hidden */
      i * h + b * (n - 1) * h * h +
      /* input biases */
      h * n;

  n_params *= b;

  switch (mode) {
    case RnnMode::LSTM:
      n_params *= 4;
      break;
    case RnnMode::GRU:
      n_params *= 3;
      break;
    case RnnMode::RELU:
    case RnnMode::TANH:
    default:
      break;
  }

  return n_params;
}
} // namespace detail

int derivePadding(int inSz, int filterSz, int stride, int pad, int dilation) {
  if (pad == static_cast<int>(PaddingMode::SAME)) {
    int newPad;
    if (inSz % stride == 0) {
      newPad = (filterSz - 1) * dilation - stride + 1;
    } else {
      newPad = (filterSz - 1) * dilation - (inSz % stride) + 1;
    }
    newPad = (newPad + 1) / 2; // equal pad on both sides
    return std::max(newPad, 0);
  }

  return pad;
}

af::array join(
    const std::vector<af::array>& inputs,
    double padValue /* = 0.0 */,
    dim_t batchDim /* = -1 */) {
  if (inputs.empty()) {
    return af::array();
  }
  af::dim4 maxDims;
  af::dtype type = inputs[0].type();
  for (const auto& in : inputs) {
    for (int d = 0; d < AF_MAX_DIMS; ++d) {
      maxDims[d] = std::max(maxDims[d], in.dims(d));
      if (in.type() != type) {
        throw std::invalid_argument("all arrays should of same type for join");
      }
    }
  }

  if (batchDim < 0) {
    batchDim = maxDims.ndims();
  }
  if (maxDims[batchDim] > 1) {
    throw std::invalid_argument("no singleton dim available for batching");
  }
  maxDims[batchDim] = inputs.size();
  auto padSeq = af::constant(padValue, maxDims, type);
  std::array<af::seq, AF_MAX_DIMS> sel;
  for (int i = 0; i < inputs.size(); ++i) {
    for (int d = 0; d < AF_MAX_DIMS; ++d) {
      sel[d] = af::seq(inputs[i].dims(d));
    }
    sel[batchDim] = af::seq(i, i);
    padSeq(sel[0], sel[1], sel[2], sel[3]) = inputs[i];
  }
  return padSeq;
}

} // namespace fl
