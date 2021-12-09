/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include "flashlight/fl/nn/Utils.h"

#include "flashlight/fl/autograd/Utils.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {

int64_t numTotalParams(std::shared_ptr<fl::Module> module) {
  int64_t params = 0;
  for (auto& p : module->params()) {
    params += p.elements();
  }
  return params;
}

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

Tensor join(
    const std::vector<Tensor>& inputs,
    double padValue /* = 0.0 */,
    dim_t batchDim /* = -1 */) {
  if (inputs.empty()) {
    return Tensor();
  }

  Dim maxNumDims = 0;
  for (const auto& in : inputs) {
    if (in.ndim() > maxNumDims) {
      maxNumDims = in.ndim();
    }
  }

  if (batchDim > maxNumDims) {
    throw std::invalid_argument(
        "join: Batch dim is larger than the number "
        "of dims of the largest tensor");
  }

  Shape maxDims(std::vector<Dim>(maxNumDims, 1));
  fl::dtype type = inputs[0].type();
  bool isEmpty = true;
  for (const auto& in : inputs) {
    isEmpty = isEmpty && in.isEmpty();
    for (int d = 0; d < in.ndim(); ++d) {
      maxDims[d] = std::max(maxDims[d], in.dim(d));
      if (in.type() != type) {
        throw std::invalid_argument(
            "join: all arrays should of same type for join");
      }
    }
  }

  if (batchDim < 0) {
    batchDim = maxDims.ndim() - 1;
  }
  if (maxDims[batchDim] > 1) {
    throw std::invalid_argument(
        "join: no singleton dim available for batching");
  }
  maxDims[batchDim] = inputs.size();
  if (isEmpty) {
    // if empty arrays are provided (some element in maxDims is zero)
    // then af::constant will create array with sizes 0 and 1 on non-zeros dims,
    // e.g. maxDims = (0, 3, 4, 5) the af::constant(pad, maxDims)
    // will have size = (0, 1, 1, 1), not (0, 3, 4, 5)
    // To avoid this we directly create empty array here with correct sizes
    return Tensor(maxDims, type);
  }
  auto padSeq = fl::full(maxDims, padValue, type);
  std::vector<fl::Index> sel(std::max(maxNumDims, batchDim + 1), fl::span);
  for (int i = 0; i < inputs.size(); ++i) {
    for (int d = 0; d < maxNumDims; ++d) {
      sel[d] = fl::range(inputs[i].dim(d));
    }
    sel[batchDim] = fl::range(i, i);
    if (!inputs[i].isEmpty()) {
      padSeq(sel) = inputs[i];
    }
  }
  return padSeq;
}

} // namespace fl
