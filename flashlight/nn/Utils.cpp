/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/nn/Utils.h"

#include "flashlight/common/Utils.h"

namespace fl {

namespace detail {
size_t getNumRnnParams(
    int input_size,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional) {
  int bidir_mul = (bidirectional ? 2 : 1);
  size_t n_params =
      /* hidden-to-hidden */
      hidden_size * hidden_size * num_layers +
      /* hidden biases */
      hidden_size * num_layers +
      /* input-to-hidden */
      input_size * hidden_size +
      bidir_mul * (num_layers - 1) * hidden_size * hidden_size +
      /* input biases */
      hidden_size * num_layers;

  n_params *= bidir_mul;

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

std::pair<dim_t, dim_t> computeFans(af::dim4 dims) {
  dim_t fan_in, fan_out;
  auto ndims = dims.ndims();
  if (ndims <= 2) {
    fan_in = dims[1];
    fan_out = dims[0];
  } else {
    fan_out = dims[ndims - 1];
    fan_in = dims.elements() / dims[ndims - 1];
  }
  return {fan_in, fan_out};
}

int derivePadding(
    int inSz,
    int filterSz,
    int stride,
    int pad) {
  if (pad == static_cast<int>(PaddingMode::SAME)) {
    int newPad;
    if (inSz % stride == 0) {
      newPad = filterSz - stride;
    } else {
      newPad = filterSz - (inSz % stride);
    }
    newPad = (newPad + 1) / 2; // equal pad on both sides
    return std::max(newPad, 0);
  }

  return pad;
}
} // namespace detail
} // namespace fl
