/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/nn/modules/LayerNorm.h"

#include "flashlight/autograd/Functions.h"
#include "flashlight/nn/Init.h"

namespace fl {

LayerNorm::LayerNorm(
    int feat_axis,
    double eps /*  = 1e-5*/,
    bool affine /*  = true*/)
    : LayerNorm(std::vector<int>({feat_axis}), eps, affine) {}

LayerNorm::LayerNorm(
    const std::vector<int>& feat_axes,
    double eps /* = 1e-5 */,
    bool affine /* = true*/)
    : featAxes_(feat_axes), epsilon_(eps), affine_(affine) {
  initialize();
}

Variable LayerNorm::forward(const Variable& input) {
  Variable dummy_in_mean, dummy_in_var;

  auto weight = Variable();
  auto bias = Variable();

  if (affine_) {
    af::dim4 tiledims(1, 1, 1, 1);
    for (int ax : featAxes_) {
      tiledims[ax] = input.dims(ax);
    }
    weight = tile(params_[0], tiledims);
    bias = tile(params_[1], tiledims);
  }

  auto out = batchnorm(
      input,
      weight,
      bias,
      dummy_in_mean,
      dummy_in_var,
      featAxes_,
      true,
      0.0,
      epsilon_);

  return out;
}

void LayerNorm::initialize() {
  if (affine_) {
    auto wt = constant(1.0, 1, af::dtype::f32, true);
    auto bs = constant(0.0, 1, af::dtype::f32, true);
    params_ = {wt, bs};
  }
}

std::string LayerNorm::prettyString() const {
  std::ostringstream ss;
  ss << "LayerNorm";
  ss << " ( axes : { ";
  for (auto x : featAxes_) {
    ss << x << " ";
  }
  ss << "} )";
  return ss.str();
}

} // namespace fl
