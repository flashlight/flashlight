/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
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

  auto out = batchnorm(
      input,
      Variable(),
      Variable(),
      dummy_in_mean,
      dummy_in_var,
      featAxes_,
      true,
      0.0,
      epsilon_);

  if (affine_) {
    auto weight = tileAs(params_[0], out.dims());
    auto bias = tileAs(params_[1], out.dims());
    out = out * weight + bias;
  }

  return out;
}

void LayerNorm::initialize() {
  if (affine_) {
    auto wt = uniform(1, 0.0, 1.0, af::dtype::f32, true);
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
