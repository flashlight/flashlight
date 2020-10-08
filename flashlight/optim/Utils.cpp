/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/flashlight/optim/Utils.h"

namespace fl {

double clipGradNorm(const std::vector<Variable>& parameters, double max_norm) {
  double grad_norm = 0.0;
  for (const auto& p : parameters) {
    if (!p.isGradAvailable()) {
      continue;
    }
    // ArrayFire v3.7.1 does not support computing the norm of an f16 tensor.
    // This cast can be removed when the support is added.
    // https://github.com/arrayfire/arrayfire/blob/v3.7.1/src/api/c/norm.cpp#L128
    if (p.grad().type() == af::dtype::f16) {
      p.grad() = p.grad().as(af::dtype::f32);
    }
    const auto& grad = p.grad().array();
    grad_norm += std::pow(af::norm(af::flat(grad)), 2);
  }
  grad_norm = std::sqrt(grad_norm);
  double scale = (max_norm / grad_norm);
  if (scale < 1.0) {
    for (auto& p : parameters) {
      if (!p.isGradAvailable()) {
        continue;
      }
      auto& grad = p.grad().array();
      grad = scale * grad;
      p.grad().array() = grad;
    }
  }
  return grad_norm;
}

} // namespace fl
